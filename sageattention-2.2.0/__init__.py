from typing import Literal, Union, overload, TypeVar, Any
import torch
import torch.nn.functional as F
import importlib
import triton
import triton.language as tl
import json
import os
import logging
from collections.abc import Callable, Sequence

LOG2_E = 1.44269504088896340736
DEFAULT_PV_ACCUM_DTYPE = os.getenv("SAGEATTN_DEFAULT_PV_ACCUM_DTYPE", "fp32").lower()
if DEFAULT_PV_ACCUM_DTYPE not in ("fp32", "fp16", "fp16+fp32"):
    DEFAULT_PV_ACCUM_DTYPE = "fp32"
_logger = logging.getLogger(__name__)

arch = str(triton.runtime.driver.active.get_current_target().arch).strip()
env_config_json = os.environ.get('FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON')
if env_config_json:
    env_config = json.loads(env_config_json)
    _TRITON_CONFIGS = [
        triton.Config(
            {
                'BLOCK_M': env_config.get('BLOCK_M'),
                'BLOCK_N': env_config.get('BLOCK_N'),
                'waves_per_eu': env_config.get('waves_per_eu', None)
            },
            num_warps=env_config.get('num_warps'),
            num_stages=env_config.get('num_stages')
        )
    ]
elif arch == "gfx1103":
    _TRITON_CONFIGS = [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 6}, num_warps=8, num_stages=2), #Ainma
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'waves_per_eu': 3}, num_warps=2, num_stages=1), #SDXL
    ]
elif arch == "gfx1035":
    _TRITON_CONFIGS = [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 1}, num_warps=2, num_stages=2), #Anima
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'waves_per_eu': 2}, num_warps=4, num_stages=2), #SDXL
    ]
elif arch.startswith("gfx"):
    _TRITON_CONFIGS = [
        triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'waves_per_eu': waves}, num_warps=nw, num_stages=ns)
        for bm in [128, 64, 32]
        for bn in [64, 32, 16]
        if bm > bn
        for waves in [1, 2, 3, 4, 6]
        for nw in [2, 4, 8]
        for ns in [1, 2, 3, 4]
    ]
elif arch == "75":
    _TRITON_CONFIGS = [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16}, num_warps=2, num_stages=2), #Anima
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16}, num_warps=4, num_stages=2), #SDXL
    ]
else:
    _TRITON_CONFIGS = [
        triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn}, num_warps=nw, num_stages=ns)
        for bm in [128, 64, 32]
        for bn in [64, 32, 16]
        if bm > bn
        for nw in [2, 4, 8]
        for ns in [1, 2, 3, 4]
    ]

_TRITON_AUTOTUNE_CONFIGS = tuple(
    (cfg.kwargs['BLOCK_M'], cfg.kwargs['BLOCK_N'], cfg.kwargs.get('waves_per_eu', None), cfg.num_warps, cfg.num_stages)
    for cfg in _TRITON_CONFIGS
)
_TRITON_AUTOTUNE_CACHE: dict[object, tuple[int, int, int, int, int]] = {}

def _padded_head_dim(head_dim: int) -> int:
    if head_dim < 64: return 64
    if 64 < head_dim < 128: return 128
    if 128 < head_dim < 256: return 256
    if head_dim in (64, 128, 256): return head_dim
    raise ValueError(f"Unsupported head_dim: {head_dim}")

def _pad_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = q.size(-1)
    pad_to = _padded_head_dim(head_dim)
    if pad_to == head_dim:
        return head_dim, q, k, v
    padding = (0, pad_to - head_dim)
    return head_dim, F.pad(q, padding), F.pad(k, padding), F.pad(v, padding)

def _lse_correction(q: torch.Tensor, km: torch.Tensor, tensor_layout: str, head_dim_index: int) -> torch.Tensor:
    num_qo_heads = q.size(head_dim_index)
    num_kv_heads = km.size(head_dim_index)
    q_per_kv_heads = num_qo_heads // num_kv_heads
    km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=head_dim_index) if q_per_kv_heads > 1 else km
    if tensor_layout == "NHD":
        correction = torch.matmul(q.transpose(1, 2), km_broadcast.permute(0, 2, 3, 1)).squeeze(-1)
    else:
        correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1)
    return correction.to(torch.float32)

def _shared_memory_limit(device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    return getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)

def _logical_shape_autotune_key(q: torch.Tensor, k: torch.Tensor, tensor_layout: str) -> tuple[int, ...]:
    if tensor_layout == "NHD":
        batch_size, qo_len, num_qo_heads, head_dim = q.shape
        _, kv_len, num_kv_heads, _ = k.shape
    elif tensor_layout == "HND":
        batch_size, num_qo_heads, qo_len, head_dim = q.shape
        _, num_kv_heads, kv_len, _ = k.shape
    else:
        raise ValueError("tensor_layout must be 'NHD' or 'HND'.")
    return (
        batch_size, num_qo_heads, num_kv_heads,
        qo_len, kv_len, head_dim,
    )

def _tensor_stride_layout_key(tensor: torch.Tensor, tensor_layout: str) -> tuple[int, ...]:
    if tensor_layout == "NHD":
        logical_dims = ((0, 0), (2, 1), (1, 2))
    elif tensor_layout == "HND":
        logical_dims = ((0, 0), (1, 1), (2, 2))
    else:
        raise ValueError("tensor_layout must be 'NHD' or 'HND'.")
    stride_roles = sorted(
        ((tensor.stride(dim), role) for dim, role in logical_dims if tensor.size(dim) > 1), reverse=True
    )
    return tuple(role for _, role in stride_roles)

def _tensor_autotune_cache_key(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str, *extra: object,
) -> tuple[object, ...]:
    return (
        q.device.index, q.dtype,
        _logical_shape_autotune_key(q, k, tensor_layout),
        _tensor_stride_layout_key(q, tensor_layout),
        _tensor_stride_layout_key(k, tensor_layout),
        _tensor_stride_layout_key(v, tensor_layout),
        *extra,
    )

def _estimated_triton_smem_bytes(block_m: int, block_n: int, head_dim: int, attn_num_stages: int, is_causal: bool) -> int:
    int8_bytes = 1
    fp16_bytes = 2
    min_smem_bytes = 8 * 1024
    pipeline_prologue_stages = 1
    stage_bookkeeping_bytes = 4
    causal_mask_smem_slack_bytes = 16
    operand_tile_bytes = head_dim * (block_m + block_n)
    kv_pipeline_stage_bytes = head_dim * block_n * (int8_bytes + fp16_bytes)
    live_kv_pipeline_stages = max(attn_num_stages - pipeline_prologue_stages, 1)
    estimated = operand_tile_bytes + live_kv_pipeline_stages * (kv_pipeline_stage_bytes + stage_bookkeeping_bytes)
    estimated = max(estimated, min_smem_bytes)
    if is_causal: estimated += causal_mask_smem_slack_bytes
    return estimated

def _triton_config_is_valid(config: tuple[int, int, int, int, int], head_dim: int, is_causal: bool, device: torch.device) -> bool:
    block_m, block_n, _, _, attn_num_stages = config
    if is_causal and block_m % block_n != 0: return False
    head_dim = _padded_head_dim(head_dim)
    return _estimated_triton_smem_bytes(block_m, block_n, head_dim, attn_num_stages, is_causal) <= _shared_memory_limit(device)

ConfigT = TypeVar("ConfigT", bound=tuple[int, ...])

def _valid_configs_for_head_dim(
    candidates: Sequence[ConfigT], is_valid: Callable[[ConfigT, int, bool, torch.device], bool],
    head_dim: int, is_causal: bool, device: torch.device,
) -> tuple[ConfigT, ...]:
    configs = tuple(config for config in candidates if is_valid(config, head_dim, is_causal, device))
    if not configs: raise RuntimeError(f"No valid config for head_dim={head_dim} is_causal={is_causal}.")
    return configs

def _valid_triton_configs_for_head_dim(head_dim: int, is_causal: bool, device: torch.device) -> tuple[tuple[int, int, int, int, int], ...]:
    return _valid_configs_for_head_dim(_TRITON_AUTOTUNE_CONFIGS, _triton_config_is_valid, head_dim, is_causal, device)

def _valid_triton_configs(q: torch.Tensor, is_causal: bool) -> tuple[tuple[int, int, int, int, int], ...]:
    return _valid_triton_configs_for_head_dim(q.size(-1), is_causal, q.device)

def _eager_autotune_select(
    configs: Sequence[ConfigT], cache: dict[object, ConfigT], cache_key: object, benchmark: Callable[[ConfigT], object],
) -> ConfigT:
    if len(configs) == 1: return configs[0]
    cached = cache.get(cache_key)
    if cached is not None: return cached
    warmup_ms = max(1, int(os.environ.get("SAGEATTN_AUTOTUNE_WARMUP_MS", "25")))
    rep_ms = max(1, int(os.environ.get("SAGEATTN_AUTOTUNE_REP_MS", "100")))
    best_config = configs[0]
    best_ms = None
    for config in configs:
        ms = triton.testing.do_bench(lambda config=config: benchmark(config), warmup=warmup_ms, rep=rep_ms)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_config = config
    cache[cache_key] = best_config
    _logger.info("SageAttention cached autotune config %s for key %s", best_config, cache_key)
    return best_config

def _eager_triton_autotune_select(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str,
    is_causal: bool, pv_accum_dtype: str, smooth_k: bool, return_lse: bool,
) -> tuple[int, int, int, int, int]:
    configs = _valid_triton_configs(q, is_causal)
    key = _tensor_autotune_cache_key(q, k, v, tensor_layout, is_causal, pv_accum_dtype, smooth_k, return_lse)
    def benchmark(config):
        return _sageattn_triton_configured(q, k, v, tensor_layout, is_causal, pv_accum_dtype, smooth_k, return_lse, config)
    return _eager_autotune_select(configs, _TRITON_AUTOTUNE_CACHE, key, benchmark)

@triton.autotune(
    configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)],
    key=["L_BUCKET", "C", "BLK", "HAS_MEAN"],
)
@triton.jit
def quant_per_block_int8_kernel(
    Input, Mean, Output, Scale, L, L_BUCKET,
    stride_iz, stride_ih, stride_in, stride_mz, stride_mh, stride_mk,
    stride_oz, stride_oh, stride_on, stride_sz, stride_sh,
    sm_scale,
    C: tl.constexpr, BLK: tl.constexpr, HAS_MEAN: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk
    x = tl.load(input_ptrs, mask=offs_n[:, None] < L, eviction_policy='evict_first')
    x = x.to(tl.float32)
    if HAS_MEAN:
        mean_ptrs = Mean + off_b * stride_mz + off_h * stride_mh + offs_k * stride_mk
        mean = tl.load(mean_ptrs).to(tl.float32)
        x -= mean[None, :]
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.0
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L, eviction_policy='evict_first')
    tl.store(scale_ptrs, scale, eviction_policy='evict_first')

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, q_scale, kv_len,
    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, start_m,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, IS_CAUSAL: tl.constexpr, PV_ACCUM_FP32: tl.constexpr,
    offs_m: tl.constexpr, offs_n: tl.constexpr,
):
    if IS_CAUSAL:
        if STAGE == 1:
            lo, hi = 0, start_m * BLOCK_M
        else:
            lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    else:
        lo, hi = 0, kv_len

    if not IS_CAUSAL:
        num_full_blocks = kv_len // BLOCK_N
        for start_n in range(0, num_full_blocks * BLOCK_N, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k = tl.load(K_ptrs, eviction_policy='evict_first')
            k_scale = tl.load(K_scale_ptr, eviction_policy='evict_first')
            qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32) * (q_scale * k_scale)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            v = tl.load(V_ptrs, eviction_policy='evict_first')
            p = p.to(tl.float16)
            if PV_ACCUM_FP32:
                acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float32)
            else:
                acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float16)
            m_i = m_ij
            K_ptrs += BLOCK_N * stride_kn
            K_scale_ptr += 1
            V_ptrs += BLOCK_N * stride_vn
        start_n = num_full_blocks * BLOCK_N
        if start_n < kv_len:
            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs, mask=k_mask, other=0.0, eviction_policy='evict_first')
            k_scale = tl.load(K_scale_ptr, eviction_policy='evict_first')
            qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32) * (q_scale * k_scale)
            qk = tl.where(k_mask, qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            v_mask = offs_n[:, None] < (kv_len - start_n)
            v = tl.load(V_ptrs, mask=v_mask, other=0.0, eviction_policy='evict_first')
            p = p.to(tl.float16)
            if PV_ACCUM_FP32:
                acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float32)
            else:
                acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float16)
            m_i = m_ij
    else:
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs, mask=k_mask, eviction_policy='evict_first')
            k_scale = tl.load(K_scale_ptr, eviction_policy='evict_first')
            qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32) * (q_scale * k_scale)
            mask = k_mask
            if STAGE == 2:
                mask &= offs_m[:, None] >= (start_n + offs_n[None, :])
            qk += tl.where(mask, 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n), eviction_policy='evict_first')
            p = p.to(tl.float16)
            if PV_ACCUM_FP32:
                acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float32)
            else:
                acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float16)
            m_i = m_ij
            K_ptrs += BLOCK_N * stride_kn
            K_scale_ptr += 1
            V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out, Lse,
    stride_qz, stride_qh, stride_qn, stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn, stride_oz, stride_oh, stride_on,
    qo_len, kv_len,
    H: tl.constexpr, num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    RETURN_LSE: tl.constexpr, IS_CAUSAL: tl.constexpr, PV_ACCUM_FP32: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len, eviction_policy='evict_last')
    q_scale = tl.load(Q_scale_ptr, eviction_policy='evict_last')
    if IS_CAUSAL:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, start_m,
            BLOCK_M, HEAD_DIM, BLOCK_N, 1, IS_CAUSAL, PV_ACCUM_FP32, offs_m, offs_n,
        )
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, start_m,
            BLOCK_M, HEAD_DIM, BLOCK_N, 2, IS_CAUSAL, PV_ACCUM_FP32, offs_m, offs_n,
        )
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, start_m,
            BLOCK_M, HEAD_DIM, BLOCK_N, 0, IS_CAUSAL, PV_ACCUM_FP32, offs_m, offs_n,
        )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len), eviction_policy='evict_last')
    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len), eviction_policy='evict_first')

def per_block_int8(
    q: torch.Tensor, k: torch.Tensor, km: torch.Tensor | None = None,
    BLKQ: int = 32, BLKK: int = 16, sm_scale: float = 1.0, tensor_layout: str = "HND",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
        if km is not None: km = km.squeeze(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
        if km is not None: km = km.squeeze(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    has_mean = km is not None
    mean = km if has_mean else k
    stride_bz_m, stride_h_m, stride_k_m = (mean.stride(0), mean.stride(1), mean.stride(2)) if has_mean else (0, 0, 0)
    q_blocks = triton.cdiv(qo_len, BLKQ)
    k_blocks = triton.cdiv(kv_len, BLKK)
    q_scale = torch.empty((b, h_qo, q_blocks), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, k_blocks), device=q.device, dtype=torch.float32)
    grid = (q_blocks, h_qo, b)
    quant_per_block_int8_kernel[grid](
        q, q, q_int8, q_scale, qo_len, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q, 0, 0, 0,
        stride_bz_qo, stride_h_qo, stride_seq_qo, q_scale.stride(0), q_scale.stride(1),
        sm_scale=sm_scale, C=head_dim, BLK=BLKQ, HAS_MEAN=False,
    )
    grid = (k_blocks, h_kv, b)
    quant_per_block_int8_kernel[grid](
        k, mean, k_int8, k_scale, kv_len, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k, stride_bz_m, stride_h_m, stride_k_m,
        stride_bz_ko, stride_h_ko, stride_seq_ko, k_scale.stride(0), k_scale.stride(1),
        sm_scale=1.0, C=head_dim, BLK=BLKK, HAS_MEAN=has_mean,
    )
    return q_int8, q_scale, k_int8, k_scale

def _attn_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    q_scale: torch.Tensor, k_scale: torch.Tensor,
    tensor_layout: str = "HND", is_causal: bool = False, pv_accum_dtype: str = "fp32",
    BLOCK_M: int = 32, BLOCK_N: int = 16, attn_num_warps: int = 4, attn_num_stages: int = 3,
    waves_per_eu: int | None = None,
    output_dtype: torch.dtype = torch.float16, return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    if is_causal and qo_len != kv_len:
        raise ValueError("qo_len and kv_len must be equal for causal attention")
    if h_qo % h_kv != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    num_kv_groups = h_qo // h_kv
    lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device) if return_lse else torch.empty([0], dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    launch_kwargs = {
        "num_warps": attn_num_warps,
        "num_stages": attn_num_stages,
    }
    if waves_per_eu is not None and waves_per_eu > 0:
        launch_kwargs["waves_per_eu"] = waves_per_eu
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o, lse,
        stride_bz_q, stride_h_q, stride_seq_q, stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v, stride_bz_o, stride_h_o, stride_seq_o,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
        RETURN_LSE=return_lse, IS_CAUSAL=is_causal, PV_ACCUM_FP32=(pv_accum_dtype == "fp32"),
        **launch_kwargs
    )
    return o, lse

SageAttnResult = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]

@overload
def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str = "HND",
    is_causal: bool = False, pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True, return_lse: Literal[False] = False, attn_mask: object = None,
    **kwargs,
) -> torch.Tensor: ...

@overload
def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str = "HND",
    is_causal: bool = False, pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True, return_lse: Literal[True] = True, attn_mask: object = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]: ...

def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str = "HND",
    is_causal: bool = False, pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True, return_lse: bool = False, attn_mask: object = None,
    **kwargs,
) -> SageAttnResult:
    assert attn_mask is None
    config = _eager_triton_autotune_select(q, k, v, tensor_layout, is_causal, pv_accum_dtype, smooth_k, return_lse)
    return _sageattn_triton_configured(q, k, v, tensor_layout, is_causal, pv_accum_dtype, smooth_k, return_lse, config)

def _sageattn_triton_configured(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str,
    is_causal: bool, pv_accum_dtype: str, smooth_k: bool, return_lse: bool,
    triton_config: tuple[int, int, int | None, int, int],
) -> SageAttnResult:
    dtype = q.dtype
    if not q.is_cuda: raise ValueError("Input tensors must be CUDA tensors.")
    cast_back_to_fp32 = False
    if dtype == torch.float32:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        dtype = torch.float16
        cast_back_to_fp32 = True
    if dtype not in (torch.float16, torch.bfloat16): raise ValueError(f"Unsupported dtype: {dtype}")
    if q.device != k.device or q.device != v.device: raise ValueError("All tensors must be on the same device.")
    if q.dtype != k.dtype or q.dtype != v.dtype: raise ValueError("All tensors must have the same dtype.")
    if k.shape != v.shape: raise ValueError("k and v must have the same shape.")
    head_dim, q, k, v = _pad_qkv(q, k, v)
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Last dimension of q, k, and v must be contiguous.")
    sm_scale = head_dim**-0.5
    seq_dim_index = 1 if tensor_layout == "NHD" else 2
    head_dim_index = 2 if tensor_layout == "NHD" else 1
    km = k.mean(dim=seq_dim_index, keepdim=True) if smooth_k else None
    if pv_accum_dtype not in ("fp32", "fp16"):
        raise ValueError("pv_accum_dtype must be 'fp32' or 'fp16'.")
    block_m, block_n, waves_per_eu, attn_num_warps, attn_num_stages = triton_config
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q, k, km=km, BLKQ=block_m, BLKK=block_n, sm_scale=sm_scale * LOG2_E, tensor_layout=tensor_layout,
    )
    output, lse = _attn_forward(
        q_int8, k_int8, v.to(torch.float16), q_scale, k_scale,
        tensor_layout=tensor_layout, is_causal=is_causal, pv_accum_dtype=pv_accum_dtype,
        BLOCK_M=block_m, BLOCK_N=block_n,
        attn_num_warps=attn_num_warps, attn_num_stages=attn_num_stages,
        waves_per_eu=waves_per_eu,
        output_dtype=dtype, return_lse=return_lse,
    )
    output = output[..., :head_dim]
    if cast_back_to_fp32: output = output.to(torch.float32)
    if not return_lse:
        return output
    lse /= LOG2_E
    if smooth_k:
        assert km is not None
        lse += _lse_correction(q, km, tensor_layout, head_dim_index) * sm_scale
    if cast_back_to_fp32: lse = lse.to(torch.float32)
    return output, lse

def sageattn(
    q, k, v, tensor_layout: str = "HND", is_causal: bool = False,
    dropout_p: float = 0.0, scale=None, **kwargs,
):
    if dropout_p: raise ValueError("sageattn: dropout_p is not supported (must be 0.0)")
    if scale is not None: raise ValueError("sageattn: custom scale is not supported")
    return sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, **kwargs)