import torch, math
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len,
K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
start_m,
BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
MIN_BLK_N: tl.constexpr
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // MIN_BLK_N
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
        
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = (offs_n[None, :] < (kv_len - start_n)) & ((tl.arange(0, 128) < 96)[:, None])
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        v = tl.load(V_ptrs, mask=(offs_n[:, None] < (kv_len - start_n)) & ((tl.arange(0, 128) < 96)[None, :]))
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float32)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += (BLOCK_N // MIN_BLK_N)
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 6}, num_warps=8, num_stages=2), #gfx1103 Ainma Self
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 1}, num_warps=2, num_stages=2), #gfx1103 Anima Cross
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'waves_per_eu': 2}, num_warps=2, num_stages=1), #gfx1103 SDXL Self
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'waves_per_eu': 6}, num_warps=4, num_stages=1), #gfx1103 SDXL Cross
#    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 1}, num_warps=2, num_stages=2), #gfx1035 Anima
#    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'waves_per_eu': 2}, num_warps=2, num_stages=2), #gfx1035 SDXL
#    triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'waves_per_eu': waves}, num_warps=nw, num_stages=ns)
#    for bm in [128, 64, 32]
#    for bn in [64, 32, 16]
#    if bm > bn
#    for waves in [1, 2, 3, 4, 6]
#    for nw in [2, 4, 8]
#    for ns in [1, 2]
]

@triton.autotune(
    list(configs),
    key=['qo_len', 'kv_len', 'h_qo']
)
@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out,
stride_qz, stride_qh, stride_qn,
stride_kz, stride_kh, stride_kn,
stride_vz, stride_vh, stride_vn,
stride_oz, stride_oh, stride_on,
H, num_kv_groups, qo_len, kv_len,
HEAD_DIM: tl.constexpr,
BLOCK_M: tl.constexpr,
BLOCK_N: tl.constexpr,
STAGE: tl.constexpr,
MIN_BLK_M: tl.constexpr,
MIN_BLK_N: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = (off_hz // H).to(tl.int64)
    off_h = (off_hz % H).to(tl.int64)
    
    q_scale_offset = off_hz * tl.cdiv(qo_len, MIN_BLK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, MIN_BLK_N)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m * (BLOCK_M // MIN_BLK_M)
    
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < qo_len) & ((tl.arange(0, 128) < 96)[None, :]))
    q_scale = tl.load(Q_scale_ptr)
    
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n, MIN_BLK_N
                                    )
    acc, l_i, _ = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    2, offs_m, offs_n, MIN_BLK_N
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len) & ((tl.arange(0, 128) < 96)[None, :]))

def forward(q, k, v, q_scale, k_scale, tensor_layout="HND", output_dtype=torch.float16):
    stage = 3
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

    assert qo_len == kv_len, "qo_len and kv_len must be equal for causal attention"

    num_kv_groups = h_qo // h_kv
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

    grid = lambda META: (triton.cdiv(qo_len, META['BLOCK_M']), b * h_qo, 1)
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,  
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o, 
        h_qo, num_kv_groups, 
        qo_len, kv_len,
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        MIN_BLK_M=32,
        MIN_BLK_N=16)
        
#    best_config = getattr(_attn_fwd, 'best_config', None)
#    if best_config is not None:
#        config_kwargs = best_config.kwargs if hasattr(best_config, 'kwargs') else best_config.all_kwargs()
#        bm = config_kwargs.get('BLOCK_M')
#        bn = config_kwargs.get('BLOCK_N')
#        waves = config_kwargs.get('waves_per_eu')
#        num_warps = best_config.num_warps
#        num_stages = best_config.num_stages
#        print(f"[Autotune Best Config] [attn_qk_int8_per_block_h96_causal] BLOCK_M: {bm}, BLOCK_N: {bn}, waves_per_eu: {waves}, num_warps: {num_warps}, num_stages: {num_stages}")

    return o