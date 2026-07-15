import torch
import torch.nn.functional as F
import gc
import sys
import os

os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'

# ==========================================
# 1. 导入与兼容性处理
# ==========================================
try:
    from flash_attn import flash_attn_func
    HAS_FA = True
except ImportError:
    HAS_FA = False
    print("[Warning] flash_attn 未安装或未找到。")

try:
    from sageattention import sageattn
    HAS_SAGE = True
except ImportError:
    try:
        from sageattention.core import sageattn
        HAS_SAGE = True
    except ImportError:
        HAS_SAGE = False
        print("[Warning] sageattention 未安装或未找到。")

try:
    from torch.nn.attention.flex_attention import flex_attention
    HAS_FLEX = True
    # 按照要求，使用 default 模式进行编译 (不进行 max-autotune 搜索)
    compiled_flex = torch.compile(flex_attention, mode="default")
except ImportError:
    HAS_FLEX = False
    print("[Warning] flex_attention 未找到 (需要 PyTorch 2.5+)。")

if not HAS_FA and not HAS_SAGE and not HAS_FLEX:
    print("错误: 至少需要安装 flash_attn, sageattention 或拥有支持 flex_attention 的 PyTorch 才能运行此脚本。")
    sys.exit(1)

# ==========================================
# 2. 核心计算函数
# ==========================================
def calculate_tflops(b, h_q, sq, sk, d, time_ms):
    """计算 Attention 的 TFLOPS (支持 GQA，计算量由 Query Heads 决定)"""
    flops = 4 * b * h_q * sq * sk * d
    tflops = flops / (time_ms / 1000.0) / 1e12
    return tflops

def get_sdpa_reference(q, k, v):
    """使用 PyTorch 原生 SDPA (FP32) 计算高精度 Reference (Ground Truth)"""
    # SDPA 期望的输入形状为 [Batch, Heads, SeqLen, Dim]
    # 原生 SDPA 支持 GQA 广播，无需手动 repeat KV
    q_sdpa = q.float().permute(0, 2, 1, 3)
    k_sdpa = k.float().permute(0, 2, 1, 3)
    v_sdpa = v.float().permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
    return out.permute(0, 2, 1, 3).half()

def calc_error(out, ref):
    """计算精度误差指标"""
    max_err = (out - ref).abs().max().item()
    mse = ((out - ref) ** 2).mean().item()
    cos_sim = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    return max_err, mse, cos_sim

def benchmark(func, q, k, v, warmup=10, iters=50):
    """GPU 精确计时"""
    for _ in range(warmup):
        _ = func(q, k, v)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        _ = func(q, k, v)
    end_event.record()
    
    torch.cuda.synchronize()
    time_ms = start_event.elapsed_time(end_event) / iters
    return time_ms

# ==========================================
# 3. 定义测试用例 (Format: name, b, h_q, h_kv, sq, sk, d)
# ==========================================
test_cases = [
    # 1. SDXL (MHA: h_q == h_kv)(Self1=长/16×宽/16, Self2=Self1/4。例：1024×1280 -> 1024/16*1280/16=5120)
    ("SDXL01", 1, 10, 10, 4096, 4096, 64),
    ("SDXL02", 1, 10, 10, 4096, 77, 64),
    ("SDXL03", 1, 10, 10, 4096, 154, 64),
    ("SDXL04", 1, 20, 20, 1024, 1024, 64),
    ("SDXL05", 1, 20, 20, 1024, 77, 64),
    ("SDXL06", 1, 20, 20, 1024, 154, 64),
    ("SDXL07", 1, 10, 10, 9216, 9216, 64),
    ("SDXL08", 1, 10, 10, 9216, 77, 64),
    ("SDXL09", 1, 10, 10, 9216, 154, 64),
    ("SDXL10", 1, 20, 20, 2304, 2304, 64),
    ("SDXL11", 1, 20, 20, 2304, 77, 64),
    ("SDXL12", 1, 20, 20, 2304, 154, 64),
    
    # 2. Anima (MHA: h_q == h_kv)(Self=长/16×宽/16。例：1280×1536 -> 1280/16*1536/16=7680)
    ("Anima01", 1, 16, 16, 4096, 4096, 128),
    ("Anima02", 1, 16, 16, 4096, 512, 128),
    ("Anima03", 1, 16, 16, 9216, 9216, 128),
    ("Anima04", 1, 16, 16, 9216, 512, 128),
    
    # 3. Krea2 (GQA: h_q = 48, h_kv = 12)
    #("Krea01", 1, 48, 12, 4213, 4213, 128),
    #("Krea02", 1, 48, 12, 117, 117, 128),
]

# ==========================================
# 4. 执行测试主循环
# ==========================================
def run_benchmarks():
    device = "cuda"
    dtype = torch.float16
    THRESHOLD = 0.05  # 设定 0.05 为宽松误差阈值
    
    for name, b, h_q, h_kv, sq, sk, d in test_cases:
        # 分离初始化 Q 与 K/V 的 Heads 数量以支持 GQA
        q = torch.randn(b, sq, h_q, d, device=device, dtype=dtype)
        k = torch.randn(b, sk, h_kv, d, device=device, dtype=dtype)
        v = torch.randn(b, sk, h_kv, d, device=device, dtype=dtype)
        
        try:
            ref_out = get_sdpa_reference(q, k, v)
        except RuntimeError as e:
            print(f"[{name}] OOM during SDPA Reference calculation: {e} ")
            continue
            
        # --- 测试 PyTorch SDPA (作为 Baseline 测速) ---
        def sdpa_func(q, k, v):
            q_p = q.permute(0, 2, 1, 3)
            k_p = k.permute(0, 2, 1, 3)
            v_p = v.permute(0, 2, 1, 3)
            return F.scaled_dot_product_attention(q_p, k_p, v_p).permute(0, 2, 1, 3)
            
        sdpa_time = benchmark(sdpa_func, q, k, v)
        sdpa_tflops = calculate_tflops(b, h_q, sq, sk, d, sdpa_time)
        print(f"{name: <15} | {'SDPA(Base)': <10} | {sdpa_time: <8.3f} | {sdpa_tflops: <6.2f} | {'Baseline': <6} | {'-': <8} | {'-': <10} | {'-': <8} | {'-': <6} ")
        
        # --- 测试 FlashAttention ---
        fa_out = None
        if HAS_FA:
            def fa_func(q, k, v):
                return flash_attn_func(q, k, v, causal=False)
            try:
                fa_time = benchmark(fa_func, q, k, v)
                fa_out = fa_func(q, k, v)
                fa_tflops = calculate_tflops(b, h_q, sq, sk, d, fa_time)
                fa_max, fa_mse, fa_cos = calc_error(fa_out, ref_out)
                speedup_fa = sdpa_time / fa_time
                status = "FAIL " if fa_max > THRESHOLD else "OK   "
                print(f"{name: <15} | {'FlashAttn': <10} | {fa_time: <8.3f} | {fa_tflops: <6.2f} | {speedup_fa: <8.2f}x| {fa_max: <8.6f} | {fa_mse: <10.8f} | {fa_cos: <8.6f} | {status: <6} ")
            except Exception as e:
                print(f"{name: <15} | {'FlashAttn': <10} | {'Error': <8} | {'-': <6} | {'-': <8} | {'-': <8} | {'-': <10} | {'-': <8} | {str(e)[:20]: <6} ")
        
        # --- 测试 SageAttention ---
        sage_out = None
        if HAS_SAGE:
            def sage_func(q, k, v):
                try:
                    return sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
                except TypeError:
                    return sageattn(q, k, v)
            try:
                sage_time = benchmark(sage_func, q, k, v)
                sage_out = sage_func(q, k, v)
                sage_tflops = calculate_tflops(b, h_q, sq, sk, d, sage_time)
                sage_max, sage_mse, sage_cos = calc_error(sage_out, ref_out)
                speedup_sage = sdpa_time / sage_time
                status = "FAIL " if sage_max > THRESHOLD else "OK   "
                print(f"{name: <15} | {'SageAttn': <10} | {sage_time: <8.3f} | {sage_tflops: <6.2f} | {speedup_sage: <8.2f}x| {sage_max: <8.6f} | {sage_mse: <10.8f} | {sage_cos: <8.6f} | {status: <6} ")
            except Exception as e:
                print(f"{name: <15} | {'SageAttn': <10} | {'Error': <8} | {'-': <6} | {'-': <8} | {'-': <8} | {'-': <10} | {'-': <8} | {str(e)[:20]: <6} ")

        # --- 测试 FlexAttention ---
        flex_out = None
        if HAS_FLEX:
            def flex_func(q, k, v):
                # Flex Attention 期望输入形状为 [Batch, Heads, SeqLen, Dim]
                q_p = q.permute(0, 2, 1, 3)
                k_p = k.permute(0, 2, 1, 3)
                v_p = v.permute(0, 2, 1, 3)
                out = compiled_flex(q_p, k_p, v_p)
                return out.permute(0, 2, 1, 3)
                
            try:
                flex_time = benchmark(flex_func, q, k, v)
                flex_out = flex_func(q, k, v)
                flex_tflops = calculate_tflops(b, h_q, sq, sk, d, flex_time)
                flex_max, flex_mse, flex_cos = calc_error(flex_out, ref_out)
                speedup_flex = sdpa_time / flex_time
                status = "FAIL " if flex_max > THRESHOLD else "OK   "
                print(f"{name: <15} | {'FlexAttn': <10} | {flex_time: <8.3f} | {flex_tflops: <6.2f} | {speedup_flex: <8.2f}x| {flex_max: <8.6f} | {flex_mse: <10.8f} | {flex_cos: <8.6f} | {status: <6} ")
            except Exception as e:
                print(f"{name: <15} | {'FlexAttn': <10} | {'Error': <8} | {'-': <6} | {'-': <8} | {'-': <8} | {'-': <10} | {'-': <8} | {str(e)[:20]: <6} ")
        
        # 清理显存
        del q, k, v, ref_out
        if fa_out is not None: del fa_out
        if sage_out is not None: del sage_out
        if flex_out is not None: del flex_out
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print(f"{'Shape': <15} | {'Backend': <10} | {'Time(ms)': <8} | {'TFLOPS': <6} | {'Speedup': <8} | {'MaxErr': <8} | {'MSE': <10} | {'CosSim': <8} | {'Status': <6}  ")
    run_benchmarks()