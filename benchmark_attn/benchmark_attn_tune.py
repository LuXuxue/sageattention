import torch
import gc
import sys
import os

os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
env_config_json = os.environ.get('FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON')
escaped_json = env_config_json.replace('"', '""')

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

if not HAS_FA and not HAS_SAGE:
    print("错误: 至少需要安装 flash_attn 或 sageattention 才能运行此脚本。")
    sys.exit(1)

# ==========================================
# 2. 核心计算函数
# ==========================================
def calculate_tflops(b, h_q, sq, sk, d, time_ms):
    """计算 Attention 的 TFLOPS (支持 GQA，计算量由 Query Heads 决定)"""
    flops = 4 * b * h_q * sq * sk * d
    tflops = flops / (time_ms / 1000.0) / 1e12
    return tflops

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
# 3. 定义测试用例
# ==========================================
test_cases = [
    # 1. SDXL (MHA: h_q == h_kv)
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
    
    # 2. Anima (MHA: h_q == h_kv)
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

    for name, b, h_q, h_kv, sq, sk, d in test_cases:
        q = torch.randn(b, sq, h_q, d, device=device, dtype=dtype)
        k = torch.randn(b, sk, h_kv, d, device=device, dtype=dtype)
        v = torch.randn(b, sk, h_kv, d, device=device, dtype=dtype)
        
        # --- 测试 FlashAttention ---
        if HAS_FA:
            def fa_func(q, k, v):
                return flash_attn_func(q, k, v, causal=False)
            
            fa_time = benchmark(fa_func, q, k, v)
            fa_tflops = calculate_tflops(b, h_q, sq, sk, d, fa_time)
            print(f"\"{escaped_json}\",{name},{fa_time:.3f},{fa_tflops:.2f},FlashAttn")
            
        # --- 测试 SageAttention ---
        if HAS_SAGE:
            def sage_func(q, k, v):
                try:
                    return sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
                except TypeError:
                    return sageattn(q, k, v)
            
            sage_time = benchmark(sage_func, q, k, v)
            sage_tflops = calculate_tflops(b, h_q, sq, sk, d, sage_time)
            print(f"\"{escaped_json}\",{name},{sage_time:.3f},{sage_tflops:.2f},SageAttn")
        
        # 清理显存
        del q, k, v
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_benchmarks()
