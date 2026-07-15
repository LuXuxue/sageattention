import torch
import triton
import time

from comfy_kitchen.backends.triton.quantization import _int8_matmul_dequant_per_row_kernel

def benchmark_kernel(M, N, K, warmup=10, iters=50):
    """
    对 _int8_matmul_dequant_per_row_kernel 进行测速
    """
    # 1. 准备输入数据 (模拟实际推理时的张量形状与类型)
    # a: [M, K] int8 (激活值)
    # b: [N, K] int8 (权重值)
    # a_scale: [M, 1] float32 (逐行激活缩放因子)
    # b_scale: [N, 1] float32 (逐行权重缩放因子)
    # bias: [N] bfloat16
    # c: [M, N] bfloat16 (输出)
    
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
    b = torch.randint(-128, 127, (N, K), dtype=torch.int8, device='cuda')
    a_scale = torch.randn(M, 1, dtype=torch.float32, device='cuda')
    b_scale = torch.randn(N, 1, dtype=torch.float32, device='cuda')
    bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
    c = torch.empty((M, N), dtype=torch.bfloat16, device='cuda')

    # 定义 Grid 划分策略
    def grid(meta):
        return (triton.cdiv(M, meta["block_m"]) * triton.cdiv(N, meta["block_n"]),)

    # 触发 Autotune (第一次调用会遍历所有 config 寻找最优解并缓存)
    _int8_matmul_dequant_per_row_kernel[grid](
        a, b, c,
        a_scale, b_scale, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(0),  # 注意: b 的形状是 [N, K]，因此 stride(1) 对应 K 维度，stride(0) 对应 N 维度
        c.stride(0), c.stride(1),
        has_bias=True
    )
    torch.cuda.synchronize()

    # Warmup 阶段
    for _ in range(warmup):
        _int8_matmul_dequant_per_row_kernel[grid](
            a, b, c,
            a_scale, b_scale, bias,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(1), b.stride(0),
            c.stride(0), c.stride(1),
            has_bias=True
        )
    torch.cuda.synchronize()

    # Benchmark 阶段
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _int8_matmul_dequant_per_row_kernel[grid](
            a, b, c,
            a_scale, b_scale, bias,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(1), b.stride(0),
            c.stride(0), c.stride(1),
            has_bias=True
        )
    end_event.record()
    torch.cuda.synchronize()

    # 计算平均耗时 (ms)
    elapsed_time_ms = start_event.elapsed_time(end_event) / iters
    
    # 计算 INT8 矩阵乘法的 TFLOPS (FLOPs = 2 * M * N * K)
    flops = 2 * M * N * K
    tflops = (flops / (elapsed_time_ms * 1e-3)) / 1e12
    
    return elapsed_time_ms, tflops

def main():
    # 定义测试用例 (Case Name, M, N, K)
    test_cases = [
        ("LLM01", 816, 1024, 4096),
        ("LLM02", 816, 4096, 4096),
        ("LLM03", 816, 12288, 4096),
        ("LLM04", 816, 4096, 12288),
        ("LLM05", 1, 1024, 4096),
        ("LLM06", 1, 4096, 4096),
        ("LLM07", 1, 12288, 4096),
        ("LLM08", 1, 4096, 12288),
        ("Anima01", 8192, 2048, 2048),
        ("Anima02", 1024, 2048, 1024),
        ("Anima03", 8192, 8192, 2048),
        ("Anima04", 8192, 2048, 8192),
        ("Anima05", 8192, 64, 2048),
        ("Anima06", 18432, 2048, 2048),
        ("Anima07", 18432, 8192, 2048),
        ("Anima08", 18432, 2048, 8192),
        ("Anima09", 18432, 64, 2048),
        ("Anima10", 18432, 2048, 2048),
        ("SDXL01", 2, 1280, 2816),
        ("SDXL02", 2, 1280, 1280),
        ("SDXL03", 2, 320, 1280),
        ("SDXL04", 2, 640, 1280),
        ("SDXL05", 8192, 640, 640),
        ("SDXL06", 308, 640, 2048),
        ("SDXL07", 8192, 5120, 640),
        ("SDXL08", 8192, 640, 2560),
        ("SDXL09", 2048, 1280, 1280),
        ("SDXL10", 308, 1280, 2048),
        ("SDXL11", 2048, 10240, 1280),
        ("SDXL12", 2048, 1280, 5120),
        ("SDXL13", 18432, 640, 640),
        ("SDXL14", 18432, 5120, 640),
        ("SDXL15", 18432, 640, 2560),
        ("SDXL16", 4608, 1280, 1280),
        ("SDXL17", 4608, 10240, 1280),
        ("SDXL18", 4608, 1280, 5120),
    ]

    print(f"{'Case':<8} | {'Time':<10} | {'TFLOPS':<6}")
    
    for name, M, N, K in test_cases:
        try:
            time_ms, tflops = benchmark_kernel(M, N, K)
            print(f"{name:<8} | {time_ms:<10.4f} | {tflops:<6.2f}")
        except Exception as e:
            print(f"{name:<8} | ERROR: {str(e)}")

if __name__ == "__main__":
    # 确保在 CUDA 环境下运行
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run this script on a machine with a CUDA-enabled GPU.")
    
    main()