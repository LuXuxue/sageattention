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
        ("Anima01", 4096, 2048, 2048),
        ("Anima03", 4096, 8192, 2048),
        ("Anima04", 4096, 2048, 8192),
        ("SDXL05", 4096, 640, 640),
        ("SDXL07", 4096, 5120, 640),
        ("SDXL08", 4096, 640, 2560),
        ("SDXL11", 1024, 10240, 1280),
        ("SDXL12", 1024, 1280, 5120),
    ]

    print(f"{'Case':<8} | {'Time (ms)':<10} | {'TFLOPS':<6}")
    
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