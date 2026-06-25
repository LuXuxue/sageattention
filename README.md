# SageAttention Autotune 修改
原SageAttention固定了BLOCK_M=BLKQ BLOCK_N=BLKK，无法在单次运行中调整，现在 configs 支持调整 BLOCK_M 和 BLOCK_N。
直接的好处是可以同时使SDXL和Anima达到最佳性能，无需关闭comfy后修改源文件再重开。经过精心调整configs还能使速度进一步上升。
由于搜索区间变大，直接使用 autotune 将极大延长启动时间，推荐使用调试脚本搜索最佳的configs，然后使 autotune 仅搜索特定的 configs。
默认configs设置为780M(gfx1103)，配置中有680M(gfx1035)的configs。默认配置对应 SDXL 1024x1280x1.5HiRes 和 Anima 1280x1536。
**sk过短时，SDXL CrossAttn可能误差爆炸，尚未找到解决方法。**
若需进行调试（开启autotune），请将已开启的configs进行注释，并取消下面这段注释。
```
#    triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'waves_per_eu': waves}, num_warps=nw, num_stages=ns)
#    for bm in [128, 64, 32]
#    for bn in [64, 32, 16]
#    if bm > bn
#    for waves in [1, 2, 3, 4, 6]
#    for nw in [2, 4, 8]
#    for ns in [1, 2]
```
以下这段可以显示搜索到的config，记下并加入到configs后关闭autotune即可达到最佳性能。
```
#    best_config = getattr(_attn_fwd, 'best_config', None)
#    if best_config is not None:
#        config_kwargs = best_config.kwargs if hasattr(best_config, 'kwargs') else best_config.all_kwargs()
#        bm = config_kwargs.get('BLOCK_M')
#        bn = config_kwargs.get('BLOCK_N')
#        waves = config_kwargs.get('waves_per_eu')
#        num_warps = best_config.num_warps
#        num_stages = best_config.num_stages
#        print(f"[Autotune Best Config] [attn_qk_int8_per_block] BLOCK_M: {bm}, BLOCK_N: {bn}, waves_per_eu: {waves}, num_warps: {num_warps}, num_stages: {num_stages}")
```

# FlashAttention Autotune Configs
配合 `flash_attn-2.8.4-py3-none-win_amd64.whl` 和 `amd_aiter-0.0.0-py3-none-win_amd64.whl` 使用
[下载地址](https://github.com/0xDELUXA/flash-attention/releases/tag/v2.8.4_win-rocm)
替换 `\aiter\ops\triton\_triton_kernels\flash_attn_triton_amd\` 下同名文件，使性能显著提升。
既然RDNA3的int8并不快，而且存在精度问题，那么为什么不直接用FlashAttention。
780M(gfx1103)和680M(gfx1035)在aiter中没有被加入到RDNA类，导致使用fallback配置，性能没有被充分发挥。
直接修改 `fwd_prefill.py` 和 `utils.py` 这两个文件就可以轻松提升性能。
```
@triton.autotune(
    configs=fwd_prefill_autotune_configs,
    key=FWD_PREFILL_AUTOTUNE_KEYS,
    use_cuda_graph=False,
)
```
autotune这段中将use_cuda_graph从`True`改为`False`，即可让RDNA使用autotune。
与SageAttention的configs搜索方法类似，将搜索到的配置写入文件，就能在SDXL下达到比SageAttention更高的性能（+10%），Anima下与SageAttention几乎相同的性能（-5%）。
推荐与[ComfyUI-FeatherOps](https://github.com/woct0rdho/ComfyUI-FeatherOps)一并使用
修改文件中已写入780M(gfx1103)和680M(gfx1035)的配置。
