# SageAttention Autotune 修改

原SageAttention(1.0.6)固定了BLOCK_M=BLKQ BLOCK_N=BLKK，无法在单次运行中调整，现在 configs 支持调整 BLOCK_M 和 BLOCK_N。直接的好处是可以同时使SDXL和Anima达到最佳性能，无需关闭comfy后修改源文件再重开。经过精心调整configs还能使速度进一步上升。

与flashattn一致，使用系统变量`FLASH_ATTENTION_TRITON_AMD_AUTOTUNE`开启autotune，或者使用`FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON`手动指定参数。

由于搜索区间变大，直接使用 autotune 将极大延长启动时间，推荐使用调试脚本搜索最佳的configs，然后使 autotune 仅搜索特定的 configs。默认configs设置为780M(gfx1103)，配置中有680M(gfx1035)的configs。默认配置对应 SDXL 1024x1280x1.5HiRes 和 Anima 1280x1536。

# FlashAttention Autotune Configs
配合 `flash_attn-2.8.4-py3-none-win_amd64.whl` 和 `amd_aiter-0.0.0-py3-none-win_amd64.whl` 使用。

[原whl下载地址](https://github.com/0xDELUXA/flash-attention/releases/tag/v2.8.4_win-rocm)

替换 `\aiter\ops\triton\_triton_kernels\flash_attn_triton_amd\` 下同名文件，使性能显著提升。

既然RDNA3的int8并不快，而且可能存在精度问题，那么为什么不直接用FlashAttention呢？

780M(gfx1103)和680M(gfx1035)在aiter中没有被加入到RDNA类，导致使用fallback配置，性能没有被充分发挥。直接修改 `fwd_prefill.py` 和 `utils.py` 这两个文件就可以轻松提升性能。修改文件中已写入780M(gfx1103)和680M(gfx1035)的配置。

```
@triton.autotune(
    configs=fwd_prefill_autotune_configs,
    key=FWD_PREFILL_AUTOTUNE_KEYS,
    use_cuda_graph=False,
)
```

autotune这段中将use_cuda_graph从`True`改为`False`，即可让RDNA使用autotune。与SageAttention的configs搜索方法类似，将搜索到的配置写入文件，就能在SDXL下达到比SageAttention更高的性能（+10%），Anima下与SageAttention几乎相同的性能（-5%）。
