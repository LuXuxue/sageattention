@echo off
chcp 65001 >nul
> benchmark_attn_config.txt (
    for %%m in (128 64 32) do (
        for %%n in (64 32 16) do (
            for %%w in (0 1 2 3 4 6) do (
                for %%r in (2 4 8) do (
                    for %%s in (1 2) do (
                        echo {"BLOCK_M":%%m,"BLOCK_N":%%n,"waves_per_eu":%%w,"PRE_LOAD_V":false,"num_warps":%%r,"num_stages":%%s}
                    )
                )
            )
        )
    )
)
