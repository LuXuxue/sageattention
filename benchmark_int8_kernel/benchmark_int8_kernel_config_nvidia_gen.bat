@echo off
chcp 65001 >nul
> benchmark_int8_kernel_config.txt (
    for %%m in (128 64) do (
        for %%n in (256 128 64 32) do (
            for %%k in (128 64 32) do (
                for %%g in (4 8) do (
                    for %%r in (2 4 8) do (
                        for %%s in (2 3 4) do (
                            echo {"BLOCK_M":%%m,"BLOCK_N":%%n,"BLOCK_K":%%k,"group_size_m":%%g,"num_warps":%%r,"num_stages":%%s}
                        )
                    )
                )
            )
        )
    )
)