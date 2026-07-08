@echo off
setlocal enabledelayedexpansion
if exist "%~dp0python_embeded" (set python_embeded_path=%~dp0python_embeded) else (set python_embeded_path=E:\python_embeded)
REM set SAGEATTN_DEFAULT_PV_ACCUM_DTYPE=fp16

choice /C yn /M "tune"
if !errorlevel! equ 1 (goto tune)

%python_embeded_path%\python.exe benchmark_attn.py>benchmark_attn-result.txt
goto eof

:tune
ren benchmark_attn-tune.txt benchmark_attn-tune.txt.bak
for /F "usebackq delims=" %%C in ("benchmark_attn_config.txt") do (
	set "FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON=%%C"
	echo !FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON!>>benchmark_attn-tune.txt
	%python_embeded_path%\python.exe benchmark_attn_tune.py>>benchmark_attn-tune.txt
)
