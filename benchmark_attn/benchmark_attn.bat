@echo off
setlocal enabledelayedexpansion
if exist "%~dp0python_embeded" (set python_embeded_path=%~dp0python_embeded) else (set python_embeded_path=E:\python_embeded)
REM set SAGEATTN_DEFAULT_PV_ACCUM_DTYPE=fp16

choice /C yn /M "tune"
if !errorlevel! equ 1 (goto tune)

%python_embeded_path%\python.exe benchmark_attn.py>benchmark_attn-result.txt 2>&1
goto eof

:tune
echo config,Case,Time,TFLOPS,Backend>benchmark_attn-tune.csv
for /F "usebackq delims=" %%C in ("benchmark_attn_config.txt") do (
	set "FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON=%%C"
	%python_embeded_path%\python.exe benchmark_attn_tune.py>>benchmark_attn-tune.csv
)
pause
