@echo off
setlocal enabledelayedexpansion
if exist "%~dp0python_embeded" (set python_embeded_path=%~dp0python_embeded) else (set python_embeded_path=E:\python_embeded)
choice /C yn /M "autotune"
if !errorlevel! equ 1 (goto autotune)

echo Default Config>benchmark_attn-result.txt
%python_embeded_path%\python.exe benchmark_attn.py>nul
%python_embeded_path%\python.exe benchmark_attn.py>>benchmark_attn-result.txt

REM set FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON={"BLOCK_M":128,"BLOCK_N":16,"waves_per_eu":6,"PRE_LOAD_V":false,"num_warps":8,"num_stages":2}
REM echo !FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON!>>benchmark_attn-result.txt
REM %python_embeded_path%\python.exe benchmark_attn.py>nul
REM %python_embeded_path%\python.exe benchmark_attn.py>>benchmark_attn-result.txt

goto eof

:autotune
set FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=1
%python_embeded_path%\python.exe benchmark_attn.py>benchmark_attn-autotune.txt
