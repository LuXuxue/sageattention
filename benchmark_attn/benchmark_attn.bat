@echo off
setlocal enabledelayedexpansion
if exist "%~dp0python_embeded" (set python_embeded_path=%~dp0python_embeded) else (set python_embeded_path=E:\python_embeded)
choice /C yn /M "autotune"
if !errorlevel! equ 1 (goto autotune)
choice /C yn /M "manually"
if !errorlevel! equ 1 (goto manually)

%python_embeded_path%\python.exe benchmark_attn.py>benchmark_attn-result.txt
goto eof

:autotune
set FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=1
%python_embeded_path%\python.exe benchmark_attn.py>benchmark_attn-autotune.txt
goto eof

:manually
ren benchmark_attn-manually.txt benchmark_attn-manually.txt.bak
for /F "usebackq delims=" %%C in ("benchmark_attn_config.txt") do (
	set "FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON=%%C"
	echo !FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON!>>benchmark_attn-manually.txt
	%python_embeded_path%\python.exe benchmark_attn_lite.py>>benchmark_attn-manually.txt
)
