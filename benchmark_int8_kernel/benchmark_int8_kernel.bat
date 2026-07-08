@echo off
setlocal enabledelayedexpansion
if exist "%~dp0python_embeded" (set python_embeded_path=%~dp0python_embeded) else (set python_embeded_path=E:\python_embeded)
choice /C yn /M "tune"
if !errorlevel! equ 1 (goto tune)

%python_embeded_path%\python.exe benchmark_int8_kernel.py>benchmark_int8_kernel-result.txt
goto eof

:tune
ren benchmark_int8_kernel-tune.txt benchmark_int8_kernel-tune.txt.bak
for /F "usebackq delims=" %%C in ("benchmark_int8_kernel_config.txt") do (
	set "COMFY_KITCHEN_CONFIG_JSON=%%C"
	echo !COMFY_KITCHEN_CONFIG_JSON!>>benchmark_int8_kernel-tune.txt
	%python_embeded_path%\python.exe benchmark_int8_kernel.py>>benchmark_int8_kernel-tune.txt
)
