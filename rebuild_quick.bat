@echo off

set "MSVC_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
set "WINSDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "GCC_PATH=C:\Users\ReyColónValero\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"

set "PATH=C:\Program Files\Go\bin;C:\Program Files\CMake\bin;%CUDA_PATH%\bin;%MSVC_PATH%;%WINSDK_BIN%;%GCC_PATH%;%PATH%"
set "CGO_ENABLED=1"
set "CC=gcc"

cd /d "C:\Users\ReyColónValero\claude\ollama-turboquant"

echo === Rebuilding CUDA ===
cmake --build build --config Release --target ggml-cuda -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% neq 0 (
    echo CUDA BUILD FAILED
    exit /b 1
)

echo === Rebuilding Go ===
go build -o ollama.exe .
if %ERRORLEVEL% neq 0 (
    echo GO BUILD FAILED
    exit /b 1
)

echo === BUILD COMPLETE ===
