@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

set "PATH=C:\Program Files\Go\bin;C:\Program Files\CMake\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set CGO_ENABLED=1

cd /d "C:\Users\ReyColónValero\claude\ollama-turboquant"

echo === Step 1: CMake Configure ===
if exist build (
    rmdir /s /q build
    timeout /t 2 /nobreak >nul
)
cmake -B build -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe"
if %ERRORLEVEL% neq 0 (
    echo CMAKE CONFIGURE FAILED
    exit /b 1
)

echo === Step 2: CMake Build ===
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% neq 0 (
    echo CMAKE BUILD FAILED
    exit /b 1
)

echo === Step 3: Go Build ===
set "PATH=C:\Users\ReyColónValero\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;%PATH%"
set CC=gcc
go build -o ollama.exe .
if %ERRORLEVEL% neq 0 (
    echo GO BUILD FAILED
    exit /b 1
)

echo === BUILD COMPLETE ===
ollama.exe --version
