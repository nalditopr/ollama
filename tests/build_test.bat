call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\ReyColónValero\claude\ollama-turboquant"

echo === Building test_turboquant ===
cl.exe /O2 /W3 /Fe:tests\test_turboquant.exe tests\test_turboquant.c
if %ERRORLEVEL% neq 0 (
    echo COMPILATION FAILED: test_turboquant
    goto :eof
)
tests\test_turboquant.exe

echo.
echo === Building bench_turboquant ===
cl.exe /O2 /W3 /Fe:tests\bench_turboquant.exe tests\bench_turboquant.c
if %ERRORLEVEL% neq 0 (
    echo COMPILATION FAILED: bench_turboquant
    goto :eof
)
tests\bench_turboquant.exe
