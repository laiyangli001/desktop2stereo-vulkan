@echo off
setlocal
pushd "%~dp0"
set ROOT=%~dp0..\..\..
set SDK=%ROOT%\native\filament\sdk\windows\v1.76.0
set OUT=%ROOT%\src\desktop2stereo\xr_viewer\native
if not exist "%OUT%" mkdir "%OUT%"
set LIB=%SDK%\lib\x86_64\md
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
setlocal EnableDelayedExpansion
cmake -S . -B build\local -G "Visual Studio 17 2022" -A x64 -DFILAMENT_SDK_ROOT="%SDK%"
cmake --build build\local --config Release --parallel 2
endlocal
popd
