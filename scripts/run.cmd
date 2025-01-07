@echo off
setlocal enabledelayedexpansion

for /l %%i in (1,1,50) do (
    echo *****************************************
    echo ********* Running iteration %%i *********
    echo *****************************************
    call ..\build\bin\Debug\Remesh.exe -nx 330 -ny 330
)
endlocal