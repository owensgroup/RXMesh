@echo off
::call compute-sanitizer  --log-file sanitize_memcheck.log --tool memcheck --check-cache-control --leak-check full ..\build\bin\Debug\DelaunayEdgeFlip.exe  -input ..\input\ear2.obj
::call compute-sanitizer  --log-file sanitize_racecheck.log --tool racecheck  --racecheck-report analysis ..\build\bin\Debug\RXMesh_test.exe --gtest_filter=RXMeshDynamic.RandomFlips
::call compute-sanitizer  --log-file sanitize_initcheck.log --tool initcheck --track-unused-memory ..\build\bin\Debug\RXMesh_test.exe --gtest_filter=RXMeshDynamic.RandomFlips


setlocal enabledelayedexpansion

for /l %%i in (1,1,20) do (
    echo *****************************************
    echo ********* Running iteration %%i *********
    echo *****************************************
    call compute-sanitizer --log-file sanitize_memcheck_%%i.log --tool memcheck --check-cache-control ..\build\bin\Release\Remesh.exe -nx 30 -ny 50000
)
endlocal