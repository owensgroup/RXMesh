@echo off
::call compute-sanitizer  --log-file sanitize_memcheck.log --tool memcheck --check-cache-control --leak-check full ..\build\bin\Debug\DelaunayEdgeFlip.exe  -input ..\input\ear2.obj
::call compute-sanitizer  --log-file sanitize_racecheck.log --tool racecheck  --racecheck-report analysis ..\build\bin\Debug\RXMesh_test.exe --gtest_filter=RXMeshDynamic.RandomFlips
::call compute-sanitizer  --log-file sanitize_initcheck.log --tool initcheck --track-unused-memory ..\build\bin\Debug\RXMesh_test.exe --gtest_filter=RXMeshDynamic.RandomFlips


setlocal enabledelayedexpansion

for /l %%i in (1,1,100) do (
    echo *****************************************
    echo ********* Running iteration %%i *********
    echo *****************************************
    call compute-sanitizer --log-file sanitize_memcheck_%%i.log --tool memcheck --check-cache-control ..\build\bin\Debug\Remesh.exe -nx 330 -ny 330
	call compute-sanitizer --log-file sanitize_racecheck_%%i.log --tool racecheck ..\build\bin\Debug\Remesh.exe -nx 330 -ny 330	
)
endlocal