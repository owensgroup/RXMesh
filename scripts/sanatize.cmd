@echo off
call compute-sanitizer  --log-file sanitize_memcheck.log --tool memcheck --check-cache-control yes --leak-check full ..\build\bin\Debug\DelaunayEdgeFlip.exe  -input ..\input\ear2.obj
::call compute-sanitizer  --log-file sanitize_racecheck.log --tool racecheck  --racecheck-report analysis ..\build\bin\Debug\RXMesh_test.exe --gtest_filter=RXMeshDynamic.RandomFlips
::call compute-sanitizer  --log-file sanitize_initcheck.log --tool initcheck --track-unused-memory yes ..\build\bin\Debug\RXMesh_test.exe --gtest_filter=RXMeshDynamic.RandomFlips