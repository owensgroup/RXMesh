@echo off
REM --generate-coredump yes --kernel-regex kns=delaunay --launch-skip 6 
call compute-sanitizer  --log-file sanitize_memcheck.log --tool memcheck  --leak-check full ..\build\bin\Debug\RXMesh_test.exe --gtest_filter=RXMeshDynamic.Cavity
call compute-sanitizer  --log-file sanitize_racecheck.log --tool racecheck  --racecheck-report analysis ..\build\bin\Debug\RXMesh_test.exe --gtest_filter=RXMeshDynamic.Cavity
REM call compute-sanitizer  --log-file sanitize_initcheck.log --tool initcheck  ..\build\bin\Debug\DelaunayEdgeFlip.exe
REM call compute-sanitizer  --log-file sanitize_synccheck.log --tool synccheck   ..\build\bin\Debug\DelaunayEdgeFlip.exe