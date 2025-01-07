@echo off
set REPORT_NAME=report
::set EXE=..\build\bin\Release\Remesh.exe -nx 330 -ny 330
::set EXE=..\build\bin\Release\SECPriority.exe -input ..\input\rocker-arm.obj
set EXE=..\build\bin\Release\RXMesh_test.exe  --gtest_filter=RXMeshStatic.Queries -input ..\input\Nefertiti.obj -num_run 10

nsys profile -t cuda,nvtx --force-overwrite true -o %REPORT_NAME% %EXE% 

nsys stats --report cuda_gpu_kern_sum %REPORT_NAME%.nsys-rep --timeunit ms --force-export=true