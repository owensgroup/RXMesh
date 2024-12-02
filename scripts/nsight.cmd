@echo off
set REPORT_NAME=report
set EXE=..\build\bin\Release\Remesh.exe -nx 800 -ny 800

nsys profile -t cuda,nvtx --force-overwrite true -o %REPORT_NAME% %EXE% 

nsys stats --report cuda_gpu_kern_sum %REPORT_NAME%.nsys-rep --timeunit ms --force-export=true