@echo off
:: Set default values
set "default_input_dir=../../../input"
set "default_output_dir=output_folder"
set "default_dt=5"
set "default_max_iter=10000"
set "default_solver=cg"
set "default_eps=1e-5"
set "default_device_id=0"
set "default_perm=nstdis"
set "default_levels=2"

:: Get user input or fallback to defaults
set /p input_dir=Enter input directory (default: %default_input_dir%): 
if "%input_dir%"=="" set "input_dir=%default_input_dir%"

set /p output_dir=Enter output directory (default: %default_output_dir%): 
if "%output_dir%"=="" set "output_dir=%default_output_dir%"

set /p dt=Enter time step dt (default: %default_dt%): 
if "%dt%"=="" set "dt=%default_dt%"

set /p max_iter=Enter maximum number of iterations (default: %default_max_iter%): 
if "%max_iter%"=="" set "max_iter=%default_max_iter%"

set /p solver=Enter solver (default: %default_solver%): 
if "%solver%"=="" set "solver=%default_solver%"

if "%solver%"=="gmg" set /p levels=Enter number of levels (default: %default_levels%): 
if "%levels%"=="" set "levels=%default_perm%"


set /p eps=Enter solver tolerance eps (default: %default_eps%): 
if "%eps%"=="" set "eps=%default_eps%"

set /p device_id=Enter device ID (default: %default_device_id%): 
if "%device_id%"=="" set "device_id=%default_device_id%"

set /p perm=Enter permutation method (default: %default_perm%): 
if "%perm%"=="" set "perm=%default_perm%"



:: Loop over all .obj files in input_dir
for %%f in ("%input_dir%\*.obj") do (
    echo Running MCF.exe on %%f
MCF.exe -input "%%f" -o "%output_dir%" -dt "%dt%" -max_iter "%max_iter%" -solver "%solver%" -eps "%eps%" -device_id "%device_id%" -perm "%perm%" -levels "%levels%"
)

pause

