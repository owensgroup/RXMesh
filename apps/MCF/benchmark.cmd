@echo off
setlocal EnableDelayedExpansion

set exe=..\..\build\bin\Release\MCF.exe

if not exist %exe% (
    echo The code has not been compiled. Please compile MCF and retry!
    exit /b 1
)

set device_id=0

rem Define the input directories
set input_dir1=C:\Github\RXMesh\input
set input_dir2=C:\Github\RXMesh\input\Thingi10K
set input_dir3=C:\Github\RXMesh\input\ThreeDScans\clean
set input_dir4=C:\Github\RXMesh\input\SI


rem Flag to indicate whether to start processing files
set start_processing=1

rem Define the file name to start processing from
rem set start_file=C:\Github\RXMesh\input\SI\usnm_1149322-20m.obj

rem Loop over each directory
for %%d in (%input_dir4%) do (
    for %%f in (%%d\*.obj) do (
		echo %%f
        if not "!start_processing!"=="1" (
            rem if "%%f"=="%start_file%" (
            rem     set start_processing=1
			rem 	echo !start_processing!
            rem )
        ) else (
            if exist "%%f" (
                echo %exe% -input "%%f" -device_id %device_id% -perm_method symamd
                %exe% -input "%%f" -device_id %device_id% -perm_method symamd
            )
        )
    )
)
endlocal