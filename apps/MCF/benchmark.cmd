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
set start_processing=0

rem Define the file name to start processing from
set start_file=C:\Github\RXMesh\input\SI\bell_x1-complete_with_vane-smooth.obj

rem Loop over each directory
for %%d in (%input_dir4% %input_dir1% %input_dir2% %input_dir3%) do (
    for %%f in (%%d\*.obj) do (
		echo %%f
        if not "!start_processing!"=="1" (
            if "%%f"=="%start_file%" (
                set start_processing=1
				echo !start_processing!
            )
        ) else (
            if exist "%%f" (
                echo %exe% -input "%%f" -device_id %device_id% -perm_method symamd
                %exe% -input "%%f" -device_id %device_id% -perm_method symamd
            )
        )
    )
)
endlocal