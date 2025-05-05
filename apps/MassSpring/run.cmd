@echo off
set SIZES=11 101 201 301 401 501 601 701 801 901 1001

for %%F in (%SIZES%) do (
    ..\..\build\bin\Release\MassSpring.exe %%F
)
