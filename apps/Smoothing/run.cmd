@echo off
set FILES=../../input/ear2.obj ../../input/edgar-allan-poe-1.obj ../../input/Nefertiti.obj ../../input/Lucy_3M_O10.obj ../../input/20131208_VenusDeMilo_Full_Scale_captured_by_Cosmo_Wenman.obj ../../input/wingedvictory.obj ../../input/blue-crab-master-geometry.obj ../../input/SI/f1930_54-part_02-smartscan-fixed-textured.obj ../../input/SI/bell_x1-complete_with_vane-smooth.obj ../../input/SI/npg_70_4_bust-hires_unwrapped.obj ../../input/SI/f1930_54-part_01-smartscan-fixed-textured.obj ../../input/SI/mammoth-master_model.obj ../../input/SI/f1961_33-part_02-x_pol-ort_texture.obj ../../input/SI/cosmic_buddha-full_resolution-no_texture.obj

set exe="../../build/bin/Release/Smoothing.exe"

for %%F in (%FILES%) do (
    REM echo Processing %%F
    %exe% -input %%F
)