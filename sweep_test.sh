#!/bin/bash

# Start value
start=0.001
# End value
end=0.9
# Step value
step=0.031

# Command path
command="./build/bin/SECPriority"
# Input file
input_file="./input/rocker-arm.obj"

# Loop through the range
for target in $(seq $start $step $end)
do
    echo "Running with target = $target"
    $command -input $input_file -target $target
done

