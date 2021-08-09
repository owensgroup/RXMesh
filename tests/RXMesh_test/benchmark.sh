#!/bin/bash
echo "This script re-generates RXMesh data in Figure 6 in the paper."
echo "Please make sure to first compile the source code and then enter the input OBJ files directory."
read -p "OBJ files directory (no trailing slash): " input_dir

echo "Input directory= $input_dir"
exe="../../build/bin/RXMesh_test"

if [ ! -f $exe ]; then 
	echo "The code has not been compiled. Please compile RXMesh_test and retry!"
	exit 1
fi

num_run=10
device_id=0

for file in $input_dir/*.obj; do 	 
    if [ -f "$file" ]; then
		echo $exe --gtest_filter=RXMesh.Queries -input "$file" -num_run $num_run -device_id $device_id
             $exe --gtest_filter=RXMesh.Queries -input "$file" -num_run $num_run -device_id $device_id
		
		echo $exe -s --gtest_filter=RXMesh.Queries -input "$file" -num_run $num_run -device_id $device_id
             $exe -s --gtest_filter=RXMesh.Queries -input "$file" -num_run $num_run -device_id $device_id

		echo $exe -p --gtest_filter=RXMesh.Queries -input "$file" -num_run $num_run -device_id $device_id
             $exe -p --gtest_filter=RXMesh.Queries -input "$file" -num_run $num_run -device_id $device_id
    fi 
done