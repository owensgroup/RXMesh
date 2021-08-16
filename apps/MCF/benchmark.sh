#!/bin/bash
echo "This script re-generates RXMesh data in Figure 8(a) in the paper."
echo "Please make sure to first compile the source code and then enter the input OBJ files directory."
read -p "OBJ files directory (no trailing slash): " input_dir

echo "Input directory= $input_dir"
exe="../../build/bin/MCF"

if [ ! -f $exe ]; then 
	echo "The code has not been compiled. Please compile MCF and retry!"
	exit 1
fi

num_run=10
device_id=0

for file in $input_dir/*.obj; do 	 
    if [ -f "$file" ]; then
		echo $exe -p -input "$file" -device_id $device_id
             $exe -p -input "$file" -device_id $device_id
    fi 
done