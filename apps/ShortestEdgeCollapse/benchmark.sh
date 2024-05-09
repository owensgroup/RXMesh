#!/bin/bash
#echo "Please make sure to first compile the source code and then enter the input OBJ files directory."
#read -p "OBJ files directory (no trailing slash): " input_dir

input_dir="/home/ahmed/RXMesh/input"
echo "Input directory= $input_dir"
exe="../../build/bin/ShortestEdgeCollapse"

if [ ! -f $exe ]; then 
	echo "The code has not been compiled. Please compile ShortestEdgeCollapse and retry!"
	exit 1
fi

num_run=1
device_id=1

for file in $input_dir/*.obj; do 	 
    if [ -f "$file" ]&& [ "$(basename "$file")" != "Engine_Assy.obj" ] && [ "$(basename "$file")" != "Chain_Hoist.obj" ] && [ "$(basename "$file")" != "Beethoven.obj" ] && [ "$(basename "$file")" != "Upper_Capsule.obj" ] && [ "$(basename "$file")" != "Jellyfish_lamp_part_A__B.obj" ]; then
		echo $exe -input "$file" -num_iter 3 -device_id $device_id
        $exe -input "$file" -num_iter 3 -device_id $device_id
    fi 
done