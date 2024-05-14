#!/bin/bash
exe="../../build/bin/SurfaceTracking"

if [ ! -f $exe ]; then 
	echo "The code has not been compiled. Please compile Remesh and retry!"
	exit 1
fi

device_id=0

for ((n = 100; n <= 1000; n += 100)); do
		echo $exe -n $n -device_id $device_id
        $exe -n $n -device_id $device_id
done