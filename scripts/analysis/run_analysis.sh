#!/bin/bash

# Simple script to run separator analysis
cd /home/behrooz/Desktop/Last_Project/RXMesh-dev

# Run the analysis with a sample mesh
./cmake-build-release/bin/RXMesh_sepComp_runtime -i input/bunnyhead.obj -o output/

echo "Analysis completed!"
