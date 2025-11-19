#!/usr/bin/env bash

cd /home/behrooz/Desktop/Last_Project/RXMesh-dev/cmake-build-release/bin

nsys profile \
  --capture-range=cudaProfilerApi \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,openmp \
  --nvtx-domain-include=default \
  -o /home/behrooz/Desktop/Last_Project/RXMesh-dev/output/profile_output/decompose_profile \
  ./RXMesh_benchmark \
  -i /media/behrooz/FarazHard/Last_Project/MIT_meshes/nefertiti.obj \
  -s CHOLMOD \
  -a POC_ND
  -g false