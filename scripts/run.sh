
#!/usr/bin/env bash

cd /home/behrooz/Desktop/Last_Project/RXMesh-dev/cmake-build-release/bin

./RXMesh_benchmark \
  -i /media/behrooz/FarazHard/Last_Project/MIT_meshes/nefertiti.obj \
  -s CHOLMOD \
  -a POC_ND
  -g false