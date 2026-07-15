#pragma once
// HIP-only redirect: <cuda_runtime.h> -> the RXMesh CUDA->HIP compat header,
// which pulls in the HIP runtime and aliases the cuda* spellings RXMesh uses.
#include "rxmesh/util/cuda_to_hip.h"
