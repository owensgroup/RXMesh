#pragma once
// HIP-only redirect: RXMesh sources include <cooperative_groups.h> (CUDA path);
// on HIP this shim (on the include path only when USE_HIP) forwards to the HIP
// cooperative-groups header so the CUDA source spelling stays untouched.
#include <hip/hip_cooperative_groups.h>
