#pragma once
// HIP-only redirect placeholder: HIP's cooperative groups does not provide the
// memcpy_async / wait async-copy API. RXMesh's loader.cuh guards its async path
// on USE_HIP and falls back to a synchronous cooperative copy, so this shim only
// needs to satisfy the <cooperative_groups/memcpy_async.h> include.
#include <hip/hip_cooperative_groups.h>
