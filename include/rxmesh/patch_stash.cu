#include "rxmesh/patch_stash.h"

#include <assert.h>
#include <cstdlib>

#include "rxmesh/util/macros.h"

namespace rxmesh {

__host__ PatchStash::PatchStash(bool on_device) : m_is_on_device(on_device)
{
    if (m_is_on_device) {
        CUDA_ERROR(cudaMalloc((void**)&m_stash, stash_size * sizeof(uint32_t)));
        CUDA_ERROR(
            cudaMemset(m_stash, INVALID8, stash_size * sizeof(uint32_t)));
    } else {
        m_stash = (uint32_t*)malloc(stash_size * sizeof(uint32_t));
        for (uint8_t i = 0; i < stash_size; ++i) {
            m_stash[i] = INVALID32;
        }
    }
}

__host__ void PatchStash::free()
{
    if (m_is_on_device) {
        GPU_FREE(m_stash);

    } else {
        ::free(m_stash);
    }
}

}  // namespace rxmesh
