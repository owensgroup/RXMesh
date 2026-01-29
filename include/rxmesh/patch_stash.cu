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

__device__ uint8_t PatchStash::insert_patch(uint32_t patch, ShmemMutex& mutex)
{
    // in case it was there already
    uint8_t ret = find_patch_index(patch);
    if (ret != INVALID8) {
        return ret;
    }

    // otherwise, we will have to lock access to m_stash
    mutex.lock();
    ret = insert_patch(patch);
    mutex.unlock();
    return ret;
}

__host__ __device__ uint8_t PatchStash::insert_patch(uint32_t patch)
{
    assert(patch != INVALID32);

    uint8_t empty_slot = INVALID8;
    for (uint8_t i = 0; i < stash_size; ++i) {
        if (m_stash[i] == patch) {
            // prevent redundancy
            return i;
        }

        // update the empty_slot if there is a an empty slot in the stash
        // and update the empty_slot only once
        if (m_stash[i] == INVALID32 && empty_slot == INVALID8) {
            empty_slot = i;
        }
    }

    // if we have found an empty_slot and also this patch has not be
    // encountered
    if (empty_slot != INVALID8) {
        m_stash[empty_slot] = patch;
    }

    // return the empty_slot even if it is not updated. If it is not
    // updated, then return INVALID8 would indicate that the patch has not
    // been added
    return empty_slot;
}

__host__ __device__ uint8_t PatchStash::find_patch_index(uint32_t patch) const
{
    assert(patch != INVALID32);
    for (uint8_t i = 0; i < stash_size; ++i) {
        if (m_stash[i] == patch) {
            return i;
        }
    }
    return INVALID8;
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
