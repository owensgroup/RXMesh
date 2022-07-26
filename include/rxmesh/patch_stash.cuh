#pragma once
#include <stdint.h>

#include "rxmesh/lp_pair.cuh"

namespace rxmesh {

/**
 * @brief Store the neighbor patches neighbor to a patch in a fixed size array.
 */
struct PatchStash
{
    static constexpr uint8_t stash_size = (1 << LPPair::PatchStashNumBits);

    __host__ __device__ PatchStash(bool on_device) : m_is_on_device(on_device)
    {
        if (m_is_on_device) {
            CUDA_ERROR(
                cudaMalloc((void**)&m_stash, stash_size * sizeof(uint32_t)));
            CUDA_ERROR(
                cudaMemset(m_stash, INVALID8, stash_size * sizeof(uint32_t)));
        } else {
            m_stash = (uint32_t*)malloc(stash_size * sizeof(uint32_t));
            for (uint8_t i = 0; i < stash_size; ++i) {
                m_stash[i] = INVALID32;
            }
        }
    }

    PatchStash()                        = default;
    PatchStash(const PatchStash& other) = default;
    PatchStash(PatchStash&&)            = default;
    PatchStash& operator=(const PatchStash&) = default;
    PatchStash& operator=(PatchStash&&) = default;
    ~PatchStash()                       = default;

    __host__ __device__ __inline__ uint32_t get_patch(uint8_t id) const
    {
        return m_stash[id];
    }

    __host__ __device__ __inline__ uint32_t& get_patch(uint8_t id)
    {
        return m_stash[id];
    }

    __host__ __device__ __inline__ uint32_t get_patch(const LPPair p) const
    {
        return get_patch(p.patch_stash_id());
    }

    __host__ __device__ __inline__ uint32_t& get_patch(const LPPair p)
    {
        return get_patch(p.patch_stash_id());
    }

    __host__ __device__ __inline__ void insert_patch(uint32_t patch)
    {
        for (uint8_t i = 0; i < stash_size; ++i) {
            if (m_stash[i] == patch) {
                // prevent redundancy
                return;
            }
            if (m_stash[i] == INVALID32) {
                m_stash[i] = patch;
                return;
            }
        }
        assert(1 != 1);
    }

    __host__ __device__ __inline__ uint8_t find_patch_index(
        uint32_t patch) const
    {
        for (uint8_t i = 0; i < stash_size; ++i) {
            if (m_stash[i] == patch) {
                return i;
            }
        }
        return INVALID8;
    }


    __host__ void free()
    {
        if (m_is_on_device) {
            GPU_FREE(m_stash);
        } else {
            ::free(m_stash);
        }
    }

    uint32_t* m_stash;
    bool      m_is_on_device;
};
}  // namespace rxmesh