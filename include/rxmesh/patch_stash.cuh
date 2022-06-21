#pragma once
#include <stdint.h>

#include "rxmesh/lp_pair.cuh"

namespace rxmesh {

/**
 * @brief Store the neighbor patches neighbor to a patch in a fixed size array.
 */
struct PatchStash
{
    PatchStash()                        = default;
    PatchStash(const PatchStash& other) = default;
    PatchStash(PatchStash&&)            = default;
    PatchStash& operator=(const PatchStash&) = default;
    PatchStash& operator=(PatchStash&&) = default;
    virtual ~PatchStash()               = default;

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

    uint32_t m_stash[1 << LPPair::PatchStashNumBits];
};
}  // namespace rxmesh