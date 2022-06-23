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

    PatchStash()
    {
        for (uint8_t i = 0; i < stash_size; ++i) {
            m_stash[i] = INVALID32;
        }
    }

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

    void insert_patch(uint32_t patch)
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

    uint8_t find_patch_index(uint32_t patch) const
    {
        for (uint8_t i = 0; i < stash_size; ++i) {
            if (m_stash[i] == patch) {
                return i;
            }
        }
        return INVALID8;
    }

    uint32_t m_stash[stash_size];
};
}  // namespace rxmesh