#pragma once
#include <stdint.h>

#include "rxmesh/lp_pair.cuh"

namespace rxmesh {

/**
 * @brief Store the neighbor patches neighbor to a patch in a fixed size array.
 */
struct PatchStash
{
    uint32_t get_patch(uint8_t id) const
    {
        return m_stash[id];
    }

    uint32_t& get_patch(uint8_t id)
    {
        return m_stash[id];
    }

    uint32_t m_stash[1 << LPPair::PatchStashNumBits];
};
}  // namespace rxmesh