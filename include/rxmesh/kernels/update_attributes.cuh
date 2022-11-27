#pragma once
#include "rxmesh/bitmask.cuh"

namespace rxmesh {
template <typename AttributeT>
__device__ __inline__ void update_attributes(const uint32_t patch_id,
                                             AttributeT     attribute,
                                             const Bitmask& ownership_change)
{
    using HandleT = AttributeT::HandleT;
    using Type    = AttributeT::T;
}

}  // namespace rxmesh