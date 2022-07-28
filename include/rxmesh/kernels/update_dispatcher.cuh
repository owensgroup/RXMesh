#pragma once

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

template <DynOp op, uint32_t blockThreads, typename predicateT>
__device__ __inline__ void update_block_dispatcher(Context&         context,
                                                   const predicateT predicate)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(context.get_dirty()) = 1;
    }

    const uint32_t patch_id = blockIdx.x;

    if (patch_id >= context.get_num_patches()) {
        return;
    }
}

}  // namespace rxmesh