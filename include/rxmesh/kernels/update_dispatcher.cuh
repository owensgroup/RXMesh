#pragma once

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/delete_edge.cuh"
#include "rxmesh/kernels/delete_face.cuh"
#include "rxmesh/kernels/delete_vertex.cuh"
#include "rxmesh/kernels/edge_flip.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

/**
 * @brief The main entry for update operations. This function should be called
 * by the whole block. In this function, threads will be assigned to mesh
 * elements depending on the update operations.The user should supply a
 * predicate lambda function that check if the update operation should be done
 * on the input mesh element.
 * @tparam DynOp the type of update operation
 * @tparam blockThreads the number of CUDA threads in the block
 * @tparam predicateT the type of predicate lambda function (inferred)
 * @param context which store various parameters needed for the update
 * operation. The context can be obtained from RXMeshDynamic
 * @param predicate the predicate lambda function that will be executed by
 * each thread in the block. This lambda function takes one input parameters
 * which is a handle depending on the update operations e.g., EdgeHandle for
 * edge flip. This lambda function should return a boolean that is true only if
 * the update operation should be done on the give mesh element
 */
template <DynOp op, uint32_t blockThreads, typename predicateT>
__device__ __inline__ void update_block_dispatcher(const Context&   context,
                                                   const predicateT predicate)
{

    const uint32_t patch_id = blockIdx.x;

    if (patch_id >= context.get_num_patches()) {
        return;
    }

    if constexpr (op == DynOp::EdgeFlip) {
        detail::edge_flip<blockThreads>(context.get_patches_info()[patch_id],
                                        predicate);
    }

    if constexpr (op == DynOp::DeleteFace) {
        detail::delete_face<blockThreads>(context.get_patches_info()[patch_id],
                                          predicate);
    }

    if constexpr (op == DynOp::DeleteEdge) {
        detail::delete_edge<blockThreads>(context.get_patches_info()[patch_id],
                                          predicate);
    }

    if constexpr (op == DynOp::DeleteVertex) {
        detail::delete_vertex<blockThreads>(
            context.get_patches_info()[patch_id], predicate);
    }
}

}  // namespace rxmesh