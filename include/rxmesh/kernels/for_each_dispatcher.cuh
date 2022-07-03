#pragma once

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/for_each.cuh"
#include "rxmesh/types.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

/**
 * @brief Apply a lambda function on all mesh elements. The type of the mesh
 * element is inferred from the op which could be
 * 1) Op::V for all vertices,
 * 2) Op::E for all edge, or
 * 3) Op::F for all faces,
 *
 * @tparam Op the type of query operation
 * @tparam blockThreads the number of CUDA threads in the block
 * @tparam computeT the type of compute lambda function (inferred)
 * @param context which store various parameters internal to RXMesh. The context
 * can be obtained from RXMeshStatic
 * @param compute_op the computation lambda function that will be executed on
 * each mesh element. This lambda function takes only one input parameters i.e.,
 *  Handle to the mesh element assigned to the thread. The handle type matches
 * the source of the query (e.g., VertexHandle for Op::V)
 */
template <Op op, uint32_t blockThreads, typename computeT>
__device__ __inline__ void for_each_dispatcher(const Context& context,
                                               computeT       compute_op)
{
    static_assert(op == Op::V || op == Op::E || op == Op::F,
                  "for_each_dispatcher() only accepts unary operator for its "
                  "template parameter i.e., Op::V, Op::E, or Op::F");

    using ComputeTraits  = detail::FunctionTraits<computeT>;
    using ComputeHandleT = typename ComputeTraits::template arg<0>::type;

    if constexpr (op == Op::V) {
        static_assert(
            std::is_same_v<ComputeHandleT, VertexHandle>,
            "for_each_dispatcher() since input template parameter operation is "
            "Op::V, the lambda function should take VertexHandle as an input");
        detail::for_each_vertex_kernel(
            context.get_num_patches(), context.get_patches_info(), compute_op);
    }
    if constexpr (op == Op::E) {
        static_assert(
            std::is_same_v<ComputeHandleT, EdgeHandle>,
            "for_each_dispatcher() since input template parameter operation is "
            "Op::E, the lambda function should take EdgeHandle as an input");
        detail::for_each_edge_kernel(
            context.get_num_patches(), context.get_patches_info(), compute_op);
    }
    if constexpr (op == Op::F) {
        static_assert(
            std::is_same_v<ComputeHandleT, FaceHandle>,
            "for_each_dispatcher() since input template parameter operation is "
            "Op::F, the lambda function should take FaceHandle as an input");
        detail::for_each_face_kernel(
            context.get_num_patches(), context.get_patches_info(), compute_op);
    }

    __syncthreads();
}

}  // namespace rxmesh