#pragma once
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/patch_info.h"
#include "rxmesh/types.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

template <typename LambdaT>
__device__ __inline__ void for_each_vertex(const PatchInfo patch_info,
                                           LambdaT         apply,
                                           bool allow_not_owned = false)
{

    const uint16_t num_v = patch_info.num_vertices[0];
    for (uint16_t v = threadIdx.x; v < num_v; v += blockDim.x) {
        if (!detail::is_deleted(v, patch_info.active_mask_v)) {
            if (!allow_not_owned &&
                !detail::is_owned(v, patch_info.owned_mask_v)) {
                continue;
            }
            VertexHandle v_handle(patch_info.patch_id, v);
            apply(v_handle);
        }
    }
}
namespace detail {
template <typename LambdaT>
__global__ void for_each_vertex(const uint32_t   num_patches,
                                const PatchInfo* patch_info,
                                LambdaT          apply)
{
    const uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        for_each_vertex(patch_info[p_id], apply);
    }
}
}  // namespace detail


template <typename LambdaT>
__device__ __inline__ void for_each_edge(const PatchInfo patch_info,
                                         LambdaT         apply,
                                         bool allow_not_owned = false)
{

    const uint16_t num_e = patch_info.num_edges[0];
    for (uint16_t e = threadIdx.x; e < num_e; e += blockDim.x) {
        if (!detail::is_deleted(e, patch_info.active_mask_e)) {
            if (!allow_not_owned &&
                !detail::is_owned(e, patch_info.owned_mask_e)) {
                continue;
            }
            EdgeHandle e_handle(patch_info.patch_id, e);
            apply(e_handle);
        }
    }
}
namespace detail {
template <typename LambdaT>
__global__ void for_each_edge(const uint32_t   num_patches,
                              const PatchInfo* patch_info,
                              LambdaT          apply)
{
    const uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        for_each_edge(patch_info[p_id], apply);
    }
}
}  // namespace detail


template <typename LambdaT>
__device__ __inline__ void for_each_face(const PatchInfo patch_info,
                                         LambdaT         apply,
                                         bool allow_not_owned = false)
{
    const uint16_t num_f = patch_info.num_faces[0];
    for (uint16_t f = threadIdx.x; f < num_f; f += blockDim.x) {
        if (!detail::is_deleted(f, patch_info.active_mask_f)) {
            if (!allow_not_owned &&
                !detail::is_owned(f, patch_info.owned_mask_f)) {
                continue;
            }
            FaceHandle f_handle(patch_info.patch_id, f);
            apply(f_handle);
        }
    }
}

namespace detail {
template <typename LambdaT>
__global__ void for_each_face(const uint32_t   num_patches,
                              const PatchInfo* patch_info,
                              LambdaT          apply)
{
    const uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        for_each_face(patch_info[p_id], apply);
    }
}
}  // namespace detail


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
__device__ __inline__ void for_each(const Context& context, computeT compute_op)
{
    static_assert(op == Op::V || op == Op::E || op == Op::F,
                  "for_each() only accepts unary operator for its template "
                  "parameter i.e., Op::V, Op::E, or Op::F");

    using ComputeTraits  = detail::FunctionTraits<computeT>;
    using ComputeHandleT = typename ComputeTraits::template arg<0>::type;

    const uint32_t p_id = blockIdx.x;
    if (p_id < context.m_num_patches[0]) {
        if (context.m_patches_info[p_id].patch_id == INVALID32) {
            return;
        }
        if constexpr (op == Op::V) {
            static_assert(
                std::is_same_v<ComputeHandleT, VertexHandle>,
                "for_each() since input template parameter operation is Op::V, "
                "the "
                "lambda function should take VertexHandle as an input");
            for_each_vertex(context.m_patches_info[p_id], compute_op);
        }
        if constexpr (op == Op::E) {
            static_assert(std::is_same_v<ComputeHandleT, EdgeHandle>,
                          "for_each() since input template parameter operation "
                          "is Op::E, the "
                          "lambda function should take EdgeHandle as an input");
            for_each_edge(context.m_patches_info[p_id], compute_op);
        }
        if constexpr (op == Op::F) {
            static_assert(
                std::is_same_v<ComputeHandleT, FaceHandle>,
                "for_each() since input template parameter operation is "
                "Op::F, the lambda function should take FaceHandle as an "
                "input");
            for_each_face(context.m_patches_info[p_id], compute_op);
        }
    }
    __syncthreads();
}


}  // namespace rxmesh