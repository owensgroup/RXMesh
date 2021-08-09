#pragma once
#include <assert.h>
#include <stdint.h>

#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/rxmesh_iterator.cuh"
#include "rxmesh/kernels/rxmesh_loader.cuh"
#include "rxmesh/kernels/rxmesh_queries.cuh"
#include "rxmesh/rxmesh.h"
#include "rxmesh/rxmesh_context.h"
#include "rxmesh/rxmesh_util.h"


namespace RXMESH {

namespace detail {

/**
 * query_block_dispatcher()
 */
template <Op op, uint32_t blockThreads, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(
    const RXMeshContext& context,
    const uint32_t       current_patch_id,
    activeSetT           compute_active_set,
    const bool           oriented,
    const bool           output_needs_mapping,
    uint32_t&            num_src_in_patch,
    uint32_t*&           input_mapping,
    uint32_t*&           s_output_mapping,
    uint16_t*&           s_offset_all_patches,
    uint16_t*&           s_output_all_patches)
{
    static_assert(op != Op::EE, "Op::EE is not supported!");
    assert(current_patch_id < context.get_num_patches());


    ELEMENT src_element, output_element;
    io_elements(op, src_element, output_element);

    extern __shared__ uint16_t shrd_mem[];


    s_offset_all_patches = shrd_mem;
    s_output_all_patches = shrd_mem;
    uint16_t *s_patch_edges(shrd_mem), *s_patch_faces(shrd_mem);

    constexpr bool load_faces = (op == Op::VF || op == Op::EE || op == Op::EF ||
                                 op == Op::FV || op == Op::FE || op == Op::FF);
    constexpr bool load_edges = (op == Op::VV || op == Op::VE || op == Op::VF ||
                                 op == Op::EV || op == Op::FV);
    static_assert(load_edges || load_faces,
                  "At least faces or edges needs to be loaded");

    constexpr bool is_fixed_offset =
        (op == Op::EV || op == Op::FV || op == Op::FE);

    __syncthreads();

    // 1) load the patch addressed and size
    uint4        ad_size;
    uint2        ad_size_ltog_v, ad_size_ltog_e, ad_size_ltog_f;
    const uint2& output_ele_ad_size =
        ((output_element == ELEMENT::EDGE) ?
             ad_size_ltog_e :
             ((output_element == ELEMENT::FACE) ? ad_size_ltog_f :
                                                  ad_size_ltog_v));
    const uint2& src_element_ad_size =
        ((src_element == ELEMENT::EDGE) ?
             ad_size_ltog_e :
             ((src_element == ELEMENT::FACE) ? ad_size_ltog_f :
                                               ad_size_ltog_v));
    load_patch_ad_size(context, current_patch_id, ad_size, ad_size_ltog_v,
                       ad_size_ltog_e, ad_size_ltog_f);

    // Check if any of the vertices are in the active set
    // input mapping does not need to be stored in shared memory since it will
    // be read coalesced, we can rely on L1 cache here
    input_mapping = nullptr;
    num_src_in_patch = 0;
    switch (src_element) {
        case RXMESH::ELEMENT::VERTEX: {
            input_mapping =
                context.get_patches_ltog_v() + src_element_ad_size.x;
            num_src_in_patch = context.get_size_owned()[current_patch_id].z;
            break;
        }
        case RXMESH::ELEMENT::EDGE: {
            input_mapping =
                context.get_patches_ltog_e() + src_element_ad_size.x;
            num_src_in_patch = context.get_size_owned()[current_patch_id].y;
            break;
        }
        case RXMESH::ELEMENT::FACE: {
            input_mapping =
                context.get_patches_ltog_f() + src_element_ad_size.x;
            num_src_in_patch = context.get_size_owned()[current_patch_id].x;
            break;
        }
    }


    bool     is_active = false;
    uint16_t local_id = threadIdx.x;
    while (local_id < num_src_in_patch) {
        is_active =
            local_id || compute_active_set(input_mapping[local_id] >> 1);
        local_id += blockThreads;
    }


    if (__syncthreads_or(is_active) == 0) {
        return;
    }

    assert(ad_size.y == ad_size_ltog_e.y * 2);
    assert(ad_size.w == ad_size_ltog_f.y * 3);


    // 2) Load the patch info
    load_mesh(context, load_edges, load_faces, s_patch_edges, s_patch_faces,
              ad_size);
    __syncthreads();

    // 3)Perform the query operation
    if (oriented) {
        assert(op == Op::VV);
        if constexpr (op == Op::VV) {
            v_v_oreinted<blockThreads>(
                s_offset_all_patches, s_output_all_patches, s_patch_edges,
                context, ad_size, ad_size_ltog_v.y, num_src_in_patch);
        }
    } else {
        query<blockThreads, op>(s_offset_all_patches, s_output_all_patches,
                                s_patch_edges, s_patch_faces, ad_size_ltog_v.y,
                                ad_size_ltog_e.y, ad_size_ltog_f.y);
    }


    // 4) load output mapping
    s_output_mapping = nullptr;
    if (output_needs_mapping) {
        // Read comments in calc_shared_memory() to understand how we calculate
        // s_output_mapping pointer location in shared memory such that it does
        // not overwrite the results

        // We add ad_size.w % 2 for padding in case ad_size.w  is not
        // dividable by 2 in which case memory misalignment happens
        if constexpr (op == Op::FE) {
            s_output_mapping =
                (uint32_t*)&shrd_mem[ad_size.w + (ad_size.w % 2)];
        }
        if constexpr (op == Op::EV) {
            s_output_mapping = (uint32_t*)&shrd_mem[ad_size.y];
        }
        if constexpr (op == Op::FV) {
            s_output_mapping =
                (uint32_t*)&shrd_mem[ad_size.w + (ad_size.w % 2) + ad_size.y];
        }
        if constexpr (op == Op::VE) {
            s_output_mapping = (uint32_t*)&shrd_mem[2 * ad_size.y];
        }
        if constexpr (op == Op::EF || op == Op::VF) {
            s_output_mapping = (uint32_t*)&shrd_mem[2 * ad_size.w];
        }
        if constexpr (op == Op::FF) {
            // FF uses a lot of shared memory and some of it can be overridden
            // but we need to wait for the query to be done.
            __syncthreads();
            s_output_mapping = (uint32_t*)&shrd_mem[0];
        }

        if constexpr (op == Op::VV) {
            // We use extra shared memory that is read only for VV which we can
            // just use for loading ltog. The drawback is that we need to wait
            // for the query to finish first before overwriting it with ltog
            __syncthreads();
            uint16_t last_vv = ad_size_ltog_v.y + 1 + 2 * ad_size_ltog_e.y;
            s_output_mapping = (uint32_t*)&shrd_mem[last_vv + last_vv % 2];
        }

        load_mapping(context, output_element, output_ele_ad_size,
                     s_output_mapping, false);
    }
    __syncthreads();
}
}  // namespace detail
/**
 * query_block_dispatcher()
 */
template <Op op, uint32_t blockThreads, typename computeT, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(
    const RXMeshContext& context,
    const uint32_t       current_patch_id,
    computeT             compute_op,
    activeSetT           compute_active_set,
    const bool           oriented = false,
    const bool           output_needs_mapping = true)
{
    static_assert(op != Op::EE, "Op::EE is not supported!");
    assert(current_patch_id < context.get_num_patches());

    uint32_t  num_src_in_patch = 0;
    uint32_t *input_mapping(nullptr), *s_output_mapping(nullptr);
    uint16_t *s_offset_all_patches(nullptr), *s_output_all_patches(nullptr);

    detail::template query_block_dispatcher<op, blockThreads>(
        context, current_patch_id, compute_active_set, oriented,
        output_needs_mapping, num_src_in_patch, input_mapping, s_output_mapping,
        s_offset_all_patches, s_output_all_patches);

    assert(input_mapping);
    assert(s_output_all_patches);

    // 5) Call compute on the output in shared memory by looping over all
    // source elements in this patch.

    uint16_t local_id = threadIdx.x;
    while (local_id < num_src_in_patch) {

        uint32_t global_id = input_mapping[local_id] >> 1;

        if (compute_active_set(global_id)) {
            constexpr uint32_t fixed_offset =
                ((op == Op::EV)                 ? 2 :
                 (op == Op::FV || op == Op::FE) ? 3 :
                                                  0);
            RXMeshIterator iter(local_id, s_output_all_patches,
                                s_offset_all_patches, s_output_mapping,
                                fixed_offset, int(op == Op::FE));

            compute_op(global_id, iter);
        }

        local_id += blockThreads;
    }
}

/**
 * query_block_dispatcher()
 */
template <Op op, uint32_t blockThreads, typename computeT, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(
    const RXMeshContext& context,
    computeT             compute_op,
    activeSetT           compute_active_set,
    const bool           oriented = false,
    const bool           output_needs_mapping = true)
{
    if (blockIdx.x >= context.get_num_patches()) {
        return;
    }
    query_block_dispatcher<op, blockThreads>(context, blockIdx.x, compute_op,
                                             compute_active_set, oriented,
                                             output_needs_mapping);
}

/**
 * query_block_dispatcher()
 */
template <Op op, uint32_t blockThreads, typename computeT>
__device__ __inline__ void query_block_dispatcher(
    const RXMeshContext& context,
    computeT             compute_op,
    const bool           oriented = false,
    const bool           output_needs_mapping = true)
{
    if (blockIdx.x >= context.get_num_patches()) {
        return;
    }
    query_block_dispatcher<op, blockThreads>(
        context, blockIdx.x, compute_op, [](uint32_t) { return true; },
        oriented, output_needs_mapping);
}
//**************************************************************************
}  // namespace RXMESH
