#pragma once
#include <stdint.h>

#include <cooperative_groups.h>

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/rxmesh_dynamic.h"

#include "rxmesh/attribute.h"

#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

namespace rxmesh {

template <uint32_t blockThreads>
__global__ static void match_patches_init(const rxmesh::Context context,
                                          const uint16_t        level)
{
    // EV qury to init the patch stash edge weight
    auto ev_update_stash_weight = [&](EdgeHandle e_id, VertexIterator& ev) {
        VertexHandle v0          = ev[0];
        uint32_t     v0_patch_id = v0.patch_id();

        VertexHandle v1          = ev[1];
        uint32_t     v1_patch_id = v1.patch_id();

        PatchInfo* pi_arr = context.m_patches_info;

        // find the boundary edges
        if (v0_patch_id != v1_patch_id) {
            PatchStash& v0_patch_stash = pi_arr[v0_patch_id].patch_stash;
            PatchStash& v1_patch_stash = pi_arr[v1_patch_id].patch_stash;

            // printf("v0_patch_id: %d, v1_patch_id: %d\n", v0_patch_id,
            // v1_patch_id);

            // update edge weight for both patches
            uint8_t v0_stash_idx = v0_patch_stash.find_patch_index(v1_patch_id);
            ::atomicAdd(&(v0_patch_stash.get_edge_weight(v0_stash_idx)), 1);

            uint8_t v1_stash_idx = v1_patch_stash.find_patch_index(v0_patch_id);
            ::atomicAdd(&(v1_patch_stash.get_edge_weight(v1_stash_idx)), 1);

            // printf("v0_stash_idx: %d, v1_stash_idx: %d\n", v0_stash_idx,
            // v1_stash_idx);
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, ev_update_stash_weight);

    // check: print the edge weight
    // uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx == 0) {
    //     for (uint32_t i = 0; i < context.m_num_patches[0]; ++i) {
    //         PatchStash& ps = context.m_patches_info[i].patch_stash;

    //         for (uint16_t j = 0; j < ps.stash_size; ++j) {
    //             printf("j: %d, stash[%d], weight[%d], %d\n",
    //                       j,
    //                    ps.get_patch(j),
    //                    ps.get_edge_weight(j),
    //                    i);
    //         }
    //     }
    // }
}

template <uint32_t blockThreads>
__global__ static void match_patches_patches_select(
    const rxmesh::Context context,
    const uint16_t        level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += gridDim.x * blockDim.x) {
        PatchInfo&  pi          = context.m_patches_info[i];
        PatchStash& patch_stash = pi.patch_stash;

        // select the patches with the highest edge weight
        uint8_t max_weight = 0;
        uint8_t max_idx    = 0;
        for (uint8_t j = 0; j < PatchStash::stash_size; ++j) {
            uint8_t weight = patch_stash.get_edge_weight(j);
            if (weight > max_weight) {
                max_weight = weight;
                max_idx    = j;
            }
        }

        // select the patch with the highest edge weight
        assert(max_weight > 0);
        patch_stash.m_tmp_patch_choice = patch_stash.get_patch(max_idx);
    }
}

template <uint32_t blockThreads>
__global__ static void match_patches_confirm(const rxmesh::Context context,
                                             const uint16_t        level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += gridDim.x * blockDim.x) {
        PatchInfo&  pi          = context.m_patches_info[i];
        PatchStash& patch_stash = pi.patch_stash;

        // confirm the patch choice
        uint32_t patch_choice = patch_stash.m_tmp_patch_choice;
        uint32_t patch_choice_choice =
            context.m_patches_info[patch_choice].patch_stash.m_tmp_patch_choice;

        if (patch_choice_choice == i) {
            patch_stash.m_tmp_is_active = false;
            patch_stash.m_coarse_level_id_list[level] =
                i < patch_choice ? i : patch_choice;
        }
    }
}

template <uint32_t blockThreads>
__global__ static void match_patches_result_update(
    const rxmesh::Context context,
    const uint16_t        level)
{
}

template <uint32_t blockThreads>
__global__ static void generate_patches_ordering(
    const rxmesh::Context             context,
    const uint16_t                    level,
    rxmesh::VertexAttribute<uint16_t> v_ordering)
{
}


}  // namespace rxmesh