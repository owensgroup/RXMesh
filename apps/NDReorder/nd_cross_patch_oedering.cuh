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
__global__ static void match_patches_init_edge_weight(
    const rxmesh::Context context)
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

            // update edge weight for both patches
            uint8_t v0_stash_idx = v0_patch_stash.find_patch_index(v1_patch_id);
            ::atomicAdd(&(v0_patch_stash.get_edge_weight(v0_stash_idx)), 1);
            uint8_t v1_stash_idx = v1_patch_stash.find_patch_index(v0_patch_id);
            ::atomicAdd(&(v1_patch_stash.get_edge_weight(v1_stash_idx)), 1);
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, ev_update_stash_weight);
}

template <uint32_t blockThreads>
__global__ static void match_patches_init_param(const rxmesh::Context context)
{
    // init the patch stash
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += gridDim.x * blockDim.x) {
        PatchStash& ps = context.m_patches_info[i].patch_stash;

        // set level 0
        ps.m_coarse_level_id_list[0]   = i;
        ps.m_coarse_level_pair_list[0] = INVALID32;
        ps.m_coarse_level_num_v[0] = context.m_patches_info[i].num_vertices[0];
        ps.m_is_node               = true;

        ps.m_is_active        = true;
        ps.m_tmp_paired_patch = INVALID32;
        for (uint8_t j = 0; j < ps.stash_size; ++j) {
            if (ps.m_stash[j] == INVALID32) {
                break;
            }

            ps.m_tmp_level_stash[j]       = ps.m_stash[j];
            ps.m_tmp_level_edge_weight[j] = ps.m_edge_weight[j];
        }
    }


    // check: print the edge weight
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
__global__ static void match_patches_select(const rxmesh::Context context,
                                            const uint16_t        level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += gridDim.x * blockDim.x) {
        PatchStash& ps = context.m_patches_info[i].patch_stash;

        // check whether the current patch is active and not merged into other
        if (!ps.m_is_node || !ps.m_is_active) {
            continue;
        }

        // select the patches of the current level with the highest edge weight
        // use tmp that represents the adj list of the current level
        uint32_t max_weight = 0;
        uint8_t  max_idx    = 0;
        for (uint8_t j = 0; j < PatchStash::stash_size; ++j) {
            if (ps.m_tmp_level_stash[j] == INVALID32) {
                break;
            }

            uint8_t  adj_patch_id = ps.m_tmp_level_stash[j];
            uint32_t weight = ps.m_tmp_level_edge_weight[j] * 1000 + adj_patch_id;
            bool     is_active_patch =
                context.m_patches_info[adj_patch_id].patch_stash.m_is_active;
            if (weight > max_weight && is_active_patch) {
                max_weight = weight;
                max_idx    = j;
            }
        }

        uint8_t max_tmp_level_patch_id = ps.m_tmp_level_stash[max_idx];

        // select the patch with the highest edge weight
        if (max_weight <= 0) {
            ps.m_tmp_paired_patch = INVALID32;
        } else {
            ps.m_tmp_paired_patch = max_tmp_level_patch_id;
        }
    }
}

template <uint32_t blockThreads>
__global__ static void match_patches_confirm(const rxmesh::Context context,
                                             const uint16_t        level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += gridDim.x * blockDim.x) {
        PatchStash& ps = context.m_patches_info[i].patch_stash;

        // check whether the current patch is active and not merged into other
        if (!ps.m_is_node || !ps.m_is_active) {
            continue;
        }

        // confirm the patch choice
        uint32_t paired_pid = ps.m_tmp_paired_patch;
        assert(paired_pid != i);
        if (paired_pid == INVALID32) {
            continue;
        }

        uint32_t patch_choice_choice =
            context.m_patches_info[paired_pid].patch_stash.m_tmp_paired_patch;

        // find the matched pair and set inactive
        if (patch_choice_choice == i) {
            ps.m_is_active = false;
        }
    }

    // check patch stash choice
    // if (idx == 0) {
    //     for (int i = 0; i < context.m_num_patches[0]; ++i) {
    //         PatchStash& ps_tmp = context.m_patches_info[i].patch_stash;
    //         printf("i: %d, tmp_patch_choice: %d, active: %d\n",
    //                i,
    //                ps_tmp.m_tmp_paired_patch,
    //                ps_tmp.m_is_active);
    //     }
    //     printf("----------\n");
    // }
}

template <uint32_t blockThreads>
__global__ static void match_patches_update_node(const rxmesh::Context context,
                                                 const uint16_t        level)
{
    // merge the stash and the edge weight
    // one thread per patch
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += gridDim.x * blockDim.x) {
        PatchStash& ps = context.m_patches_info[i].patch_stash;

        if (!ps.m_is_node) {
            // skip the non-node patch
            continue;
        }

        if (ps.m_is_active) {
            // update next level for unmatched patches
            ps.m_coarse_level_id_list[level + 1] =
                ps.m_coarse_level_id_list[level];
            ps.m_coarse_level_pair_list[level + 1] =
                ps.m_coarse_level_pair_list[level];
            ps.m_coarse_level_num_v[level + 1] = ps.m_coarse_level_num_v[level];
        } else {
            // update next level for matched patches
            uint32_t    paired_pid = ps.m_tmp_paired_patch;
            PatchStash& paired_ps =
                context.m_patches_info[paired_pid].patch_stash;
            assert(i != paired_pid);
            if (i < paired_pid) {
                // for the remaining verticies
                ps.m_coarse_level_id_list[level + 1]   = i;
                ps.m_coarse_level_pair_list[level + 1] = paired_pid;
                ps.m_coarse_level_num_v[level + 1] =
                    ps.m_coarse_level_num_v[level] +
                    paired_ps.m_coarse_level_num_v[level];
            } else {
                // for the merged vertices
                ps.m_coarse_level_id_list[level + 1]   = paired_pid;
                ps.m_coarse_level_pair_list[level + 1] = paired_pid;
                ps.m_coarse_level_num_v[level + 1]     = INVALID32;
            }
        }
    }
}

template <uint32_t blockThreads>
__global__ static void match_patches_update_not_node(
    const rxmesh::Context context,
    const uint16_t        level)
{
    // update for non-node patches
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += gridDim.x * blockDim.x) {
        PatchStash& ps = context.m_patches_info[i].patch_stash;

        if (!ps.m_is_node) {
            uint32_t prev_pid = ps.m_coarse_level_id_list[level];
            assert(prev_pid != INVALID32);
            ps.m_coarse_level_id_list[level + 1] =
                context.m_patches_info[prev_pid]
                    .patch_stash.m_coarse_level_id_list[level + 1];
            ps.m_coarse_level_pair_list[level + 1] = INVALID32;
            ps.m_coarse_level_num_v[level + 1]     = INVALID32;

            continue;
        }

        // mark the patch that is merged into other
        if (!ps.m_is_active && i > ps.m_tmp_paired_patch) {
            ps.m_is_node = false;
        }
    }
}


template <uint32_t blockThreads>
__global__ static void match_patches_update_next_level(
    const rxmesh::Context context,
    const uint16_t        level,
    uint16_t*                  num_node)
{
    // update for non-node patches
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += gridDim.x * blockDim.x) {
        PatchStash& ps         = context.m_patches_info[i].patch_stash;
        uint32_t    paired_pid = ps.m_tmp_paired_patch;

        if (i != 0 && ps.m_is_node) {
            atomicAdd(num_node, 1);
        }

        // choose matched node patches
        if (ps.m_is_node) {
            // assert(i < paired_pid);
            // generate the stash and edge weight of next level
            uint32_t next_stash[PatchStash::stash_size];
            uint32_t next_edge_weight[PatchStash::stash_size];
            uint8_t  next_stash_size = 0;

            if (i == 0) {
                for (int m = 0; m < PatchStash::stash_size; ++m) {
                    if (ps.m_tmp_level_stash[m] == INVALID32) {
                        break;
                    }
                    printf("i: %d, stash[%d], weight[%d]\n",
                           i,
                           ps.m_tmp_level_stash[m],
                           ps.m_tmp_level_edge_weight[m]);
                }
            }

            PatchStash& process_ps          = ps;
            auto        generate_next_stash = [&](PatchStash& process_ps) {
                for (uint8_t j = 0; j < PatchStash::stash_size; ++j) {
                    if (process_ps.m_tmp_level_stash[j] == INVALID32) {
                        break;
                    }

                    uint32_t adj_patch_level_id =
                        context.m_patches_info[process_ps.m_tmp_level_stash[j]]
                            .patch_stash.m_coarse_level_id_list[level + 1];

                    // ignore the paired patch which is the same
                    if (adj_patch_level_id == i) {
                        continue;
                    }


                    bool is_duplicate = false;
                    for (uint8_t k = 0; k < next_stash_size; ++k) {
                        if (next_stash[k] == adj_patch_level_id) {
                            next_edge_weight[k] +=
                                process_ps.m_tmp_level_edge_weight[j];
                            is_duplicate = true;
                            break;
                        }
                    }

                    if (!is_duplicate) {
                        next_stash[next_stash_size] = adj_patch_level_id;
                        next_edge_weight[next_stash_size] =
                            process_ps.m_tmp_level_edge_weight[j];
                        ++next_stash_size;
                    }
                }
            };

            generate_next_stash(ps);

            // extra merge for matched patches
            if (!ps.m_is_active) {
                PatchStash& paired_ps =
                    context.m_patches_info[paired_pid].patch_stash;
                generate_next_stash(paired_ps);
            }

            // update to the next level
            for (uint8_t j = 0; j < PatchStash::stash_size; ++j) {
                if (ps.get_tmp_level_patch(j) == INVALID32) {
                    break;
                }
                ps.m_tmp_level_stash[j]       = INVALID32;
                ps.m_tmp_level_edge_weight[j] = INVALID32;
            }

            for (uint8_t j = 0; j < next_stash_size; ++j) {
                ps.m_tmp_level_stash[j]       = next_stash[j];
                ps.m_tmp_level_edge_weight[j] = next_edge_weight[j];
            }

            ps.m_is_active = true;
        }
    }
}

template <uint32_t blockThreads>
__global__ static void check(const rxmesh::Context context,
                             const uint16_t        level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // check patch stash choice
        for (int i = 0; i < context.m_num_patches[0]; ++i) {
            PatchStash& ps_tmp = context.m_patches_info[i].patch_stash;
            printf(
                "i: %d, level: %d, level_id: %d, pair: %d, num_v: %d, "
                "is_node: %d, is_active: %d\n",
                i,
                level,
                ps_tmp.m_coarse_level_id_list[level + 1],
                ps_tmp.m_coarse_level_pair_list[level + 1],
                ps_tmp.m_coarse_level_num_v[level + 1],
                ps_tmp.m_is_node,
                ps_tmp.m_is_active);
        }
        printf("--show--\n");
        for (int i = 0; i < context.m_num_patches[0]; ++i) {
            PatchStash& ps_tmp = context.m_patches_info[i].patch_stash;
            if (ps_tmp.m_is_node) {
                for (uint16_t j = 0; j < ps_tmp.stash_size; ++j) {
                    if (ps_tmp.m_tmp_level_stash[j] == INVALID32) {
                        break;
                    }
                    printf("i: %d, stash[%d], weight[%d]\n",
                           i,
                           ps_tmp.m_tmp_level_stash[j],
                           ps_tmp.m_tmp_level_edge_weight[j]);
                }
            }
        }
        printf("----------\n");
    }
}


template <uint32_t blockThreads>
__global__ static void match_patches_extract_vertices(
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