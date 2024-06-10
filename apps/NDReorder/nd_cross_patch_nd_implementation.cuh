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

// write to patch stash
template <uint32_t blockThreads>
__global__ static void nd_init_edge_weight(const rxmesh::Context& context)
{
    // EV qury to init the patch stash edge weight
    auto ev_update_stash_weight = [&](EdgeHandle e_id, VertexIterator& ev) {
        // VertexHandle v0          = ev[0];
        // uint32_t     v0_patch_id = v0.patch_id();

        // VertexHandle v1          = ev[1];
        // uint32_t     v1_patch_id = v1.patch_id();

        // PatchInfo* pi_arr = context.m_patches_info;

        // // find the boundary edges
        // if (v0_patch_id != v1_patch_id) {
        //     PatchStash& v0_patch_stash = pi_arr[v0_patch_id].patch_stash;
        //     PatchStash& v1_patch_stash = pi_arr[v1_patch_id].patch_stash;

        //     // update edge weight for both patches
        //     uint8_t v0_stash_idx = v0_patch_stash.find_patch_index(v1_patch_id);
        //     ::atomicAdd(&(v0_patch_stash.get_edge_weight(v0_stash_idx)), 1);
        //     uint8_t v1_stash_idx = v1_patch_stash.find_patch_index(v0_patch_id);
        //     ::atomicAdd(&(v1_patch_stash.get_edge_weight(v1_stash_idx)), 1);
        // }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, ev_update_stash_weight);
}

template <uint32_t blockThreads>
__global__ static void bipartition_init_seed(const rxmesh::Context& context,
                                             uint32_t*              frontiers,
                                             uint32_t* frontier_head,
                                             uint32_t* frontier_size,
                                             uint16_t  partition_label,
                                             uint32_t* d_patch_label)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += blockDim.x * gridDim.x) {
        if (d_patch_label[i] != partition_label) {
            // filter out not active patches for this bipartition
            continue;
        }

        PatchStash& ps = context.m_patches_info[i].patch_stash;

        // set the initial status of seed
        for (uint32_t j = frontier_head[0];
             j < frontier_head[0] + frontier_size[0];
             j++) {
            if (i == frontiers[j]) {
                ps.m_is_seed     = true;
                ps.m_is_frontier = true;
                ps.m_settle_id   = j;
                break;
            }
        }
    }
}

template <uint32_t blockThreads>
__global__ static void bipartition_propogation(const rxmesh::Context& context,
                                               uint32_t*              frontiers,
                                               uint32_t* frontier_head,
                                               uint32_t* frontier_size,
                                               uint32_t* new_frontier_head,
                                               uint32_t* new_frontier_size,
                                               uint16_t  partition_label,
                                               uint32_t* d_patch_label)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += blockDim.x * gridDim.x) {
        if (d_patch_label[i] != partition_label) {
            // filter out not active patches for this bipartition
            continue;
        }

        PatchStash& ps = context.m_patches_info[i].patch_stash;

        bool is_frontier = false;
        for (uint32_t j = frontier_head[0];
             j < frontier_head[0] + frontier_size[0];
             j++) {
            if (i == frontiers[j]) {
                is_frontier      = true;
                ps.m_is_frontier = true;
                break;
            }
        }

        // expand frontier
        if (is_frontier) {
            for (uint32_t j = 0; j < ps.stash_size; j++) {
                if (ps.m_stash[j] == INVALID32) {
                    break;
                }

                uint32_t    adj_patch_id = ps.m_stash[j];
                PatchStash& adj_ps =
                    context.m_patches_info[adj_patch_id].patch_stash;

                // skip if not in the same partition
                if (d_patch_label[adj_patch_id] != partition_label) {
                    continue;
                }

                // skip if already settled or seed
                if (adj_ps.m_is_seed || adj_ps.m_settle_id != INVALID32) {
                    continue;
                }

                // update frontier
                uint32_t frontier_idx   = ::atomicAdd(new_frontier_size, 1);
                frontiers[frontier_idx] = adj_patch_id;
                adj_ps.m_is_frontier    = true;
                adj_ps.m_settle_id      = ps.m_settle_id;
            }

            ps.m_is_frontier = false;
        }
    }
}

template <uint32_t blockThreads>
__global__ static void bipartition_recenter(const rxmesh::Context context,
                                            uint32_t*             d_patch_label)
{
}

__global__ static void check_patch_stash(const rxmesh::Context context)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (uint32_t i = 0; i < context.m_num_patches[0]; i++) {
            PatchInfo*  pi = context.m_patches_info + i;
            PatchStash& ps = pi->patch_stash;

            printf("Patch %d\n", i);
            for (uint32_t j = 0; j < ps.stash_size; j++) {
                if (ps.m_stash[j] == INVALID32) {
                    break;
                }
                printf(" stash: %d, weight: %d\n",
                       ps.m_stash[j],
                       ps.get_edge_weight(j));
            }
        }
    }
}

template <uint32_t blockThreads>
void run_bipartition_lloyd(RXMeshStatic& rx,
                           uint16_t      partition_label,
                           uint32_t*     d_patch_label)
{
    const uint32_t threads_p = blockThreads;
    const uint32_t blocks_p  = DIVIDE_UP(rx.get_num_patches(), threads_p);

    // get random seed for 2
    uint16_t              num_seeds = 2;
    std::vector<uint32_t> components;
    uint32_t*             frontiers;
    CUDA_ERROR(
        cudaMallocManaged(&frontiers, rx.get_num_patches() * sizeof(uint32_t)));
    uint32_t* frontier_head;
    CUDA_ERROR(cudaMallocManaged(&frontier_head, sizeof(uint32_t)));
    uint32_t* frontier_size;
    CUDA_ERROR(cudaMallocManaged(&frontier_size, sizeof(uint32_t)));
    uint32_t* new_frontier_head;
    CUDA_ERROR(cudaMallocManaged(&new_frontier_head, sizeof(uint32_t)));
    uint32_t* new_frontier_size;
    CUDA_ERROR(cudaMallocManaged(&new_frontier_size, sizeof(uint32_t)));

    for (uint32_t i = 0; i < rx.get_num_patches(); i++) {
        if (d_patch_label[i] == partition_label) {
            components.push_back(i);
        }
    }

    // set the first frontier to be two seeds
    random_shuffle(components.data(), components.size());
    frontiers[0] = components[0];
    frontiers[1] = components[1];

    bipartition_init_seed<blockThreads>
        <<<blocks_p, threads_p>>>(rx.get_context(),
                                  frontiers,
                                  frontier_head,
                                  frontier_size,
                                  partition_label,
                                  d_patch_label);

    // propogation until all the patches are assigned to a partition
    while (true) {
        new_frontier_head[0] = frontier_head[0] + frontier_size[0];
        new_frontier_size[0] = 0;
        // bipartition_propogation<blockThreads><<<blocks_p,
        // threads_p>>>(rx.get_context(),
        //                                                  frontiers,
        //                                                  frontier_head,
        //                                                  frontier_size,
        //                                                  new_frontier_head,
        //                                                  new_frontier_size,
        //                                                  partition_label,
        //                                                  d_patch_label);
        break;
    }

    // propogation to get centroids
    //  bipartition_recenter(rx, frontiers, d_patch_label);

    // repeat the process until convergence

    // refinement on the boundary
}


template <uint32_t blockThreads>
__global__ static void nd_extract_vertices(const rxmesh::Context     context,
                                           VertexAttribute<uint16_t> v_ordering,
                                           uint32_t* d_v_ordering_prefix_sum,
                                           uint32_t* d_v_ordering_spv_idx)
{
    // VV qury to extract the vertex separators
    auto vv_extract_separartors = [&](VertexHandle v_id, VertexIterator& vv) {
        uint32_t v_patch_id = v_id.patch_id();

        bool is_separator = false;
        for (uint16_t i = 0; i < vv.size(); ++i) {
            VertexHandle adj_v_id     = vv[i];
            uint32_t     adj_patch_id = adj_v_id.patch_id();

            if (adj_patch_id > v_patch_id) {
                is_separator = true;
            }
        }

        uint32_t v_order = INVALID32;
        if (is_separator) {
            v_order = ::atomicAdd((unsigned int*)&d_v_ordering_prefix_sum
                                      [d_v_ordering_spv_idx[0]],
                                  (unsigned int)1);
        } else {
            v_order =
                ::atomicAdd((unsigned int*)&d_v_ordering_prefix_sum[v_patch_id],
                            (unsigned int)1);
        }
        assert(v_order != INVALID32);

        v_ordering(v_id, 0) = v_order;
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, vv_extract_separartors);
}

template <uint32_t blockThreads>
__global__ static void nd_assign_numbering(const RXMeshStatic&       rx,
                                           const rxmesh::Context     context,
                                           VertexAttribute<uint16_t> v_ordering,
                                           uint32_t* d_v_ordering_prefix_sum,
                                           uint32_t* d_v_ordering_spv_idx)
{
    // VV qury to asssign the numbering to the vertices
    auto vv_assign_numbering = [&](VertexHandle v_id, VertexIterator& vv) {
        uint32_t v_patch_id = v_id.patch_id();

        bool is_separator = false;
        for (uint16_t i = 0; i < vv.size(); ++i) {
            VertexHandle adj_v_id     = vv[i];
            uint32_t     adj_patch_id = adj_v_id.patch_id();

            if (adj_patch_id > v_patch_id) {
                is_separator = true;
            }
        }

        if (is_separator) {
            v_ordering(v_id, 0) +=
                d_v_ordering_prefix_sum[d_v_ordering_spv_idx[0]];
        } else {
            v_ordering(v_id, 0) += d_v_ordering_prefix_sum[v_patch_id];
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, vv_assign_numbering);
}

void nd_reorder_test(RXMeshStatic& rx)
{
    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> launch_box_nd_init_edge_weight;
    rx.prepare_launch_box({rxmesh::Op::EV},
                          launch_box_nd_init_edge_weight,
                          (void*)nd_init_edge_weight<blockThreads>);

    nd_init_edge_weight<blockThreads>
        <<<launch_box_nd_init_edge_weight.blocks,
           launch_box_nd_init_edge_weight.num_threads,
           launch_box_nd_init_edge_weight.smem_bytes_dyn>>>(rx.get_context());

    check_patch_stash<<<1, 1>>>(rx.get_context());
}

void nd_reorder(RXMeshStatic& rx, uint32_t* ordering_arr, uint16_t nd_level)
{
    constexpr uint32_t blockThreads = 256;

    // vertex color attribute
    auto v_ordering = rx.add_vertex_attribute<uint16_t>("v_ordering", 1);

    uint32_t num_v_separator = (1 << nd_level) - 1;
    uint32_t v_ordering_prefix_sum_size =
        rx.get_num_patches() + 1 + num_v_separator;
    uint32_t* d_v_ordering_prefix_sum;
    cudaMallocManaged(&d_v_ordering_prefix_sum,
                      v_ordering_prefix_sum_size * sizeof(uint32_t));
    cudaMemset(d_v_ordering_prefix_sum,
               0,
               v_ordering_prefix_sum_size * sizeof(uint32_t));

    uint32_t* d_v_ordering_spv_idx;
    cudaMallocManaged(&d_v_ordering_spv_idx,
                      v_ordering_prefix_sum_size * sizeof(uint32_t));
    d_v_ordering_spv_idx[0] = rx.get_num_patches();

    uint32_t* d_patch_label;
    cudaMallocManaged(&d_patch_label, rx.get_num_patches() * sizeof(uint32_t));
    cudaMemset(d_patch_label, 0, rx.get_num_patches() * sizeof(uint32_t));

    uint32_t blocks  = rx.get_num_patches();
    uint32_t threads = blockThreads;

    LaunchBox<blockThreads> launch_box_extract_vertices;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_extract_vertices,
                          (void*)nd_extract_vertices<blockThreads>);
    LaunchBox<blockThreads> launch_box_assign_numbering;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_assign_numbering,
                          (void*)nd_assign_numbering<blockThreads>);

    // main body here

    for (uint32_t i = 0; i < nd_level; i++) {
        //partition for each label
        uint32_t num_label = 1 << i;
        uint32_t next_num_label = 1 << (i + 1);
        
        // update the prefix_sum location
        // += num_label

        for (uint32_t j = 0; j < num_label; j++) {
            run_bipartition_lloyd<blockThreads>(rx, j, d_patch_label);

            // extract vertices
        }

        // calculate prefix sum

        // assign numbering
    }


    // verify the result
    // v_ordering->move(rxmesh::DEVICE, rxmesh::HOST);
    // rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
    //     uint32_t v_global_id = rx.map_to_global(vh);
    //     uint32_t v_linea_id  = rx.linear_id(vh);
    //     uint32_t v_order_idx = (*v_ordering)(vh, 0);

    //     ordering_arr[v_order_idx] = v_global_id;
    // });
}

}  // namespace rxmesh