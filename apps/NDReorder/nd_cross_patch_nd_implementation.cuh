#pragma once
#include <stdint.h>

#include <cooperative_groups.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/rxmesh_dynamic.h"

#include "rxmesh/attribute.h"

#include "rxmesh/query.cuh"


namespace rxmesh {

/**
 * @brief Initializes the edge weights of patches in a mesh.
 *
 * Device function that updates the edge weights of patches in a mesh. The
 * results are stored in the patch stash of the patches. The patch adjacency
 * information is also stored in the patch stash.
 *
 */
template <uint32_t blockThreads>
__global__ static void nd_init_edge_weight(const rxmesh::Context context)
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

// TODO use shared alloc for this instead of static shared memory
// TODO: check correctness of compilation
template <uint32_t blockThreads>
__global__ void bipartition_init_random_seeds(
    uint32_t* patch_partition_label,
    uint16_t* patch_seed_label_count,
    uint16_t* patch_local_partition_role,
    uint16_t* patch_local_partition_label,
    uint32_t* seeds,
    uint32_t  num_patches,
    uint32_t  num_labels)
{
    __shared__ uint32_t shared_indices[8192];  // Adjust size as needed
    __shared__ uint32_t count;
    __shared__ uint32_t selected[2];

    curandState state;

    for (uint32_t label = blockIdx.x; label < num_labels; label += gridDim.x) {
        if (threadIdx.x == 0) {
            count       = 0;
            selected[0] = selected[1] = INVALID32;  // Invalid index
            curand_init(clock64() + label, 0, 0, &state);
        }
        __syncthreads();

        // Collect indices
        for (uint32_t i = threadIdx.x; i < num_patches; i += blockDim.x) {
            if (patch_partition_label[i] == label) {
                uint32_t idx =
                    ::atomicAdd((unsigned int*)&count, (unsigned int)1);
                if (idx < 8192)
                    shared_indices[idx] = i;
            }
        }
        __syncthreads();

        if (count < 2) {
            if (threadIdx.x == 0) {
                printf("Error: Label %d does not have at least two indices.\n",
                       label);
            }
            continue;
        }

        // Select two random indices
        if (threadIdx.x == 0) {
            uint32_t idx1 = curand(&state) % min(count, 1024U);
            uint32_t idx2;
            do {
                idx2 = curand(&state) % min(count, 1024U);
            } while (idx2 == idx1);

            selected[0] = shared_indices[idx1];
            selected[1] = shared_indices[idx2];

            // set the role to be 2 meanning seed, 0 meanning normal patches
            patch_local_partition_role[selected[0]] = 2;
            patch_local_partition_role[selected[1]] = 2;

            // set the local label to be 0 and 1 while unlabelled is INVALID16
            patch_local_partition_label[selected[0]] = 0;
            patch_local_partition_label[selected[1]] = 1;

            patch_seed_label_count[(label << 1) + 0] = 1;
            patch_seed_label_count[(label << 1) + 1] = 1;

            seeds[(label << 1) + 0] = selected[0];
            seeds[(label << 1) + 1] = selected[1];
        }
        __syncthreads();
    }
}

template <uint32_t blockThreads>
__global__ static void bipartition_seed_propogation(
    const rxmesh::Context context,
    uint32_t*             d_patch_partition_label,
    uint16_t*             d_patch_local_partition_role,
    uint16_t*             d_patch_local_partition_label,
    uint16_t*             d_patch_local_seed_label_count,
    uint32_t*             d_patch_local_frontiers,
    uint32_t*             d_frontier_head,
    uint32_t*             d_frontier_size,
    uint32_t*             d_new_frontier_head,
    uint32_t*             d_new_frontier_size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for debug only
    // for (uint32_t i = 0; i < context.m_num_patches[0]; i++) {
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += blockDim.x * gridDim.x) {

        PatchStash& ps = context.m_patches_info[i].patch_stash;

        bool is_frontier = false;
        for (uint32_t j = d_frontier_head[0];
             j < d_frontier_head[0] + d_frontier_size[0];
             j++) {
            if (i == d_patch_local_frontiers[j]) {
                is_frontier = true;
                break;
            }
        }

        // expand frontier
        if (is_frontier) {
            for (uint32_t j = 0; j < ps.stash_size; j++) {
                if (ps.m_stash[j] == INVALID32) {
                    break;
                }

                uint32_t adj_patch_id = ps.m_stash[j];

                // skip if not in the same partition
                // TODO: check the multi-partition phase
                if (d_patch_partition_label[adj_patch_id] !=
                    d_patch_partition_label[i]) {
                    // printf(" adj patch %d is not in the same partition\n",
                    //          adj_patch_id);
                    continue;
                }

                // skip if it is seed
                if (d_patch_local_partition_role[adj_patch_id] == 2) {
                    // printf(" adj patch %d is seed\n",
                    //        adj_patch_id);
                    continue;
                }

                uint16_t old_id =
                    atomicCAS(&d_patch_local_partition_label[adj_patch_id],
                              INVALID16,
                              d_patch_local_partition_label[i]);

                if (old_id == INVALID16) {
                    // update frontier only when it's not settled
                    uint32_t curr_frontier_size =
                        ::atomicAdd(&d_new_frontier_size[0], 1);
                    d_patch_local_frontiers[d_new_frontier_head[0] +
                                            curr_frontier_size] = adj_patch_id;

                    // update the seed count
                    uint32_t patch_seed_label =
                        (d_patch_partition_label[i] << 1) +
                        d_patch_local_partition_label[i];

                    atomicAdd(&d_patch_local_seed_label_count[patch_seed_label],
                              1);
                }
            }
        }
    }
}

template <uint32_t blockThreads>
__global__ static void bipartition_seed_propogation_refined(
    const rxmesh::Context context,
    uint32_t*             d_patch_partition_label,
    uint16_t*             d_patch_local_partition_role,
    uint16_t*             d_patch_local_partition_label,
    uint16_t*             d_patch_local_seed_label_count,
    uint32_t*             d_patch_local_frontiers,
    uint32_t*             d_frontier_head,
    uint32_t*             d_frontier_size,
    uint32_t*             d_new_frontier_head,
    uint32_t*             d_new_frontier_size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for debug only
    // for (uint32_t i = 0; i < context.m_num_patches[0]; i++) {
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += blockDim.x * gridDim.x) {

        PatchStash& ps = context.m_patches_info[i].patch_stash;

        if (d_patch_local_partition_label[i] != INVALID16) {
            ::atomicAdd(&d_new_frontier_size[0], 1);
            // printf("check: patch %d is already labeled\n", i);
            continue;
        }

        uint32_t seed_label0_count = 0;
        uint32_t seed_label1_count = 0;

        for (uint32_t j = 0; j < ps.stash_size; j++) {
            if (ps.m_stash[j] == INVALID32) {
                break;
            }

            uint32_t adj_patch_id = ps.m_stash[j];

            // skip if not in the same partition
            if (d_patch_partition_label[adj_patch_id] !=
                d_patch_partition_label[i]) {
                // printf(" adj patch %d is not in the same partition\n",
                //          adj_patch_id);
                continue;
            }

            if (d_patch_local_partition_label[adj_patch_id] == 0) {
                seed_label0_count++;
            } else if (d_patch_local_partition_label[adj_patch_id] == 1) {
                seed_label1_count++;
            }
        }

        if (seed_label0_count == 0 && seed_label1_count == 0) {
            // printf("Check: patch %d is isolated for now\n", i);
            continue;
        }

        if (seed_label0_count > seed_label1_count) {
            d_patch_local_partition_label[i] = 0;
            atomicAdd(&d_patch_local_seed_label_count[(d_patch_partition_label[i] << 1) + 0], 1);
        } else {
            d_patch_local_partition_label[i] = 1;
            atomicAdd(&d_patch_local_seed_label_count[(d_patch_partition_label[i] << 1) + 1], 1);
        }
    }
}

template <uint32_t blockThreads>
__global__ static void bipartition_mark_boundary(
    const rxmesh::Context context,
    uint32_t*             d_patch_partition_label,
    uint16_t*             d_patch_local_seed_label_count,
    uint16_t*             d_patch_local_partition_role,
    uint16_t*             d_patch_local_partition_label)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += blockDim.x * gridDim.x) {
        PatchStash& ps = context.m_patches_info[i].patch_stash;

        // skip the unlabled patch
        if (d_patch_local_partition_role[i] == INVALID16) {
            continue;
        }

        // selecct the vertices that is adjacent to other vertices
        bool is_boundary = false;
        for (uint32_t j = 0; j < ps.stash_size; j++) {
            if (ps.m_stash[j] == INVALID32) {
                break;
            }

            uint32_t adj_patch_id = ps.m_stash[j];

            if (d_patch_partition_label[adj_patch_id] !=
                d_patch_partition_label[i]) {
                // printf(" adj patch %d is not in the same partition\n",
                //        adj_patch_id);
                continue;
            }

            if (d_patch_local_partition_label[adj_patch_id] !=
                d_patch_local_partition_label[i]) {
                is_boundary = true;
            }
        }

        if (is_boundary) {
            d_patch_local_partition_role[i] = 1;
        }
    }
}

template <uint32_t blockThreads>
__global__ static void bipartition_seed_recenter(
    const rxmesh::Context context,
    uint32_t*             d_patch_partition_label,
    uint16_t*             d_patch_local_seed_label_count,
    uint16_t*             d_patch_local_partition_role,
    uint16_t*             d_patch_local_partition_label,
    uint32_t*             seeds,
    uint32_t*             num_seeds)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += blockDim.x * gridDim.x) {

        PatchStash& ps = context.m_patches_info[i].patch_stash;

        // skip the unlabled patch
        if (d_patch_local_partition_role[i] == INVALID16) {
            continue;
        }

        uint32_t patch_seed_label = (d_patch_partition_label[i] << 1) +
                                    d_patch_local_partition_label[i];

        if (d_patch_local_partition_role[i] = 1 &&
            d_patch_local_seed_label_count[patch_seed_label] > 0) {
            uint16_t remaining_count = atomicAdd(
                &d_patch_local_seed_label_count[patch_seed_label], -1);

            if (remaining_count == 1) {
                // the new seed is selected!
                d_patch_local_partition_role[i]                  = 2;
                d_patch_local_seed_label_count[patch_seed_label] = 1;
                seeds[patch_seed_label]                          = i;

                ::atomicAdd(&num_seeds[0], 1);
            } else {
                // proceed until the seed is found
                d_patch_local_partition_role[i]  = 0;
                d_patch_local_partition_label[i] = INVALID16;
            }
        }
    }
}

template <uint32_t blockThreads>
__global__ static void bipartition_refine_partition(
    const rxmesh::Context context,
    uint32_t*             d_tmp_local_seed_label_count_difference,
    uint32_t*             d_patch_partition_label,
    uint16_t*             d_patch_local_seed_label_count,
    uint16_t*             d_patch_local_partition_role,
    uint16_t*             d_patch_local_partition_label)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += blockDim.x * gridDim.x) {
        PatchStash& ps = context.m_patches_info[i].patch_stash;

        // skip the unlabled patch
        if (d_patch_local_partition_role[i] == INVALID16) {
            printf("Error: patch %d is not labeled\n", i);
            assert(false && "Error: patch is not labeled when refining\n");
        }

        uint32_t patch_seed_label = (d_patch_partition_label[i] << 1) +
                                    d_patch_local_partition_label[i];

        uint32_t patch_another_seed_label =
            (d_patch_partition_label[i] << 1) +
            (d_patch_local_partition_label[i] ^ 1);

        if (d_patch_local_partition_role[i] == 1 &&
            d_tmp_local_seed_label_count_difference[patch_seed_label] != 0 &&
            d_tmp_local_seed_label_count_difference[patch_seed_label] <
                (INVALID32 - 1024) &&
            ::atomicAdd(
                &d_tmp_local_seed_label_count_difference[patch_seed_label],
                -1) < (INVALID32 - 1024)) {
            d_patch_local_partition_label[i] ^= 1;
            atomicAdd(&d_patch_local_seed_label_count[patch_seed_label], -1);
            atomicAdd(&d_patch_local_seed_label_count[patch_another_seed_label],
                      1);
        }
    }
}

// TODO: becareful with the pair or triple isolated patches
template <uint32_t blockThreads>
__global__ static void bipartition_check_isolation(
    const rxmesh::Context context,
    uint32_t*             d_patch_partition_label,
    uint16_t*             d_patch_local_seed_label_count,
    uint16_t*             d_patch_local_partition_role,
    uint16_t*             d_patch_local_partition_label)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;


    for (uint32_t i = idx; i < context.m_num_patches[0];
         i += blockDim.x * gridDim.x) {

        PatchStash& ps = context.m_patches_info[i].patch_stash;

        // printf("same label same seed: ");
        bool is_isolated = true;
        for (uint32_t j = 0; j < ps.stash_size; j++) {
            if (ps.m_stash[j] == INVALID32) {
                break;
            }

            uint32_t    adj_patch_id = ps.m_stash[j];

            if (d_patch_partition_label[adj_patch_id] !=
                d_patch_partition_label[i]) {
                // printf(" adj patch %d is not in the same
                // partition\n",
                //        adj_patch_id);
                continue;
            }

            if (d_patch_local_partition_label[adj_patch_id] ==
                d_patch_local_partition_label[i]) {
                is_isolated = false;
            }
        }

        if (is_isolated) {
            uint32_t patch_seed_label = (d_patch_partition_label[i] << 1) +
                                        d_patch_local_partition_label[i];

            uint32_t patch_another_seed_label =
                (d_patch_partition_label[i] << 1) +
                (d_patch_local_partition_label[i] ^ 1);

            printf("Check: patch %d is isolated\n", i);
            d_patch_local_partition_label[i] ^= 1;
            atomicAdd(&d_patch_local_seed_label_count[patch_seed_label], -1);
            atomicAdd(&d_patch_local_seed_label_count[patch_another_seed_label],
                      1);
        }
    }
}

template <uint32_t blockThreads>
__global__ static void bipartition_check_propogation(
    const rxmesh::Context context,
    uint32_t*             d_patch_partition_label,
    uint16_t*             d_patch_local_seed_label_count,
    uint16_t*             d_patch_local_partition_role,
    uint16_t*             d_patch_local_partition_label,
    uint32_t*             seeds,
    uint32_t*             num_seeds)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        for (uint32_t i = 0; i < context.m_num_patches[0]; i += 1) {

            PatchStash& ps = context.m_patches_info[i].patch_stash;

            // process the unlabled patch
            if (d_patch_local_partition_label[i] == INVALID16) {
                printf("Error: patch %d is not labeled\n", i);

                printf("different label: ");
                for (uint32_t j = 0; j < ps.stash_size; j++) {
                    if (ps.m_stash[j] == INVALID32) {
                        break;
                    }

                    uint32_t    adj_patch_id = ps.m_stash[j];

                    if (d_patch_partition_label[adj_patch_id] !=
                        d_patch_partition_label[i]) {
                        // printf(" adj patch %d is not in the same
                        // partition\n",
                        //        adj_patch_id);
                        printf(" %d", adj_patch_id);
                    }

                    // if (d_patch_local_partition_label[adj_patch_id] !=
                    //     d_patch_local_partition_label[i]) {
                    //     is_boundary = true;
                    // }
                }
                printf("\n");


                printf("same label different seed: ");
                for (uint32_t j = 0; j < ps.stash_size; j++) {
                    if (ps.m_stash[j] == INVALID32) {
                        break;
                    }

                    uint32_t    adj_patch_id = ps.m_stash[j];

                    if (d_patch_partition_label[adj_patch_id] !=
                        d_patch_partition_label[i]) {
                        // printf(" adj patch %d is not in the same
                        // partition\n",
                        //        adj_patch_id);
                        continue;
                    }

                    if (d_patch_local_partition_label[adj_patch_id] !=
                        d_patch_local_partition_label[i]) {
                        printf(" %d", adj_patch_id);
                    }
                }
                printf("\n");

                printf("same label same seed: ");
                for (uint32_t j = 0; j < ps.stash_size; j++) {
                    if (ps.m_stash[j] == INVALID32) {
                        break;
                    }

                    uint32_t    adj_patch_id = ps.m_stash[j];

                    if (d_patch_partition_label[adj_patch_id] !=
                        d_patch_partition_label[i]) {
                        // printf(" adj patch %d is not in the same
                        // partition\n",
                        //        adj_patch_id);
                        continue;
                    }

                    if (d_patch_local_partition_label[adj_patch_id] ==
                        d_patch_local_partition_label[i]) {
                        printf(" %d", adj_patch_id);
                    }
                }
                printf("\n");
            }
        }
    }
}

template <typename T>
__global__ static void check_d_arr(T* d_arr, uint32_t size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (uint32_t i = 0; i < size; i++) {
            printf(" %u,", d_arr[i]);
        }
        printf("\n");
    }
}

// Bisecting k-means instead os normal k-means for initialization?
// now just normal k-means
// 1. how good is bi-secting k-means
// 2. how slow is bi-secting k-means
template <uint32_t blockThreads>
void run_partition_lloyd(RXMeshStatic& rx,
                         uint16_t      nd_level,
                         uint32_t*     d_patch_partition_label,
                         uint16_t*     d_patch_local_seed_label_count,
                         uint16_t*     d_patch_local_partition_role,
                         uint16_t*     d_patch_local_partition_label,
                         uint32_t*     d_patch_local_frontiers,
                         uint32_t*     d_frontier_head,
                         uint32_t*     d_frontier_size,
                         uint32_t*     d_new_frontier_head,
                         uint32_t*     d_new_frontier_size)
{
    // sanity check that the label the num matches

    const uint32_t lloyd_iter_limit = 5;
    const uint32_t threads_p        = blockThreads;
    const uint32_t blocks_p = DIVIDE_UP(rx.get_num_patches(), threads_p);

    uint16_t curr_num_partitions = 1 << nd_level;
    uint16_t curr_num_seeds      = curr_num_partitions * 2;

    for (uint32_t i = 0; i < rx.get_num_patches(); i++) {
        assert(d_patch_partition_label[i] < curr_num_partitions);
    }

    printf("---------- lloys start with %d seed ----------\n", curr_num_seeds);

    // initialize the tmp parameters, set by bytes
    cudaMemset(d_patch_local_partition_role,
               0,
               rx.get_num_patches() * sizeof(uint16_t));
    cudaMemset(d_patch_local_partition_label,
               INVALID16,
               rx.get_num_patches() * sizeof(uint16_t));
    cudaMemset(d_patch_local_frontiers,
               INVALID32,
               rx.get_num_patches() * sizeof(uint32_t));
    cudaMemset(d_patch_local_seed_label_count,
               INVALID16,
               rx.get_num_patches() * sizeof(uint16_t));

    bipartition_init_random_seeds<blockThreads>
        <<<rx.get_num_patches() + 1, threads_p>>>(
            d_patch_partition_label,
            d_patch_local_seed_label_count,
            d_patch_local_partition_role,
            d_patch_local_partition_label,
            d_patch_local_frontiers,
            rx.get_num_patches(),
            curr_num_partitions);

    // set the frontier index
    d_frontier_head[0]     = 0;
    d_frontier_size[0]     = curr_num_seeds;
    d_new_frontier_head[0] = d_frontier_head[0] + d_frontier_size[0];
    d_new_frontier_size[0] = 0;

    CUDA_ERROR(cudaDeviceSynchronize());

    // printf("d_patch_partition_label: ");
    // check_d_arr<<<1, 1>>>(d_patch_partition_label, rx.get_num_patches());
    // CUDA_ERROR(cudaDeviceSynchronize());

    // printf("d_patch_local_frontiers: ");
    // check_d_arr<<<1, 1>>>(d_patch_local_frontiers, rx.get_num_patches());
    // CUDA_ERROR(cudaDeviceSynchronize());

    // printf("d_patch_local_partition_role: ");
    // check_d_arr<<<1, 1>>>(d_patch_local_partition_role, rx.get_num_patches());
    // CUDA_ERROR(cudaDeviceSynchronize());

    // printf("d_patch_local_partition_label: ");
    // check_d_arr<<<1, 1>>>(d_patch_local_partition_label, rx.get_num_patches());
    // CUDA_ERROR(cudaDeviceSynchronize());

    // printf("d_patch_local_seed_label_count: ");
    // check_d_arr<<<1, 1>>>(d_patch_local_seed_label_count, rx.get_num_patches());
    // CUDA_ERROR(cudaDeviceSynchronize());

    uint32_t lloyd_iter = 0;
    while (lloyd_iter < lloyd_iter_limit) {

        uint32_t tmp_break_counter = 0;

        // recenter to regenerate the seed when it's greater than 0
        if (lloyd_iter > 0) {
            // printf("---------- recenter ----------\n");

            // reset seed
            cudaMemset(d_patch_local_frontiers,
                       INVALID32,
                       rx.get_num_patches() * sizeof(uint32_t));


            // reset the seeds, essentially the same as the initialization
            while (true) {
                d_frontier_size[0] = 0;
                bipartition_mark_boundary<blockThreads>
                    <<<blocks_p, threads_p>>>(rx.get_context(),
                                              d_patch_partition_label,
                                              d_patch_local_seed_label_count,
                                              d_patch_local_partition_role,
                                              d_patch_local_partition_label);
                CUDA_ERROR(cudaDeviceSynchronize());

                bipartition_seed_recenter<blockThreads>
                    <<<blocks_p, threads_p>>>(rx.get_context(),
                                              d_patch_partition_label,
                                              d_patch_local_seed_label_count,
                                              d_patch_local_partition_role,
                                              d_patch_local_partition_label,
                                              d_patch_local_frontiers,
                                              d_frontier_size);
                CUDA_ERROR(cudaDeviceSynchronize());

                // prints
                // printf("--- recenter iter ---\n");
                // printf("d_patch_local_partition_role: ");
                // check_d_arr<<<1, 1>>>(d_patch_local_partition_role,
                //                       rx.get_num_patches());
                // CUDA_ERROR(cudaDeviceSynchronize());

                // printf("d_patch_local_partition_label: ");
                // check_d_arr<<<1, 1>>>(d_patch_local_partition_label,
                //                       rx.get_num_patches());
                // CUDA_ERROR(cudaDeviceSynchronize());

                // printf("d_patch_local_seed_label_count: ");
                // check_d_arr<uint16_t><<<1,
                // 1>>>(d_patch_local_seed_label_count,
                //                                 rx.get_num_patches());
                // CUDA_ERROR(cudaDeviceSynchronize());

                // printf("d_patch_local_frontiers: ");
                // check_d_arr<<<1, 1>>>(d_patch_local_frontiers,
                //                       rx.get_num_patches());
                // CUDA_ERROR(cudaDeviceSynchronize());

                // printf("d_frontier_size: %d\n", d_frontier_size[0]);

                if (d_frontier_size[0] == curr_num_seeds) {
                    break;
                }

                tmp_break_counter++;

                if (tmp_break_counter > 20) {
                    printf("Error: recenter failed\n");
                    exit(1);
                }
            }

            // re-set the frontier index
            d_frontier_head[0]     = 0;
            d_frontier_size[0]     = curr_num_seeds;
            d_new_frontier_head[0] = d_frontier_head[0] + d_frontier_size[0];
            d_new_frontier_size[0] = 0;
        }

        // printf("---------- propogate ----------\n");

        // tmp check
        if (d_patch_local_frontiers[curr_num_seeds - 1] == INVALID32) {
            printf("last frontier: %u, seeds: %u \n", d_patch_local_frontiers[curr_num_seeds - 1], curr_num_seeds);
            printf("d_patch_local_frontiers: ");
            check_d_arr<<<1, 1>>>(d_patch_local_frontiers,
                                  rx.get_num_patches());
            CUDA_ERROR(cudaDeviceSynchronize());
            printf("Error: d_patch_local_frontiers is not initialized\n");
            exit(1);
        }

        tmp_break_counter = 0;
        // propogation until all the patches are assigned to a partition
        while (true) {
            // bipartition_seed_propogation<blockThreads>
            //     <<<blocks_p, threads_p>>>(rx.get_context(),
            //                               d_patch_partition_label,
            //                               d_patch_local_partition_role,
            //                               d_patch_local_partition_label,
            //                               d_patch_local_seed_label_count,
            //                               d_patch_local_frontiers,
            //                               d_frontier_head,
            //                               d_frontier_size,
            //                               d_new_frontier_head,
            //                               d_new_frontier_size);

            bipartition_seed_propogation_refined<blockThreads>
                <<<blocks_p, threads_p>>>(rx.get_context(),
                                          d_patch_partition_label,
                                          d_patch_local_partition_role,
                                          d_patch_local_partition_label,
                                          d_patch_local_seed_label_count,
                                          d_patch_local_frontiers,
                                          d_frontier_head,
                                          d_frontier_size,
                                          d_new_frontier_head,
                                          d_new_frontier_size);

            CUDA_ERROR(cudaDeviceSynchronize());

            // update the frontiers
            d_frontier_head[0]     = d_new_frontier_head[0];
            d_frontier_size[0]     = d_new_frontier_size[0];
            d_new_frontier_head[0] = d_frontier_head[0] + d_frontier_size[0];
            d_new_frontier_size[0] = 0;

            CUDA_ERROR(cudaDeviceSynchronize());

            // prints
            // printf("--- propogate iter ---\n");
            // printf("d_patch_local_frontiers: ");
            // check_d_arr<<<1, 1>>>(d_patch_local_frontiers,
            //                       rx.get_num_patches());
            // CUDA_ERROR(cudaDeviceSynchronize());

            // printf("d_patch_local_partition_role: ");
            // check_d_arr<<<1, 1>>>(d_patch_local_partition_role,
            //                       rx.get_num_patches());
            // CUDA_ERROR(cudaDeviceSynchronize());

            // printf("d_patch_local_partition_label: ");
            // check_d_arr<<<1, 1>>>(d_patch_local_partition_label,
            //                       rx.get_num_patches());
            // CUDA_ERROR(cudaDeviceSynchronize());

            // printf("d_patch_local_seed_label_count: ");
            // check_d_arr<uint16_t><<<1, 1>>>(d_patch_local_seed_label_count,
            //                                 rx.get_num_patches());
            // CUDA_ERROR(cudaDeviceSynchronize());

            if (d_frontier_size[0] == rx.get_num_patches()) {
                break;
            }

            tmp_break_counter++;
            if (tmp_break_counter > 50) {
                printf("Error: propogation failed\n");

                bipartition_check_propogation<blockThreads>
                    <<<1, 1>>>(rx.get_context(),
                               d_patch_partition_label,
                               d_patch_local_seed_label_count,
                               d_patch_local_partition_role,
                               d_patch_local_partition_label,
                               d_patch_local_frontiers,
                               d_frontier_size);
                CUDA_ERROR(cudaDeviceSynchronize());

                bipartition_check_isolation<blockThreads>
                    <<<blocks_p, threads_p>>>(rx.get_context(),
                                              d_patch_partition_label,
                                              d_patch_local_seed_label_count,
                                              d_patch_local_partition_role,
                                              d_patch_local_partition_label);

                exit(0);
                break;
            }
        }

        lloyd_iter++;
    }

    printf("d_patch_local_seed_label_count: ");
    check_d_arr<uint16_t>
        <<<1, 1>>>(d_patch_local_seed_label_count, rx.get_num_patches());
    CUDA_ERROR(cudaDeviceSynchronize());

    // tmp array for balancing
    // TODO: move the allocation outside of the function
    uint32_t* d_tmp_local_seed_label_count_difference;
    CUDA_ERROR(cudaMallocManaged(&d_tmp_local_seed_label_count_difference,
                                 curr_num_seeds * sizeof(uint32_t)));
    for (uint32_t i = 0; i < curr_num_partitions; i++) {
        uint32_t patch_seed0_label = i << 1;
        uint32_t patch_seed1_label = (i << 1) + 1;

        if (d_patch_local_seed_label_count[patch_seed0_label] >
            d_patch_local_seed_label_count[patch_seed1_label]) {
            d_tmp_local_seed_label_count_difference[patch_seed0_label] =
                (d_patch_local_seed_label_count[patch_seed0_label] -
                 d_patch_local_seed_label_count[patch_seed1_label]) /
                2;
            d_tmp_local_seed_label_count_difference[patch_seed1_label] = 0;
        } else {
            d_tmp_local_seed_label_count_difference[patch_seed1_label] =
                (d_patch_local_seed_label_count[patch_seed1_label] -
                 d_patch_local_seed_label_count[patch_seed0_label]) /
                2;
            d_tmp_local_seed_label_count_difference[patch_seed0_label] = 0;
        }
    }
    CUDA_ERROR(cudaDeviceSynchronize());

    printf("d_tmp_local_seed_label_count_difference: ");
    check_d_arr<uint32_t>
        <<<1, 1>>>(d_tmp_local_seed_label_count_difference, curr_num_seeds);
    CUDA_ERROR(cudaDeviceSynchronize());

    // refinement
    // bipartition_mark_boundary<blockThreads>
    //     <<<blocks_p, threads_p>>>(rx.get_context(),
    //                               d_patch_partition_label,
    //                               d_patch_local_seed_label_count,
    //                               d_patch_local_partition_role,
    //                               d_patch_local_partition_label);
    // CUDA_ERROR(cudaDeviceSynchronize());

    // bipartition_refine_partition<blockThreads>
    //     <<<blocks_p, threads_p>>>(rx.get_context(),
    //                               d_tmp_local_seed_label_count_difference,
    //                               d_patch_partition_label,
    //                               d_patch_local_seed_label_count,
    //                               d_patch_local_partition_role,
    //                               d_patch_local_partition_label);

    // // check the isolation
    // bipartition_check_isolation<blockThreads>
    //     <<<blocks_p, threads_p>>>(rx.get_context(),
    //                               d_patch_partition_label,
    //                               d_patch_local_seed_label_count,
    //                               d_patch_local_partition_role,
    //                               d_patch_local_partition_label);

    

    printf("d_tmp_local_seed_label_count_difference: ");
    check_d_arr<uint32_t>
        <<<1, 1>>>(d_tmp_local_seed_label_count_difference, curr_num_seeds);
    CUDA_ERROR(cudaDeviceSynchronize());

    printf("---------- check label ----------\n");

    printf("d_patch_partition_label: ");
    check_d_arr<<<1, 1>>>(d_patch_partition_label, rx.get_num_patches());
    CUDA_ERROR(cudaDeviceSynchronize());

    printf("d_patch_local_partition_label: ");
    check_d_arr<<<1, 1>>>(d_patch_local_partition_label, rx.get_num_patches());
    CUDA_ERROR(cudaDeviceSynchronize());

    printf("d_patch_local_seed_label_count: ");
    check_d_arr<uint16_t>
        <<<1, 1>>>(d_patch_local_seed_label_count, rx.get_num_patches());
    CUDA_ERROR(cudaDeviceSynchronize());

    printf("---------- lloyd finish ----------\n");

    // refinement on the boundary
}

template <uint32_t blockThreads>
__global__ static void nd_mark_vertex_separator(
    const rxmesh::Context     context,
    VertexAttribute<uint16_t> v_attr_spv_label,
    uint32_t*                 d_patch_partition_label,
    uint16_t*                 d_patch_local_partition_label,
    uint32_t                  curr_nd_level)
{
    // VV qury to extract the vertex separators
    auto vv_extract_separartors = [&](VertexHandle v_id, VertexIterator& vv) {
        uint32_t v_patch_id = v_id.patch_id();

        assert(v_patch_id < context.m_num_patches[0]);

        bool is_separator = false;
        for (uint16_t i = 0; i < vv.size(); ++i) {
            VertexHandle adj_v_id     = vv[i];
            uint32_t     adj_patch_id = adj_v_id.patch_id();

            assert(adj_patch_id < context.m_num_patches[0]);

            if (d_patch_partition_label[adj_patch_id] ==
                    d_patch_partition_label[v_patch_id] &&
                d_patch_local_partition_label[adj_patch_id] !=
                    d_patch_local_partition_label[v_patch_id] &&
                adj_patch_id > v_patch_id) {
                is_separator = true;
                break;
            }
        }

        if (is_separator) {
            // printf("is separator !!!");
            // level 0 partition 0: 0
            // level 1 partition 0: 1
            // level 1 partition 1: 2
            // level 2 partition 0: 3
            // ...
            uint32_t spv_index = ((1 << curr_nd_level) - 1) +
                                 d_patch_partition_label[v_patch_id];
            v_attr_spv_label(v_id, 0) = spv_index;
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, vv_extract_separartors);
}

// TODO: CPU implementition, move to GPU later
template <uint32_t blockThreads>
void update_partition_label(uint32_t* d_patch_partition_label,
                            uint16_t* d_patch_local_partition_label,
                            uint32_t  num_patches)
{
    for (uint32_t i = 0; i < num_patches; i++) {
        uint32_t patch_seed_label = (d_patch_partition_label[i] << 1) +
                                    d_patch_local_partition_label[i];
        d_patch_partition_label[i] = patch_seed_label;
    }
    CUDA_ERROR(cudaDeviceSynchronize());
}

template <uint32_t blockThreads>
__global__ static void nd_count_vertex_num(
    const rxmesh::Context     context,
    VertexAttribute<uint16_t> v_attr_spv_label,
    VertexAttribute<uint16_t> v_attr_ordering,
    uint32_t*                 d_patch_num_v,
    uint32_t*                 d_spv_num_v_heap)
{
    // VV qury to extract the vertex separators
    auto ve_count_num_v = [&](VertexHandle v_id, VertexIterator& vv) {
        uint32_t v_patch_id = v_id.patch_id();

        uint32_t local_ordering = INVALID32;
        if (v_attr_spv_label(v_id, 0) == INVALID16) {
            local_ordering = ::atomicAdd(&d_patch_num_v[v_patch_id], 1);
        } else {
            uint32_t spv_index = v_attr_spv_label(v_id, 0);
            local_ordering     = ::atomicAdd(&d_spv_num_v_heap[spv_index], 1);
        }

        assert(local_ordering != INVALID32);
        v_attr_ordering(v_id, 0) = local_ordering;
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VE>(block, shrd_alloc, ve_count_num_v);
}

template <uint32_t blockThreads>
__global__ void tmp_copy_spv_index(uint32_t* d_spv_prefix_sum_mapping_arr,
                                   uint32_t* d_patch_prefix_sum_mapping_arr,
                                   uint32_t  num_patches,
                                   uint32_t  num_patch_separator)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (uint32_t i = 0; i < num_patch_separator; i++) {
            d_spv_prefix_sum_mapping_arr[i] =
                d_patch_prefix_sum_mapping_arr[num_patches + i];
        }
    }
}

template <uint32_t blockThreads>
void generate_total_num_v_prefix_sum(uint32_t* d_patch_partition_label,
                                     uint32_t* d_patch_num_v,
                                     uint32_t* d_spv_num_v_heap,
                                     uint32_t* d_total_num_v_prefix_sum,
                                     uint32_t* d_patch_prefix_sum_mapping_arr,
                                     uint32_t* d_spv_prefix_sum_mapping_arr,
                                     uint32_t  nd_level,
                                     uint32_t  num_patches,
                                     uint32_t  num_patch_separator,
                                     uint32_t  total_prefix_sum_size)
{
    // load the total num v spv
    CUDA_ERROR(cudaMemcpy(d_total_num_v_prefix_sum,
                          d_patch_num_v,
                          num_patches * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemcpy(d_total_num_v_prefix_sum + num_patches,
                          d_spv_num_v_heap,
                          num_patch_separator * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaDeviceSynchronize());

    // load the labels and multiply them by 10
    // assume nd level < 10
    uint32_t* d_tmp_total_label;
    CUDA_ERROR(cudaMallocManaged(&d_tmp_total_label,
                                 total_prefix_sum_size * sizeof(uint32_t)));
    cudaMemset(
        d_tmp_total_label, INVALID32, total_prefix_sum_size * sizeof(uint32_t));

    // load patch labels
    for (uint32_t i = 0; i < num_patches; i++) {
        d_tmp_total_label[i] = d_patch_partition_label[i] * 10;
    }

    // load spv labels
    uint32_t num_partitions = 1 << nd_level;
    for (uint32_t i = 0; i < nd_level; i++) {
        uint32_t start_idx = (1 << i) - 1;
        uint32_t offset    = (1 << i);  // current num of partitions
        uint32_t interval  = num_partitions / offset;

        for (uint32_t j = 0; j < offset; j++) {
            uint32_t spv_label =
                ((interval * (j + 1)) - 1) * 10 + (nd_level - i);
            d_tmp_total_label[num_patches + start_idx + j] = spv_label;
        }
    }
    CUDA_ERROR(cudaDeviceSynchronize());

    // sort by d_tmp_total_label
    uint32_t* d_tmp_indices;
    CUDA_ERROR(cudaMallocManaged(&d_tmp_indices,
                                 total_prefix_sum_size * sizeof(uint32_t)));
    thrust::sequence(thrust::device,
                     d_tmp_indices,
                     d_tmp_indices + total_prefix_sum_size - 1);

    printf("d_tmp_total_label: ");
    check_d_arr<<<1, 1>>>(d_tmp_total_label, total_prefix_sum_size);
    CUDA_ERROR(cudaDeviceSynchronize());

    // the last index is reserved for exclusive sum which means nothing for the
    // sorting
    thrust::sort_by_key(thrust::device,
                        d_tmp_total_label,
                        d_tmp_total_label + total_prefix_sum_size - 1,
                        thrust::make_zip_iterator(thrust::make_tuple(
                            d_tmp_indices, d_total_num_v_prefix_sum)));

    printf("d_tmp_indices: ");
    check_d_arr<<<1, 1>>>(d_tmp_indices, total_prefix_sum_size);
    CUDA_ERROR(cudaDeviceSynchronize());

    // get the mapping array
    thrust::sequence(
        thrust::device,
        d_patch_prefix_sum_mapping_arr,
        d_patch_prefix_sum_mapping_arr + total_prefix_sum_size - 1);

    printf("INIT - d_patch_prefix_sum_mapping_arr: ");
    check_d_arr<<<1, 1>>>(d_patch_prefix_sum_mapping_arr,
                          total_prefix_sum_size);
    CUDA_ERROR(cudaDeviceSynchronize());

    uint32_t* d_tmp_indices_copy;
    CUDA_ERROR(cudaMalloc(&d_tmp_indices_copy,
                          total_prefix_sum_size * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_tmp_indices_copy,
                          d_tmp_indices,
                          total_prefix_sum_size * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaDeviceSynchronize());

    // TODO: result messed up when finer partitions
    thrust::sort_by_key(
        thrust::device,
        d_tmp_indices_copy,
        d_tmp_indices_copy + total_prefix_sum_size - 1,
        thrust::make_zip_iterator(d_patch_prefix_sum_mapping_arr));
    CUDA_ERROR(cudaDeviceSynchronize());

    printf("d_patch_prefix_sum_mapping_arr: ");
    check_d_arr<<<1, 1>>>(d_patch_prefix_sum_mapping_arr,
                          total_prefix_sum_size);
    CUDA_ERROR(cudaDeviceSynchronize());

    // extract the spv label
    printf("num_patches: %u, num_patch_separator: %u, check: %u\n",
           num_patches,
           num_patch_separator,
           d_patch_prefix_sum_mapping_arr[num_patches]);

    // TODO: weird index mis-match when the nd_levels is large
    // cudaMemcpy(d_spv_prefix_sum_mapping_arr,
    //            &d_patch_prefix_sum_mapping_arr[num_patches],
    //            num_patch_separator * sizeof(uint32_t),
    //            cudaMemcpyDeviceToDevice);
    // tmp_copy_spv_index<blockThreads>
    //     <<<1, 1>>>(d_spv_prefix_sum_mapping_arr,
    //                d_patch_prefix_sum_mapping_arr,
    //                num_patches,
    //                num_patch_separator);

    printf("d_spv_prefix_sum_mapping_arr: ");
    check_d_arr<<<1, 1>>>(d_spv_prefix_sum_mapping_arr, num_patch_separator);
    CUDA_ERROR(cudaDeviceSynchronize());

    // generate the prefix sum
    thrust::exclusive_scan(d_total_num_v_prefix_sum,
                           d_total_num_v_prefix_sum + total_prefix_sum_size,
                           d_total_num_v_prefix_sum);
}

template <uint32_t blockThreads>
__global__ static void nd_generate_numbering(
    const rxmesh::Context     context,
    VertexAttribute<uint16_t> v_attr_spv_label,
    VertexAttribute<uint16_t> v_attr_ordering,
    uint32_t*                 d_total_num_v_prefix_sum,
    uint32_t*                 d_patch_prefix_sum_mapping_arr,
    uint32_t*                 d_spv_prefix_sum_mapping_arr,
    uint32_t                  num_patches)
{
    auto ve_generate_numbering = [&](VertexHandle v_id, EdgeIterator& ve) {
        uint32_t v_patch_id = v_id.patch_id();

        if (v_attr_spv_label(v_id, 0) == INVALID16) {
            // not a separator
            v_attr_ordering(v_id, 0) += d_total_num_v_prefix_sum
                [d_patch_prefix_sum_mapping_arr[v_patch_id]];
        } else {
            // is a separator
            v_attr_ordering(v_id, 0) +=
                d_total_num_v_prefix_sum[d_patch_prefix_sum_mapping_arr[(
                    num_patches + v_attr_spv_label(v_id, 0))]];
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VE>(block, shrd_alloc, ve_generate_numbering);
}

void nd_reorder(RXMeshStatic& rx, uint32_t* ordering_arr, uint16_t nd_level)
{
    constexpr uint32_t blockThreads = 256;

    uint32_t blocks  = rx.get_num_patches();
    uint32_t threads = blockThreads;

    // store the final ordering result in the vertex color attribute
    auto v_attr_ordering =
        rx.add_vertex_attribute<uint16_t>("v_attr_ordering", 1);
    v_attr_ordering->reset(INVALID16, rxmesh::DEVICE);

    auto v_attr_spv_label =
        rx.add_vertex_attribute<uint16_t>("v_attr_spv_label", 1);
    v_attr_spv_label->reset(INVALID16, rxmesh::DEVICE);

    // variables for lloyd calculation
    uint32_t* d_patch_local_frontiers;
    CUDA_ERROR(cudaMallocManaged(&d_patch_local_frontiers,
                                 rx.get_num_patches() * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_patch_local_frontiers,
                          INVALID32,
                          rx.get_num_patches() * sizeof(uint32_t)));

    uint16_t* d_patch_local_partition_role;
    CUDA_ERROR(cudaMallocManaged(&d_patch_local_partition_role,
                                 rx.get_num_patches() * sizeof(uint16_t)));
    CUDA_ERROR(cudaMemset(d_patch_local_partition_role,
                          INVALID16,
                          rx.get_num_patches() * sizeof(uint16_t)));

    uint16_t* d_patch_local_partition_label;
    CUDA_ERROR(cudaMallocManaged(&d_patch_local_partition_label,
                                 rx.get_num_patches() * sizeof(uint16_t)));
    CUDA_ERROR(cudaMemset(d_patch_local_partition_label,
                          INVALID16,
                          rx.get_num_patches() * sizeof(uint16_t)));

    uint32_t* d_frontier_head;
    CUDA_ERROR(cudaMallocManaged(&d_frontier_head, sizeof(uint32_t)));
    uint32_t* d_frontier_size;
    CUDA_ERROR(cudaMallocManaged(&d_frontier_size, sizeof(uint32_t)));
    uint32_t* d_new_frontier_head;
    CUDA_ERROR(cudaMallocManaged(&d_new_frontier_head, sizeof(uint32_t)));
    uint32_t* d_new_frontier_size;
    CUDA_ERROR(cudaMallocManaged(&d_new_frontier_size, sizeof(uint32_t)));

    uint16_t* d_patch_local_seed_label_count;
    CUDA_ERROR(cudaMallocManaged(&d_patch_local_seed_label_count,
                                 rx.get_num_patches() * sizeof(uint16_t)));
    CUDA_ERROR(cudaMemset(d_patch_local_seed_label_count,
                          INVALID16,
                          rx.get_num_patches() * sizeof(uint16_t)));


    // variables for prefixsum & ordering calculation
    // TODO tmp variable for testing
    nd_level                     = 6;
    uint32_t num_patch_separator = (1 << nd_level) - 1;
    uint32_t total_prefix_sum_size =
        rx.get_num_patches() + 1 + num_patch_separator;

    uint32_t* d_patch_partition_label;  // label of v_ordering_prefix_sum for
                                        // each patch
    cudaMallocManaged(&d_patch_partition_label,
                      rx.get_num_patches() * sizeof(uint32_t));
    cudaMemset(
        d_patch_partition_label, 0, rx.get_num_patches() * sizeof(uint32_t));

    uint32_t* d_patch_num_v;
    cudaMallocManaged(&d_patch_num_v, rx.get_num_patches() * sizeof(uint32_t));
    cudaMemset(d_patch_num_v, 0, rx.get_num_patches() * sizeof(uint32_t));

    uint32_t* d_spv_num_v_heap;  // manage the separators in a heap manner
    cudaMallocManaged(&d_spv_num_v_heap,
                      num_patch_separator * sizeof(uint32_t));
    cudaMemset(d_spv_num_v_heap, 0, num_patch_separator * sizeof(uint32_t));

    uint32_t* d_total_num_v_prefix_sum;
    cudaMallocManaged(&d_total_num_v_prefix_sum,
                      total_prefix_sum_size * sizeof(uint32_t));
    cudaMemset(
        d_total_num_v_prefix_sum, 0, total_prefix_sum_size * sizeof(uint32_t));

    uint32_t* d_patch_prefix_sum_mapping_arr;
    cudaMallocManaged(&d_patch_prefix_sum_mapping_arr,
                      rx.get_num_patches() * sizeof(uint32_t));
    cudaMemset(d_patch_prefix_sum_mapping_arr,
               INVALID32,
               total_prefix_sum_size * sizeof(uint32_t));

    uint32_t* d_spv_prefix_sum_mapping_arr;
    cudaMallocManaged(&d_spv_prefix_sum_mapping_arr,
                      num_patch_separator * sizeof(uint32_t));
    cudaMemset(d_spv_prefix_sum_mapping_arr,
               INVALID32,
               num_patch_separator * sizeof(uint32_t));


    // prepare launch box for GPU kernels
    LaunchBox<blockThreads> launch_box_nd_init_edge_weight;
    rx.prepare_launch_box({rxmesh::Op::EV},
                          launch_box_nd_init_edge_weight,
                          (void*)nd_init_edge_weight<blockThreads>);
    LaunchBox<blockThreads> launch_box_mark_vertex_separator;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_mark_vertex_separator,
                          (void*)nd_mark_vertex_separator<blockThreads>);
    LaunchBox<blockThreads> launch_box_count_vertex_num;
    rx.prepare_launch_box({rxmesh::Op::VE},
                          launch_box_count_vertex_num,
                          (void*)nd_count_vertex_num<blockThreads>);
    LaunchBox<blockThreads> launch_box_generate_numbering;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_generate_numbering,
                          (void*)nd_generate_numbering<blockThreads>);

    // ---------- main body starts here ----------
    // init edge weight
    nd_init_edge_weight<blockThreads>
        <<<launch_box_nd_init_edge_weight.blocks,
           launch_box_nd_init_edge_weight.num_threads,
           launch_box_nd_init_edge_weight.smem_bytes_dyn>>>(rx.get_context());
    CUDA_ERROR(cudaDeviceSynchronize());

    // big loop for each level
    for (uint32_t i = 0; i < nd_level; i++) {
        // run partition lloyd
        run_partition_lloyd<blockThreads>(rx,
                                          i,
                                          d_patch_partition_label,
                                          d_patch_local_seed_label_count,
                                          d_patch_local_partition_role,
                                          d_patch_local_partition_label,
                                          d_patch_local_frontiers,
                                          d_frontier_head,
                                          d_frontier_size,
                                          d_new_frontier_head,
                                          d_new_frontier_size);

        // mark vertex separator
        nd_mark_vertex_separator<blockThreads>
            <<<launch_box_mark_vertex_separator.blocks,
               launch_box_mark_vertex_separator.num_threads,
               launch_box_mark_vertex_separator.smem_bytes_dyn>>>(
                rx.get_context(),
                *v_attr_spv_label,
                d_patch_partition_label,
                d_patch_local_partition_label,
                i);
        CUDA_ERROR(cudaDeviceSynchronize());

        // update partition label
        update_partition_label<blockThreads>(d_patch_partition_label,
                                             d_patch_local_partition_label,
                                             rx.get_num_patches());
    }

    printf("--------- starting ordering generation ---------\n");

    // count the number of vertices in each patch and vertex separator
    nd_count_vertex_num<blockThreads>
        <<<launch_box_count_vertex_num.blocks,
           launch_box_count_vertex_num.num_threads,
           launch_box_count_vertex_num.smem_bytes_dyn>>>(rx.get_context(),
                                                         *v_attr_spv_label,
                                                         *v_attr_ordering,
                                                         d_patch_num_v,
                                                         d_spv_num_v_heap);
    CUDA_ERROR(cudaDeviceSynchronize());

    printf("d_patch_num_v: ");
    check_d_arr<<<1, 1>>>(d_patch_num_v, rx.get_num_patches());
    CUDA_ERROR(cudaDeviceSynchronize());

    printf("d_spv_num_v_heap: ");
    check_d_arr<<<1, 1>>>(d_spv_num_v_heap, num_patch_separator);
    CUDA_ERROR(cudaDeviceSynchronize());

    // generate the total num v prefix sum
    generate_total_num_v_prefix_sum<blockThreads>(
        d_patch_partition_label,
        d_patch_num_v,
        d_spv_num_v_heap,
        d_total_num_v_prefix_sum,
        d_patch_prefix_sum_mapping_arr,
        d_spv_prefix_sum_mapping_arr,
        nd_level,
        rx.get_num_patches(),
        num_patch_separator,
        total_prefix_sum_size);

    printf("d_total_num_v_prefix_sum: ");
    check_d_arr<<<1, 1>>>(d_total_num_v_prefix_sum, total_prefix_sum_size);
    CUDA_ERROR(cudaDeviceSynchronize());

    // generate numbering
    nd_generate_numbering<blockThreads>
        <<<launch_box_generate_numbering.blocks,
           launch_box_generate_numbering.num_threads,
           launch_box_generate_numbering.smem_bytes_dyn>>>(
            rx.get_context(),
            *v_attr_spv_label,
            *v_attr_ordering,
            d_total_num_v_prefix_sum,
            d_patch_prefix_sum_mapping_arr,
            d_spv_prefix_sum_mapping_arr,
            rx.get_num_patches());

    printf("---------- finish ordering generation ----------\n");

    // generate the result for testing
    v_attr_ordering->move(rxmesh::DEVICE, rxmesh::HOST);
    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        uint32_t v_global_id = rx.map_to_global(vh);
        uint32_t v_linea_id  = rx.linear_id(vh);
        uint32_t v_order_idx = (*v_attr_ordering)(vh, 0);

        ordering_arr[v_order_idx] = v_global_id;
    });
}

// automatically num levels - no need to specify hyper parameters

}  // namespace rxmesh