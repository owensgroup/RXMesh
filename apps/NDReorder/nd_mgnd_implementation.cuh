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
__global__ static void extract_vertices(const rxmesh::Context     context,
                                        VertexAttribute<uint16_t> v_ordering,
                                        uint32_t* v_ordering_prefix_sum,
                                        uint32_t* v_ordering_spv_idx)
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
            v_order = ::atomicAdd(
                (unsigned int*)&v_ordering_prefix_sum[v_ordering_spv_idx[0]],
                (unsigned int)1);
        } else {
            v_order =
                ::atomicAdd((unsigned int*)&v_ordering_prefix_sum[v_patch_id],
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
__global__ static void assign_numbering(const RXMeshStatic&       rx,
                                        const rxmesh::Context     context,
                                        VertexAttribute<uint16_t> v_ordering,
                                        uint32_t* v_ordering_prefix_sum,
                                        uint32_t* v_ordering_spv_idx)
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
            v_ordering(v_id, 0) += v_ordering_prefix_sum[v_ordering_spv_idx[0]];
        } else {
            v_ordering(v_id, 0) += v_ordering_prefix_sum[v_patch_id];
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, vv_assign_numbering);
}


void mgnd_reorder(RXMeshStatic& rx, uint32_t* ordering_arr)
{
    constexpr uint32_t blockThreads = 256;

    // vertex color attribute
    auto      v_ordering = rx.add_vertex_attribute<uint16_t>("v_ordering", 1);
    uint32_t  v_ordering_prefix_sum_size = rx.get_num_patches() + 2;
    uint32_t* v_ordering_prefix_sum;
    cudaMallocManaged(&v_ordering_prefix_sum,
                      v_ordering_prefix_sum_size * sizeof(uint32_t));
    cudaMemset(v_ordering_prefix_sum,
               0,
               v_ordering_prefix_sum_size * sizeof(uint32_t));

    uint32_t* v_ordering_spv_idx;
    cudaMallocManaged(&v_ordering_spv_idx, 1 * sizeof(uint32_t));
    v_ordering_spv_idx[0] = rx.get_num_patches();

    uint32_t blocks  = rx.get_num_patches();
    uint32_t threads = blockThreads;

    LaunchBox<blockThreads> launch_box_extract_vertices;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_extract_vertices,
                          (void*)extract_vertices<blockThreads>);
    LaunchBox<blockThreads> launch_box_assign_numbering;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_assign_numbering,
                          (void*)assign_numbering<blockThreads>);

    RXMESH_INFO("mgnd start");

    extract_vertices<blockThreads>
        <<<launch_box_extract_vertices.blocks,
           launch_box_extract_vertices.num_threads,
           launch_box_extract_vertices.smem_bytes_dyn>>>(rx.get_context(),
                                                         *v_ordering,
                                                         v_ordering_prefix_sum,
                                                         v_ordering_spv_idx);
    cudaDeviceSynchronize();

    printf("v_ordering_prefix_sum: ");
    for (int i = 0; i < v_ordering_prefix_sum_size; i++) {
        printf("%d ", v_ordering_prefix_sum[i]);
    }
    printf("\n");

    thrust::exclusive_scan(v_ordering_prefix_sum,
                           v_ordering_prefix_sum + v_ordering_prefix_sum_size,
                           v_ordering_prefix_sum);

    printf("v_ordering_prefix_sum: ");
    for (int i = 0; i < v_ordering_prefix_sum_size; i++) {
        printf("%d ", v_ordering_prefix_sum[i]);
    }
    printf("\n");

    assign_numbering<blockThreads>
        <<<launch_box_assign_numbering.blocks,
           launch_box_assign_numbering.num_threads,
           launch_box_assign_numbering.smem_bytes_dyn>>>(rx,
                                                         rx.get_context(),
                                                         *v_ordering,
                                                         v_ordering_prefix_sum,
                                                         v_ordering_spv_idx);
    cudaDeviceSynchronize();

    v_ordering->move(rxmesh::DEVICE, rxmesh::HOST);

    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        uint32_t v_global_id = rx.map_to_global(vh);
        uint32_t v_linea_id  = rx.linear_id(vh);
        uint32_t v_order_idx = (*v_ordering)(vh, 0);
        
        ordering_arr[v_order_idx] = v_global_id;
    });

    RXMESH_INFO("mgnd end");
}

}  // namespace rxmesh