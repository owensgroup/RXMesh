#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace rxmesh {

namespace detail {

__inline__ __device__ bool is_v_on_separator(const VertexHandle    v_id,
                                             const VertexIterator& iter)
{
    for (uint16_t i = 0; i < iter.size(); ++i) {

        if (iter[i].patch_id() > v_id.patch_id()) {
            return true;
        }
    }
    return false;
}

template <uint32_t blockThreads>
__global__ static void extract_separartor(const Context context,
                                          int*          d_permute,
                                          uint32_t*     d_v_ordering_prefix_sum)
{

    // VV query to extract the vertex separators
    auto extract = [&](VertexHandle v_id, VertexIterator& iter) {
        uint32_t v_local_order = INVALID32;
        if (is_v_on_separator(v_id, iter)) {
            // if the vertex is on the separator, then we could it towards the
            // the end of this prefix_sum array
            v_local_order =
                ::atomicAdd(&d_v_ordering_prefix_sum[context.get_num_patches()],
                            uint32_t(1));
        } else {
            v_local_order = ::atomicAdd(
                &d_v_ordering_prefix_sum[v_id.patch_id()], uint32_t(1));
        }

        assert(v_local_order != INVALID32);

        d_permute[context.linear_id(v_id)] = v_local_order;
        // v_ordering(v_id) = v_local_order;
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, extract);
}

template <uint32_t blockThreads>
__global__ static void assign_permutation(const Context context,
                                          int*          d_permute,
                                          uint32_t*     d_v_ordering_prefix_sum)
{
    auto assign = [&](VertexHandle v_id, VertexIterator& iter) {
        if (is_v_on_separator(v_id, iter)) {
            d_permute[context.linear_id(v_id)] +=
                d_v_ordering_prefix_sum[context.get_num_patches()];
        } else {
            d_permute[context.linear_id(v_id)] +=
                d_v_ordering_prefix_sum[v_id.patch_id()];
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, assign);
}
}  // namespace detail

/**
 * @brief Modified Generalized Nested Dissection permutation where the separator
 * is order last. h_permute should be allocated with size equal to num of
 * vertices of the mesh.
 */
inline void mgnd_permute(RXMeshStatic& rx, std::vector<int>& h_permute)
{
    constexpr uint32_t blockThreads = 256;

    h_permute.resize(rx.get_num_vertices());

    // auto v_ordering = *rx.add_vertex_attribute<uint32_t>("v_ordering", 1);

    int* d_permute = nullptr;
    CUDA_ERROR(
        cudaMalloc((void**)&d_permute, rx.get_num_vertices() * sizeof(int)));

    uint32_t v_ordering_prefix_sum_size = rx.get_num_patches() + 2;

    // This array will contain the prefix sum of the number of vertices
    // in each patch along with number of vertices on the separator, i.e.,
    // since we extract certain vertices from each patch as a separator, then
    // the number of vertices in each patch has to be re-calculated (since it
    // is different than what RXMeshStatic stores). In addition, we add to the
    // end of this array, the number of vertices on the separator. Finally, we
    // take the prefix sum of this array (and then use it to assign the
    // permutation)
    uint32_t* d_v_ordering_prefix_sum(nullptr);

    CUDA_ERROR(cudaMalloc((void**)&d_v_ordering_prefix_sum,
                          v_ordering_prefix_sum_size * sizeof(uint32_t)));

    CUDA_ERROR(cudaMemset(d_v_ordering_prefix_sum,
                          0,
                          v_ordering_prefix_sum_size * sizeof(uint32_t)));

    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box(
        {Op::VV}, lb, (void*)detail::extract_separartor<blockThreads>);

    detail::extract_separartor<blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), d_permute, d_v_ordering_prefix_sum);


    thrust::exclusive_scan(
        thrust::device,
        d_v_ordering_prefix_sum,
        d_v_ordering_prefix_sum + (v_ordering_prefix_sum_size - 1),
        d_v_ordering_prefix_sum);

    rx.prepare_launch_box(
        {Op::VV}, lb, (void*)detail::assign_permutation<blockThreads>);

    detail::assign_permutation<blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), d_permute, d_v_ordering_prefix_sum);

    // v_ordering.move(rxmesh::DEVICE, rxmesh::HOST);

    // rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
    //     // uint32_t v_global_id = rx.map_to_global(vh);
    //     uint32_t v_linea_id  = rx.linear_id(vh);
    //     uint32_t v_order_idx = v_ordering(vh);
    //
    //     h_permute[v_order_idx] = v_linea_id;
    // });

    CUDA_ERROR(cudaMemcpy(h_permute.data(),
                          d_permute,
                          rx.get_num_vertices() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    GPU_FREE(d_v_ordering_prefix_sum);
    GPU_FREE(d_permute);
}

}  // namespace rxmesh