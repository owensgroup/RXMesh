#include <iostream>
#include <vector>
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/kernels/query_dispatcher.cuh"
#include "rxmesh/patch_ptr.h"
#include "rxmesh/types.h"

namespace rxmesh {

template <uint32_t blockThreads>
__global__ static void sparse_mat_test(const rxmesh::Context context,
                                       uint32_t*             patch_ptr_v,
                                       uint32_t*             vet_degree)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        // printf(" %" PRIu32 " - %" PRIu32 " - %" PRIu32 " - %" PRIu32 " \n",
        //        row_ptr[0],
        //        row_ptr[1],
        //        row_ptr[2],
        //        row_ptr[3]);
        auto     ids                                 = v_id.unpack();
        uint32_t patch_id                            = ids.first;
        uint16_t local_id                            = ids.second;
        vet_degree[patch_ptr_v[patch_id] + local_id] = iter.size() + 1;
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, init_lambda);
}

// this is the function for the CSR calculation
template <uint32_t blockThreads>
__global__ static void sparse_mat_prescan(const rxmesh::Context context,
                                          uint32_t*             patch_ptr_v,
                                          uint32_t*             row_ptr)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     ids                              = v_id.unpack();
        uint32_t patch_id                         = ids.first;
        uint16_t local_id                         = ids.second;
        row_ptr[patch_ptr_v[patch_id] + local_id] = iter.size() + 1;
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, init_lambda);
}

template <uint32_t blockThreads>
__global__ static void sparse_mat_col_fill(const rxmesh::Context context,
                                           uint32_t*             patch_ptr_v,
                                           uint32_t*             row_ptr,
                                           uint32_t*             col_idx)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     ids      = v_id.unpack();
        uint32_t patch_id = ids.first;
        uint16_t local_id = ids.second;
        col_idx[row_ptr[patch_ptr_v[patch_id] + local_id]] =
            patch_ptr_v[patch_id] + local_id;
        for (uint32_t v = 0; v < iter.size(); ++v) {
            auto     s_ids      = iter[v].unpack();
            uint32_t s_patch_id = s_ids.first;
            uint16_t s_local_id = s_ids.second;
            col_idx[row_ptr[patch_ptr_v[patch_id] + local_id] + v + 1] =
                patch_ptr_v[s_patch_id] + s_local_id;
        }
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, init_lambda);
}

// Follow the idea of "All calculations and storage is done on the GPU." This is
// for initial mem allocation. This is currently VV implementation, will bge
// extended
template <typename T>
void sparse_mat_init(RXMeshStatic& rx,
                     uint32_t*&    patch_ptr_v,
                     uint32_t*&    row_ptr,
                     uint32_t*&    col_idx,
                     uint32_t&     entry_size,
                     T*&           val)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    uint32_t num_patches  = rx.get_num_patches();
    uint32_t num_vertices = rx.get_num_vertices();
    uint32_t num_edges    = rx.get_num_edges();

    // row pointer allocation and init with prefix sum for CRS
    CUDA_ERROR(
        cudaMalloc((void**)&row_ptr, (num_vertices + 1) * sizeof(uint32_t)));

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_prescan<blockThreads>);

    sparse_mat_prescan<blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), patch_ptr_v, row_ptr);

    uint32_t last_ele = 0;
    CUDA_ERROR(cudaMemcpy(&last_ele,
                          (row_ptr + num_vertices - 1),
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // prefix sum
    void*  d_cub_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_cub_temp_storage, temp_storage_bytes, row_ptr, row_ptr, num_vertices);
    CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage, temp_storage_bytes));

    cub::DeviceScan::ExclusiveSum(
        d_cub_temp_storage, temp_storage_bytes, row_ptr, row_ptr, num_vertices);

    CUDA_ERROR(cudaFree(d_cub_temp_storage));

    // get index size
    CUDA_ERROR(cudaMemcpy(&entry_size,
                          (row_ptr + num_vertices - 1),
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    entry_size += last_ele;

    CUDA_ERROR(cudaMemcpy((row_ptr + num_vertices),
                          &entry_size,
                          sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // column index allocation and init
    CUDA_ERROR(cudaMalloc((void**)&col_idx, entry_size * sizeof(uint32_t)));
    rx.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_col_fill<blockThreads>);

    sparse_mat_col_fill<blockThreads><<<launch_box.blocks,
                                        launch_box.num_threads,
                                        launch_box.smem_bytes_dyn>>>(
        rx.get_context(), patch_ptr_v, row_ptr, col_idx);

    // value allocation but init should be done in another function.
    CUDA_ERROR(cudaMalloc((void**)&val, entry_size * sizeof(uint32_t)));
}


// TODO: add compatibility for EE, FF, VE......
template <typename T>
struct SparseMatInfo
{
    SparseMatInfo(RXMeshStatic& rx)
        : m_patch_ptr_v(nullptr),
          m_patch_ptr_e(nullptr),
          m_patch_ptr_f(nullptr),
          row_ptr(nullptr),
          col_idx(nullptr),
          entry_size(0),
          val(nullptr)
    {
        detail::init(rx,
                     m_patch_ptr_v,
                     m_patch_ptr_e,
                     m_patch_ptr_f);  // patch pointer init
        sparse_mat_init(rx, m_patch_ptr_v, row_ptr, col_idx, entry_size, val);
    }

    void set_ones()
    {
        std::vector<T> init_tmp_arr(entry_size, 1);
        CUDA_ERROR(cudaMemcpy(val,
                              init_tmp_arr.data(),
                              entry_size * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    void free()
    {
        CUDA_ERROR(cudaFree(m_patch_ptr_v));
        CUDA_ERROR(cudaFree(m_patch_ptr_e));
        CUDA_ERROR(cudaFree(m_patch_ptr_f));
        CUDA_ERROR(cudaFree(row_ptr));
        CUDA_ERROR(cudaFree(col_idx));
        CUDA_ERROR(cudaFree(val));
    }

    uint32_t *m_patch_ptr_v, *m_patch_ptr_e, *m_patch_ptr_f;
    uint32_t* row_ptr;
    uint32_t* col_idx;
    uint32_t  entry_size;
    T*        val;
};

}  // namespace rxmesh