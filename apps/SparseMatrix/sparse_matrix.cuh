#include <iostream>
#include <vector>
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/types.h"
#include "rxmesh/patch_ptr.h"
#include "rxmesh/kernels/query_dispatcher.cuh"

namespace rxmesh {

// this is the function for the CSR calculation
template <typename T, uint32_t blockThreads>
__global__ static void sparse_mat_scan(const rxmesh::Context context,
                                       uint32_t*&            patch_ptr_v,
                                       uint32_t*&            row_ptr)
{
    using namespace rxmesh;


    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto ids = v_id.unpack();
        uint32_t patch_id = ids.first;
        uint16_t local_id = ids.second;
        row_ptr[patch_ptr_v[patch_id] + local_id] = iter.size();
    };

    // With uniform Laplacian, we just need the valence, thus we
    // call query_block_dispatcher and set oriented to false
    query_block_dispatcher<Op::VV, blockThreads>(context, init_lambda);
}

// Follow the idea of "All calculations and storage is done on the GPU." This is
// for initial mem allocation. This is currently VV implementation, will bge
// extended
template <typename T>
void sparse_mat_init(RXMeshStatic& rx,
                     uint32_t*&    patch_ptr_v,
                     uint32_t*&    row_ptr,
                     uint32_t*&    col_ptr,
                     T*&           val)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    uint32_t num_patches  = rx.get_num_patches();
    uint32_t num_vertives = rx.get_num_vertices();
    uint32_t num_edges    = rx.get_num_edges();

    CUDA_ERROR(
        cudaMalloc((void**)&row_ptr, (num_vertives + 1) * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&col_ptr, (num_edges * 2) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&val, (num_edges * 2) * sizeof(uint32_t)));

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_scan<T, blockThreads>);

    sparse_mat_scan<T, blockThreads><<<launch_box.blocks,
                                       launch_box.num_threads,
                                       launch_box.smem_bytes_dyn>>>(
        rx.get_context(), patch_ptr_v, row_ptr);

    void*  d_cub_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                  temp_storage_bytes,
                                  row_ptr,
                                  row_ptr,
                                  num_patches);
    CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage, temp_storage_bytes));

    cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                  temp_storage_bytes,
                                  row_ptr,
                                  row_ptr,
                                  num_patches);
    CUDA_ERROR(cudaMemset(d_cub_temp_storage, 0, temp_storage_bytes));

    CUDA_ERROR(cudaFree(d_cub_temp_storage));
}


// TODO: add compatibility for EE, FF, VE......
template <typename T>
struct SparseMatInfo
{
    SparseMatInfo(RXMeshStatic& rx)
        : patch_ptr_v(nullptr),
          patch_ptr_e(nullptr),
          patch_ptr_f(nullptr),
          row_ptr(nullptr),
          col_ptr(nullptr),
          val(nullptr)
    {
        detail::init(
            rx, patch_ptr_v, patch_ptr_e, patch_ptr_f);  // patch pointer init
        sparse_mat_init(rx, patch_ptr_v, row_ptr, col_ptr, val);
    }

    void free()
    {
        CUDA_ERROR(cudaFree(patch_ptr_v));
        CUDA_ERROR(cudaFree(patch_ptr_e));
        CUDA_ERROR(cudaFree(patch_ptr_f));
        CUDA_ERROR(cudaFree(row_ptr));
        CUDA_ERROR(cudaFree(col_ptr));
        CUDA_ERROR(cudaFree(val));
    }

    uint32_t *patch_ptr_v, *patch_ptr_e, *patch_ptr_f;
    uint32_t* row_ptr;
    uint32_t* col_ptr;
    T*        val;
};

}  // namespace rxmesh