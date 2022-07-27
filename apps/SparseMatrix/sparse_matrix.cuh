#include <iostream>
#include <vector>
#include "rxmesh/context.h"
#include "rxmesh/types.h"

namespace rxmesh {

// Follow the idea of "All calculations and storage is done on the GPU." This is
// for initial mem allocation. This is currently VV implementation, will bge extended
template <typename T>
 void sparse_mat_init(RXMeshStatic& rx,
          uint32_t*&    row_ptr,
          uint32_t*&    col_ptr,
          T*&    val)
{

    uint32_t num_patches = rx.get_num_patches();
    uint32_t num_vertives = rx.get_num_vertices();

    uint32_t entry_size = 0;

    CUDA_ERROR(
        cudaMalloc((void**)&row_ptr, (num_vertives + 1) * sizeof(uint32_t)));

    //: TODO col and val size are num entries + 1 

}

// this is the function for the CSR calculation
template <typename T, uint32_t blockThreads>
__global__ static void sparse_mat_scan(const rxmesh::Context context,
                                       uint32_t*&    row_ptr
                                       )
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        for (uint32_t v = 0; v < iter.size(); ++v) {
            // TODO:
        }
    };

    // With uniform Laplacian, we just need the valence, thus we
    // call query_block_dispatcher and set oriented to false
    query_block_dispatcher<Op::VV, blockThreads>(
        context, init_lambda);
}


// TODO: add compatibility for EE, FF, VE......
template <typename T>
struct SparseMatInfo
{
    SparseMatInfo(RXMeshStatic& rx)
        : patch_ptr_v(nullptr),
          patch_ptr_e(nullptr),
          patch_ptr_f(nullptr),
          row_pointer(nullptr),
          col_pointer(nullptr),
          val(nullptr)
    {
        detail::init(rx, patch_ptr_v, patch_ptr_e, patch_ptr_f); // patch pointer init
        sparse_mat_init(rx, row_ptr, col_ptr, val);
    }

    uint32_t *patch_ptr_v, *patch_ptr_e, *patch_ptr_f;
    uint32_t* row_ptr;
    uint32_t* col_ptr;
    T*        val;
};

}  // namespace rxmesh