#include <iostream>
#include <vector>
#include "rxmesh/context.h"
#include "rxmesh/types.h"

namespace rxmesh {

template <typename T, uint32_t blockThreads>
__global__ static void init_sparse_mat(const rxmesh::Context context,
                                       SparseMatInfo         spmat_container)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        for (uint32_t v = 0; v < iter.size(); ++v) {
            //TODO:
        }
    };

    // With uniform Laplacian, we just need the valence, thus we
    // call query_block_dispatcher and set oriented to false
    query_block_dispatcher<Op::VV, blockThreads>(
        context, init_lambda, !use_uniform_laplace);
}


// TODO: add compatibility for EE, FF, VE......
template <typename T>
struct SparseMatInfo
{
    PatchPtr(RXMeshStatic& rx)
        : patch_ptr_v(nullptr),
          patch_ptr_e(nullptr),
          patch_ptr_f(nullptr),
          row_pointer(nullptr),
          col_pointer(nullptr),
          val(nullptr)
    {
        detail::init(rx, patch_ptr_v, patch_ptr_e, patch_ptr_f);
    }

    uint32_t *patch_ptr_v, *patch_ptr_e, *patch_ptr_f;
    uint32_t* row_ptr;
    uint32_t* col_ptr;
    T*        val;
};

}  // namespace rxmesh