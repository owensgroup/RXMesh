#pragma once

#include "rxmesh/diff/hessian_sparse_matrix.h"

#include "rxmesh/diff/scalar.h"

namespace rxmesh {

/**
 * @brief Construct the 'dynamic' sparse Hessian of type T with K variables per
 * vertex. Dynamic here means the sparsity of the Hessian is going to change
 during runtime. The matrix size is (V * K) X (V * K) where V is the number of
 * vertices in the mesh.
 *
 */
template <typename T, int K>
struct DynamicHessianSparseMatrix : public HessianSparseMatrix<T, k>
{
    using Type = T;

    static constexpr int K_ = K;

    using ScalarT = Scalar<T, K, true>;

    using IndexT = typename SparseMatrix<T>::IndexT;


    DynamicHessianSparseMatrix() : HessianSparseMatrix<T, k>()
    {
    }

    DynamicHessianSparseMatrix(const RXMeshStatic& rx, Op op = Op::VV)
        : HessianSparseMatrix<T>(rx, op)
    {
    }


    /**
     * @brief insert more entries to the hessian matrix. The input here is the
     * list of vertices that will be interacting. Thus, we need to extend them
     * by the number of their replicate, i.e., k
     */
    __host__ void insert(uint32_t size, IndexT* d_rows, IndexT* d_cols)
    {
        // Here, we assume the number of rows and cols is the same and only
        // the sparsity is changing

        constexpr uint32_t blockThreads = 256;

        // fill the new row_ptr with the data from the mesh connectivity
        rx.run_kernel<blockThreads>(
            {Op::VV},
            detail::sparse_mat_prescan<Op::VV, blockThreads>,
            m_d_row_ptr,
            k);

        // fill the new row_ptr with the data from the new entries
        detail::
            sparse_mat_prescan<<<DIVIDE_UP(size, blockThreads), blockThreads>>>(
                m_d_row_ptr, size, d_rows, d_cols, k);
    }

    // delete the functions that access the matrix using only the VertexHandle
    // since with the Hessian, we should also have the local index (the index
    // within the kxk matrix)
    __device__ __host__ const T& operator()(const VertexHandle& row_v,
                                            const VertexHandle& col_v) const =
        delete;

    __device__ __host__ T& operator()(const VertexHandle& row_v,
                                      const VertexHandle& col_v) = delete;
};

}  // namespace rxmesh
