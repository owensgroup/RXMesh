#pragma once


#include "rxmesh/matrix/sparse_matrix.cuh"

#include "rxmesh/diff/scalar.h"

namespace rxmesh {

/**
 * @brief Construct the sparse Hessian of type T with K variables per vertex.
 * The matrix size is (V * K) X (V*K) where V is the number of vertices in the
 * mesh
 *
 */
template <typename T, int K>
struct HessianSparseMatrix : public SparseMatrix<T>
{
    static constexpr int K_ = K;

    using ScalarT = Scalar<T, K, true>;

    using IndexT = typename SparseMatrix<T>::IndexT;


    HessianSparseMatrix(const RXMeshStatic& rx) : SparseMatrix<T>(rx, K)
    {
    }


    /**
     * @brief access the matrix using row and col as VertexHandle's
     * along with the local indices within the Hessian
     */
    __device__ __host__ const T& operator()(const VertexHandle& row_v,
                                            const VertexHandle& col_v,
                                            const IndexT        local_i,
                                            const IndexT        local_j) const
    {
        const IndexT r_id = this->get_row_id(row_v) * this->m_replicate + local_i;
        const IndexT c_id = this->get_row_id(col_v) * this->m_replicate + local_j;

        return SparseMatrix<T>::operator()(r_id, c_id);
    }

    /**
     * @brief access the matrix using row and col as VertexHandle's
     * along with the local indices within the Hessian
     */
    __device__ __host__ T& operator()(const VertexHandle& row_v,
                                      const VertexHandle& col_v,
                                      const IndexT        local_i,
                                      const IndexT        local_j)
    {
        const IndexT r_id = this->get_row_id(row_v) * this->m_replicate + local_i;
        const IndexT c_id = this->get_row_id(col_v) * this->m_replicate + local_j;

        return SparseMatrix<T>::operator()(r_id, c_id);
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
