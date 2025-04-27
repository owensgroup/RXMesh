#pragma once


#include "rxmesh/matrix/sparse_matrix.h"

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
    using Type = T;

    static constexpr int K_ = K;

    using ScalarT = Scalar<T, K, true>;

    using IndexT = typename SparseMatrix<T>::IndexT;


    HessianSparseMatrix() : SparseMatrix<T>()
    {
    }

    HessianSparseMatrix(const RXMeshStatic& rx, Op op = Op::VV)
        : SparseMatrix<T>(rx, op, K)
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
        const IndexT r_id =
            this->get_row_id(row_v) * this->m_replicate + local_i;
        const IndexT c_id =
            this->get_row_id(col_v) * this->m_replicate + local_j;

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
        const IndexT r_id =
            this->get_row_id(row_v) * this->m_replicate + local_i;
        const IndexT c_id =
            this->get_row_id(col_v) * this->m_replicate + local_j;

        return SparseMatrix<T>::operator()(r_id, c_id);
    }

    /**
     * @brief give (vertex) handles to the entires in the matrix and the
     * local Hessian indices, return the raw indices into the sparse matrix
     */
    __device__ __host__ const std::pair<int, int> get_indices(
        const VertexHandle& row_v,
        const VertexHandle& col_v,
        const IndexT        local_i,
        const IndexT        local_j) const
    {
        const IndexT r_id =
            this->get_row_id(row_v) * this->m_replicate + local_i;
        const IndexT c_id =
            this->get_row_id(col_v) * this->m_replicate + local_j;

        return {r_id, c_id};
    }

    __device__ __host__ const T& operator()(const IndexT& row_id,
                                            const IndexT& col_id) const
    {
        return SparseMatrix<T>::operator()(row_id, col_id);
    }

    __device__ __host__ T& operator()(const IndexT& row_id,
                                      const IndexT& col_id)
    {
        return SparseMatrix<T>::operator()(row_id, col_id);
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
