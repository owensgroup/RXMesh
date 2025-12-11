#pragma once

#include "rxmesh/diff/scalar.h"
#include "rxmesh/matrix/sparse_matrix.h"

namespace rxmesh {
template <typename T>
struct JacobianSparseMatrix : public SparseMatrix<T>
{
    using Type = T;

    using IndexT = typename SparseMatrix<T>::IndexT;

    JacobianSparseMatrix()
        : SparseMatrix<T>(),
          m_num_terms(0),
          m_ops(nullptr),
          m_block_shapes(nullptr)
    {
    }

    JacobianSparseMatrix(const RXMeshStatic&           rx,
                         const std::vector<Op>&        ops,
                         const std::vector<BlockShape> block_shapes)
        : SparseMatrix<T>()
    {

        // we follow the same logic as in the SparseMatrix construction but the
        // difference here is we stuck multiple matrices along the

        if (ops.size() != block_shapes.size()) {
            RXMESH_ERROR(
                "JacobianSparseMatrix::JacobianSparseMatrix() mismatch between "
                "ops size {} and block_shapes size {}",
                ops.size(),
                block_shapes.size());
        }

        // safe guard against trivial cases
        if (ops.size() == 0) {
            return;
        }

        // copy ops and block_shapes
        m_num_terms = ops.size();
        m_ops       = (Op*)malloc(sizeof(Op) * m_num_terms);
        memcpy(m_ops, ops.data(), sizeof(Op) * m_num_terms);

        m_block_shapes = (BlockShape*)malloc(sizeof(BlockShape) * m_num_terms);
        memcpy(m_block_shapes,
               block_shapes.data(),
               sizeof(BlockShape) * m_num_terms);

        m_h_terms_rows_prefix =
            (IndexT*)malloc(sizeof(IndexT) * (m_num_terms + 1));
        CUDA_ERROR(cudaMalloc((void**)&m_d_terms_rows_prefix,
                              sizeof(IndexT) * (m_num_terms + 1)));

        // check that the output of all ops are the same (this represents the
        // number of columns in the sparse Jacobian which should be the same)
        for (int i = 1; i < ops.size(); ++i) {
            // vertex column
            if (ops[0] == Op::V || ops[0] == Op::VV || ops[0] == Op::EV ||
                ops[0] == Op::FV || ops[0] == Op::EVDiamond) {
                if (!(ops[i] == Op::V || ops[i] == Op::VV || ops[i] == Op::EV ||
                      ops[i] == Op::FV || ops[i] == Op::EVDiamond)) {
                    RXMESH_ERROR(
                        "JacobianSparseMatrix::JacobianSparseMatrix() mismatch "
                        "between op output in the input ops");
                }
            }

            // edge column
            if (ops[0] == Op::E || ops[0] == Op::VE || ops[0] == Op::EE ||
                ops[0] == Op::FE) {
                if (!(ops[i] == Op::E || ops[i] == Op::VE || ops[i] == Op::EE ||
                      ops[i] == Op::FE)) {
                    RXMESH_ERROR(
                        "JacobianSparseMatrix::JacobianSparseMatrix() mismatch "
                        "between op output in the input ops");
                }
            }

            // face column
            if (ops[0] == Op::F || ops[0] == Op::VF || ops[0] == Op::EF ||
                ops[0] == Op::FF) {
                if (!(ops[i] == Op::F || ops[i] == Op::VF || ops[i] == Op::EF ||
                      ops[i] == Op::FF)) {
                    RXMESH_ERROR(
                        "JacobianSparseMatrix::JacobianSparseMatrix() mismatch "
                        "between op output in the input ops");
                }
            }
        }

        // check that the y component for all blocks has the same dimensions
        for (int i = 1; i < block_shapes.size(); ++i) {
            if (block_shapes[i].y != block_shapes[0].y) {
                RXMESH_ERROR(
                    "JacobianSparseMatrix::JacobianSparseMatrix() mismatch "
                    "in the block dim.");
            }
        }

        // calc num_rows and cols in the matrix
        m_num_rows = 0;
        m_num_cols = 0;
        for (int i = 0; i < m_num_terms; ++i) {
            auto [rows, cols] =
                get_num_rows_and_cols(rx, m_ops[i], m_block_shapes[i]);

            m_num_rows += rows;
            m_h_terms_rows_prefix[i] = rows;
            if (i == 0) {
                m_num_cols = cols;
            } else {
                if (cols != m_num_cols) {
                    RXMESH_ERROR(
                        "JacobianSparseMatrix::JacobianSparseMatrix() mismatch "
                        "in the number of columns!");
                }
            }
        }

        // scan m_h_terms_rows_prefix
        int prv = 0;
        for (int i = 0; i <= m_num_terms; ++i) {
            IndexT temp = m_h_terms_rows_prefix[i];

            m_h_terms_rows_prefix[i] = prv;
            prv += temp;
        }

        //????
        /*if (add_diagonal && (block_shape.x != block_shape.y)) {
            RXMESH_ERROR(
                "SparseMatrix::SparseMatrix() adding diagonal blocks for "
                "non-symmetric blocks has not been tested before!");
        }*/


        // row pointer allocation and init with prefix sum for CSR
        CUDA_ERROR(cudaMalloc((void**)&m_d_row_ptr,
                              (m_num_rows + 1) * sizeof(IndexT)));
        CUDA_ERROR(
            cudaMalloc((void**)&m_d_row_acc, m_num_rows * sizeof(IndexT)));

        CUDA_ERROR(
            cudaMemset(m_d_row_ptr, 0, (m_num_rows + 1) * sizeof(IndexT)));


        // count the nnz per row
        for (int i = 0; i < m_num_terms; ++i) {
            bool add_diagonal = false;
            if (m_ops[i] == Op::V || m_ops[i] == Op::E || m_ops[i] == Op::F ||
                m_ops[i] == Op::VV || m_ops[i] == Op::FF ||
                m_ops[i] == Op::EE) {

                if (m_block_shapes[i].x == m_block_shapes[i].y) {
                    add_diagonal = true;
                }
            }
            this->mat_prescan(rx,
                              m_ops[i],
                              m_d_row_ptr + m_h_terms_rows_prefix[i],
                              m_block_shapes[i],
                              add_diagonal);
        }


        // prefix sum using CUB.
        m_d_cub_temp_storage = nullptr;

        cub::DeviceScan::ExclusiveSum(m_d_cub_temp_storage,
                                      m_cub_temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      m_num_rows + 1);
        CUDA_ERROR(cudaMalloc((void**)&m_d_cub_temp_storage,
                              m_cub_temp_storage_bytes));

        cub::DeviceScan::ExclusiveSum(m_d_cub_temp_storage,
                                      m_cub_temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      m_num_rows + 1);


        // get nnz
        CUDA_ERROR(cudaMemcpy(&m_nnz,
                              (m_d_row_ptr + m_num_rows),
                              sizeof(IndexT),
                              cudaMemcpyDeviceToHost));

        this->update_max_nnz();

        // column index allocation and init
        CUDA_ERROR(
            cudaMalloc((void**)&m_d_col_idx, m_max_nnz * sizeof(IndexT)));

        // fill in col_idx
        for (int i = 0; i < m_num_terms; ++i) {
            bool add_diagonal = false;
            if (m_ops[i] == Op::V || m_ops[i] == Op::E || m_ops[i] == Op::F ||
                m_ops[i] == Op::VV || m_ops[i] == Op::FF ||
                m_ops[i] == Op::EE) {

                if (m_block_shapes[i].x == m_block_shapes[i].y) {
                    add_diagonal = true;
                }
            }

            this->mat_col_fill(rx,
                               m_ops[i],
                               m_d_row_ptr + m_h_terms_rows_prefix[i],
                               m_d_col_idx,
                               m_block_shapes[i],
                               add_diagonal);
        }


        // allocate value ptr
        CUDA_ERROR(cudaMalloc((void**)&m_d_val, m_max_nnz * sizeof(T)));
        CUDA_ERROR(cudaMemset(m_d_val, 0, m_nnz * sizeof(T)));
        m_allocated = m_allocated | DEVICE;

        alloce_and_move_to_host();


        CUDA_ERROR(cudaMemcpy(m_d_terms_rows_prefix,
                              m_h_terms_rows_prefix,
                              (m_num_terms + 1) * sizeof(IndexT),
                              cudaMemcpyHostToDevice));

        // create cusparse matrix
        init_cusparse(*this);

#ifndef NDEBUG
        check_repeated_indices();
#endif
        init_cudss(*this);
    }

    /**
     * @brief release all allocated memory
     */
    __host__ virtual void release() override
    {
        free(m_ops);
        free(m_block_shapes);
        free(m_h_terms_rows_prefix);
        GPU_FREE(m_d_terms_rows_prefix);
        SparseMatrix<T>::release();
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
        assert(is_non_zero(row_v, col_v));

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
        assert(is_non_zero(row_v, col_v));

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
        assert(is_non_zero(row_v, col_v));

        const IndexT r_id =
            this->get_row_id(row_v) * this->m_replicate + local_i;
        const IndexT c_id =
            this->get_row_id(col_v) * this->m_replicate + local_j;

        return {r_id, c_id};
    }


    /**
     * @brief return the number of terms in this Jacobian matrix
     * @return
     */
    __device__ __host__ IndexT get_num_terms() const
    {
        return m_num_terms;
    }
    /**
     * @brief get the number of rows for specific term in this Jacobian matrix
     */
    __device__ __host__ IndexT get_term_num_rows(IndexT i) const
    {
#ifdef __CUDA_ARCH__
        return m_d_terms_rows_prefix[i + 1] - m_d_terms_rows_prefix[i];
#else
        return m_h_terms_rows_prefix[i + 1] - m_h_terms_rows_prefix[i];
#endif
    }

    /**
     * @brief get the start and end of the row index for a specific term
     */
    __device__ __host__ std::pair<IndexT, IndexT> get_term_rows_range(
        IndexT i) const
    {
#ifdef __CUDA_ARCH__
        return {m_d_terms_rows_prefix[i], m_d_terms_rows_prefix[i + 1]};
#else
        return {m_h_terms_rows_prefix[i], m_h_terms_rows_prefix[i + 1]};
#endif
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

   private:
    int         m_num_terms;
    IndexT*     m_h_terms_rows_prefix;
    IndexT*     m_d_terms_rows_prefix;
    Op*         m_ops;
    BlockShape* m_block_shapes;
};

}  // namespace rxmesh
