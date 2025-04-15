#pragma once

#include "rxmesh/matrix/sparse_matrix.h"

namespace rxmesh {
/**
 * @brief Implement sparse matrix where the number of nnz per is fixed
 */
template <typename T, int RowNNZ>
struct SparseMatrixConstantNNZRow : public SparseMatrix<T>
{
    using IndexT = typename SparseMatrix<T>::IndexT;

    /**
     * @brief the constructor only builds the row_ptr. The user is responsible
     * of populating the col_idx and moving it to the device/host. Pointer to
     * col_idx can be accessed via col_idx()
     */
    SparseMatrixConstantNNZRow(const RXMeshStatic& rx,
                               IndexT              num_rows,
                               IndexT              num_cols)
        : SparseMatrix<T>()
    {
        this->m_context   = rx.get_context();
        this->m_replicate = 1;

        this->m_num_rows = num_rows * this->m_replicate;
        this->m_num_cols = num_cols * this->m_replicate;
        this->m_nnz      = this->m_num_rows * RowNNZ;

        int num_row_1 = this->m_num_rows + 1;

        // alloc device
        CUDA_ERROR(cudaMalloc((void**)&this->m_d_col_idx,
                              this->m_nnz * sizeof(IndexT)));
        CUDA_ERROR(cudaMalloc((void**)&this->m_d_val, this->m_nnz * sizeof(T)));
        CUDA_ERROR(
            cudaMalloc((void**)&this->m_d_row_ptr, num_row_1 * sizeof(IndexT)));


        this->m_allocated = this->m_allocated | DEVICE;

        // alloc host
        this->m_h_val = static_cast<T*>(malloc(this->m_nnz * sizeof(T)));
        this->m_h_row_ptr =
            static_cast<IndexT*>(malloc(num_row_1 * sizeof(IndexT)));
        this->m_h_col_idx =
            static_cast<IndexT*>(malloc(this->m_nnz * sizeof(IndexT)));
        this->m_allocated = this->m_allocated | HOST;

        for (int i = 0; i < num_row_1; ++i) {
            this->m_h_row_ptr[i] = i * RowNNZ;
        }

        // copy to device
        CUDA_ERROR(cudaMemcpy(this->m_d_row_ptr,
                              this->m_h_row_ptr,
                              num_row_1 * sizeof(IndexT),
                              cudaMemcpyHostToDevice));

        // create cusparse matrix
        this->init_cusparse(*this);

#ifndef NDEBUG
        // can not do this check because it is done on m_h_col_idx which is not
        //  populated yet
        // check_repeated_indices();
#endif
    }

    /**
     * @brief return the column index pointer of the CSR matrix
     * @return
     */
    __device__ __host__ IndexT* col_idx()
    {
#ifdef __CUDA_ARCH__
        return this->m_d_col_idx;
#else
        return this->m_h_col_idx;
#endif
    }

    /**
     * @brief move col_idx to/from device/host
     */
    __host__ void move_col_idx(locationT    source,
                               locationT    target,
                               cudaStream_t stream = NULL)
    {
        if (source == target) {
            RXMESH_WARN(
                "SparseMatrixConstantNNZRow::move_col_idx() source ({}) and "
                "target ({}) are the same.",
                location_to_string(source),
                location_to_string(target));
            return;
        }

        if ((source == HOST || source == DEVICE) &&
            ((source & this->m_allocated) != source)) {
            RXMESH_ERROR(
                "SparseMatrixConstantNNZRow::move_col_idx() moving source is "
                "not valid because it was not allocated on source i.e., {}",
                location_to_string(source));
            return;
        }

        if (((target & HOST) == HOST || (target & DEVICE) == DEVICE) &&
            ((target & this->m_allocated) != target)) {
            RXMESH_ERROR(
                "SparseMatrixConstantNNZRow::move_col_idx() target {} is not "
                "allocated!",
                location_to_string(target));
            return;
        }

        if (source == HOST && target == DEVICE) {
            CUDA_ERROR(cudaMemcpyAsync(this->m_d_col_idx,
                                       this->m_h_col_idx,
                                       this->m_nnz * sizeof(IndexT),
                                       cudaMemcpyHostToDevice,
                                       stream));
        } else if (source == DEVICE && target == HOST) {
            CUDA_ERROR(cudaMemcpyAsync(this->m_h_col_idx,
                                       this->m_d_col_idx,
                                       this->m_nnz * sizeof(IndexT),
                                       cudaMemcpyDeviceToHost,
                                       stream));
        }
    }
};
}  // namespace rxmesh