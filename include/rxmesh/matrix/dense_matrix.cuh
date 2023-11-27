#pragma once
#include <vector>
#include "cusparse.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/types.h"

namespace rxmesh {

/**
 * @brief dense matrix use for device and host, inside is a array.
 * The dense matrix is initialized as col major on device.
 * We would only support col major dense matrix for now since that's what
 * cusparse and cusolver wants.
 */
template <typename T, typename IndexT = int, unsigned int MemAlignSize = 0>
struct DenseMatrix
{
    DenseMatrix(IndexT row_size, IndexT col_size)
        : m_row_size(row_size),
          m_col_size(col_size),
          m_dendescr(NULL),
          m_allocated(LOCATION_NONE),
          m_col_pad_bytes(0),
          m_col_pad_idx(0)
    {
        m_allocated = m_allocated | DEVICE;

        IndexT col_data_bytes = m_row_size * sizeof(T);
        if (MemAlignSize != 0 && col_data_bytes % MemAlignSize != 0) {
            m_col_pad_bytes = MemAlignSize - (col_data_bytes % MemAlignSize);
            m_col_pad_idx   = m_col_pad_bytes / sizeof(T);
        }

        CUDA_ERROR(cudaMalloc((void**)&m_d_val, bytes()));

        CUSPARSE_ERROR(cusparseCreateDnMat(&m_dendescr,
                                           m_row_size,
                                           m_col_size,
                                           m_row_size,  // leading dim
                                           m_d_val,
                                           CUDA_R_32F,
                                           CUSPARSE_ORDER_COL));
    }

    IndexT lead_dim() const
    {
        return m_row_size;
    }

    __host__ __device__ T& operator()(const uint32_t row, const uint32_t col)
    {
        assert(row < m_row_size);
        assert(col < m_col_size);

#ifdef __CUDA_ARCH__
        return m_d_val[col * (m_row_size + m_col_pad_idx) + row];
#else
        return m_h_val[col * (m_row_size + m_col_pad_idx) + row];
#endif
    }

    __host__ __device__ T& operator()(const uint32_t row,
                                      const uint32_t col) const
    {
        assert(row < m_row_size);
        assert(col < m_col_size);

#ifdef __CUDA_ARCH__
        return m_d_val[col * (m_row_size + m_col_pad_idx) + row];
#else
        return m_h_val[col * (m_row_size + m_col_pad_idx) + row];
#endif
    }

    /**
     * @brief return the raw pointer based on the location specified
     */
    T* data(locationT location = DEVICE) const
    {
        if ((location & HOST) == HOST) {
            return m_h_val;
        }

        if ((location & DEVICE) == DEVICE) {
            return m_d_val;
        }

        assert(1 != 1);
        return 0;
    }

    /**
     * @brief return the raw pointer to columns based on column index the
     * location specified and
     */
    T* col_data(const uint32_t ld_idx, locationT location = DEVICE) const
    {
        if ((location & HOST) == HOST) {
            return m_h_val + ld_idx * (m_row_size + m_col_pad_idx);
        }

        if ((location & DEVICE) == DEVICE) {
            return m_d_val + ld_idx * (m_row_size + m_col_pad_idx);
        }

        if ((location & m_allocated) == location) {
            RXMESH_ERROR("Requested data not allocated on {}",
                         location_to_string(location));
        }

        assert(1 != 1);
        return 0;
    }

    /**
     * @brief return the total number bytes used by the array
    */
    IndexT bytes() const
    {
        return (m_row_size + m_col_pad_idx) * m_col_size * sizeof(T);
    }

    /**
     * @brief move the data between host an device 
    */
    void move(locationT source, locationT target, cudaStream_t stream = NULL)
    {
        if (source == target) {
            RXMESH_WARN(
                "DenseMatrix::move() source ({}) and target ({}) "
                "are the same.",
                location_to_string(source),
                location_to_string(target));
            return;
        }

        if ((source == HOST || source == DEVICE) &&
            ((source & m_allocated) != source)) {
            RXMESH_ERROR(
                "DenseMatrix::move() moving source is not valid"
                " because it was not allocated on source i.e., {}",
                location_to_string(source));
        }

        if (((target & HOST) == HOST || (target & DEVICE) == DEVICE) &&
            ((target & m_allocated) != target)) {
            RXMESH_WARN(
                "DenseMatrix::move() allocating target before moving to {}",
                location_to_string(target));
            allocate(target);
        }

        if (source == HOST && target == DEVICE) {
            CUDA_ERROR(cudaMemcpyAsync(
                m_d_val, m_h_val, bytes(), cudaMemcpyHostToDevice, stream));
        } else if (source == DEVICE && target == HOST) {
            CUDA_ERROR(cudaMemcpyAsync(
                m_h_val, m_d_val, bytes(), cudaMemcpyDeviceToHost, stream));
        }
    }

    /**
     * @brief release the data on host or device 
    */
    void release(locationT location = LOCATION_ALL)
    {
        if (((location & HOST) == HOST) && ((m_allocated & HOST) == HOST)) {
            free(m_h_val);
            m_h_val     = nullptr;
            m_allocated = m_allocated & (~HOST);
        }

        if (((location & DEVICE) == DEVICE) &&
            ((m_allocated & DEVICE) == DEVICE)) {
            GPU_FREE(m_d_val);
            m_allocated = m_allocated & (~DEVICE);
        }
    }

    /**
     * @brief allocate the data on host or device 
    */
    void allocate(locationT location)
    {
        if ((location & HOST) == HOST) {
            release(HOST);

            m_h_val = static_cast<T*>(malloc(bytes()));

            m_allocated = m_allocated | HOST;
        }

        if ((location & DEVICE) == DEVICE) {
            release(DEVICE);

            CUDA_ERROR(cudaMalloc((void**)&m_d_val, bytes()));

            m_allocated = m_allocated | DEVICE;
        }
    }

    // TODO: something like attribute->move()

    cusparseDnMatDescr_t m_dendescr;
    locationT            m_allocated;
    IndexT               m_row_size;
    IndexT               m_col_size;
    T*                   m_d_val;
    T*                   m_h_val;

    IndexT m_col_pad_bytes;
    IndexT m_col_pad_idx;
};

}  // namespace rxmesh
