#pragma once
#include <vector>
#include "cusparse.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/types.h"

namespace rxmesh {

// Currently this is device only
// host/device transormation will be added
// only col major is supported
template <typename T, typename IndexT = int>
struct DenseMatrix
{
    DenseMatrix(IndexT row_size, IndexT col_size)
        : m_row_size(row_size),
          m_col_size(col_size),
          m_dendescr(NULL),
          m_allocated(LOCATION_NONE)
    {
        CUDA_ERROR(cudaMalloc((void**)&m_d_val, bytes()));

        CUSPARSE_ERROR(cusparseCreateDnMat(&m_dendescr,
                                           m_row_size,
                                           m_col_size,
                                           m_row_size,  // leading dim
                                           m_d_val,
                                           CUDA_R_32F,
                                           CUSPARSE_ORDER_COL));
    }

    void set_ones()
    {
        std::vector<T> init_tmp_arr(m_row_size * m_col_size, 1);
        CUDA_ERROR(cudaMemcpy(m_d_val,
                              init_tmp_arr.data(),
                              bytes() * sizeof(T),
                              cudaMemcpyHostToDevice));
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
        return m_d_val[col * m_row_size + row];
#else
        return m_h_val[col * m_row_size + row];
#endif
    }

    __host__ __device__ T& operator()(const uint32_t row,
                                      const uint32_t col) const
    {
        assert(row < m_row_size);
        assert(col < m_col_size);

#ifdef __CUDA_ARCH__
        return m_d_val[col * m_row_size + row];
#else
        return m_h_val[col * m_row_size + row];
#endif
    }

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

    T* col_data(const uint32_t ld_idx, locationT location = DEVICE) const
    {
        if ((location & HOST) == HOST) {
            return m_h_val + ld_idx * lead_dim();
        }

        if ((location & DEVICE) == DEVICE) {
            return m_d_val + ld_idx * lead_dim();
        }

        assert(1 != 1);
        return 0;
    }

    IndexT bytes() const
    {
        return m_row_size * m_col_size * sizeof(T);
    }

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
};

}  // namespace rxmesh
