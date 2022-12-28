#pragma once

#include <iostream>
#include <vector>
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/types.h"

namespace rxmesh {

// Currently this is device & col major only
// host/device and leading dimension compatiobility will be added
template <typename T, typename IndexT = int>
struct DenseMatInfo
{
    DenseMatInfo(IndexT row_size, IndexT col_size)
        : m_row_size(row_size), m_col_size(col_size)
    {
        cudaMalloc((void**)&m_d_val, bytes());
        m_is_row_major = false;
    }

    DenseMatInfo(IndexT row_size, IndexT col_size, bool is_row_major)
        : m_row_size(row_size),
          m_col_size(col_size),
          m_is_row_major(is_row_major)
    {
        cudaMalloc((void**)&m_d_val, bytes());
    }

    void set_ones()
    {
        std::vector<T> init_tmp_arr(m_row_size * m_col_size, 1);
        CUDA_ERROR(cudaMemcpy(m_d_val,
                              init_tmp_arr.data(),
                              bytes() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    IndexT& lead_dim() const
    {
        if (m_is_row_major) {
            return m_row_size;
        } else {
            return m_col_size;
        }
    }

    __device__ T& operator()(const uint32_t row, const uint32_t col)
    {
        if (m_is_row_major) {
            return m_d_val[row * m_row_size + col];
        } else {
            return m_d_val[col * m_row_size + row];
        }
    }

    __device__ T& operator()(const uint32_t row, const uint32_t col) const
    {
        if (m_is_row_major) {
            return m_d_val[row * m_row_size + col];
        } else {
            return m_d_val[col * m_row_size + row];
        }
    }

    T* data() const
    {
        return m_d_val;
    }

    T* col_data(const uint32_t col) const
    {
        if (m_is_row_major) {
            RXMESH_ERROR(
                "Row major format!"
                "Can't be accessed by column");
        }
        return m_d_val + col * m_row_size;
    }

    IndexT bytes() const
    {
        return m_row_size * m_col_size * sizeof(T);
    }

    bool   m_is_row_major;
    IndexT m_row_size;
    IndexT m_col_size;
    T*     m_d_val;
};

}  // namespace rxmesh
