#pragma once

#include <iostream>
#include <vector>
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/types.h"

namespace rxmesh {

template <typename T, typename IndexT = int>
struct DenseMatInfo
{
    DenseMatInfo(IndexT row_size, IndexT col_size)
        : m_row_size(row_size), m_col_size(col_size)
    {
        cudaMalloc((void**)&m_d_val, bytes());
        m_ld = m_col_size; // col major for now
    }

    void set_ones()
    {
        std::vector<T> init_tmp_arr(m_row_size * m_col_size, 1);
        CUDA_ERROR(cudaMemcpy(m_d_val,
                              init_tmp_arr.data(),
                              bytes() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    // __host__ __device__ T& operator()(const u_int32_t row,
    //                                   const u_int32_t col)
    // {
    //     return m_d_val[row * m_col_size + col];
    // }

    // __host__ __device__ T& operator()(const u_int32_t row,
    //                                   const u_int32_t col) const
    // {
    //     return m_d_val[row * m_col_size + col];
    // }

    __host__ __device__ T& operator()(const u_int32_t row, const u_int32_t col)
    {
        return m_d_val[col * m_row_size + row];  // pitch & stride
    }

    __host__ __device__ T& operator()(const u_int32_t row,
                                      const u_int32_t col) const
    {
        return m_d_val[col * m_row_size + row];
    }

    T* data() const
    {
        return m_d_val;
    }

    T* col_data(const u_int32_t col) const
    {
        return m_d_val + col * m_row_size;
    }

    IndexT bytes() const
    {
        return m_row_size * m_col_size * sizeof(T);
    }

    IndexT m_ld;
    IndexT m_row_size;
    IndexT m_col_size;
    T*       m_d_val;
};

}  // namespace rxmesh
