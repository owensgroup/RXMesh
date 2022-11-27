#pragma once

#include <iostream>
#include <vector>
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/types.h"

namespace rxmesh {

template <typename T>
struct DenseMatInfo
{
    DenseMatInfo(uint32_t row_size, uint32_t col_size)
        : m_nnz_row_size(row_size), m_nnz_col_size(col_size)
    {
        cudaMalloc((void**)&m_d_val, bytes());
    }

    void set_ones()
    {
        std::vector<T> init_tmp_arr(m_nnz_row_size * m_nnz_col_size, 1);
        CUDA_ERROR(cudaMemcpy(m_d_val,
                              init_tmp_arr.data(),
                              bytes() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    // __host__ __device__ T& operator()(const u_int32_t row,
    //                                   const u_int32_t col)
    // {
    //     return m_d_val[row * m_nnz_col_size + col];
    // }

    // __host__ __device__ T& operator()(const u_int32_t row,
    //                                   const u_int32_t col) const
    // {
    //     return m_d_val[row * m_nnz_col_size + col];
    // }

        __host__ __device__ T& operator()(const u_int32_t row,
                                      const u_int32_t col)
    {
        return m_d_val[col * m_nnz_row_size + row]; // pitch & stride
    }

    __host__ __device__ T& operator()(const u_int32_t row,
                                      const u_int32_t col) const
    {
        return m_d_val[col * m_nnz_row_size + row];
    }

    uint32_t bytes() const
    {
        return m_nnz_row_size * m_nnz_col_size * sizeof(T);
    }

    uint32_t m_nnz_row_size;
    uint32_t m_nnz_col_size;
    T*       m_d_val;
};

}  // namespace rxmesh
