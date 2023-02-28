#pragma once
#include <vector>
#include "cusparse.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/types.h"

namespace rxmesh {

// Currently this is device only
// host/device transormation will be added
// only col major is supportedF
template <typename T, typename IndexT = int>
struct DenseMatrix
{
    DenseMatrix(IndexT row_size, IndexT col_size)
        : m_row_size(row_size), m_col_size(col_size), m_dendescr(NULL)
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

    __device__ T& operator()(const uint32_t row, const uint32_t col)
    {
        //assert(col < m_col_size);
        return m_d_val[col * m_row_size + row];
    }

    __device__ T& operator()(const uint32_t row, const uint32_t col) const
    {
        return m_d_val[col * m_row_size + row];
    }

    T* data() const
    {
        return m_d_val;
    }

    T* col_data(const uint32_t ld_idx) const
    {
        return m_d_val + ld_idx * lead_dim();
    }

    IndexT bytes() const
    {
        return m_row_size * m_col_size * sizeof(T);
    }

    // TODO: something like attribute->move()

    cusparseDnMatDescr_t m_dendescr;
    IndexT               m_row_size;
    IndexT               m_col_size;
    T*                   m_d_val;
};

}  // namespace rxmesh
