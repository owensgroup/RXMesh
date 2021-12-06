#pragma once
#include <cub/block/block_reduce.cuh>
#include "rxmesh/util/macros.h"


namespace rxmesh {

template <typename T>
class RXMeshAttribute;

namespace detail {

template <class T, uint32_t blockSize>
__device__ __forceinline__ void cub_block_sum(const T thread_val,
                                              T*      d_block_output)
{
    typedef cub::BlockReduce<T, blockSize>       BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T block_sum = BlockReduce(temp_storage).Sum(thread_val);
    if (threadIdx.x == 0) {
        d_block_output[blockIdx.x] = block_sum;
    }
}

template <class T, uint32_t blockSize>
__launch_bounds__(blockSize) __global__
    void norm2_kernel(const RXMeshAttribute<T> X,
                      const uint16_t*          d_element_per_patch,
                      const uint32_t           num_patches,
                      const uint32_t           num_attributes,
                      T*                       d_block_output)
{
    uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t element_per_patch = d_element_per_patch[p_id];
        T              thread_val        = 0;
        for (uint16_t i = threadIdx.x; i < element_per_patch; i += blockSize) {
            for (uint32_t j = 0; j < num_attributes; ++j) {
                const T val = X(p_id, i, j);
                thread_val += val * val;
            }
        }

        cub_block_sum<T, blockSize>(thread_val, d_block_output);
    }
}


template <typename T, uint32_t blockSize>
__launch_bounds__(blockSize) __global__
    void dot_kernel(const RXMeshAttribute<T> X,
                    const RXMeshAttribute<T> Y,
                    const uint16_t*          d_element_per_patch,
                    const uint32_t           num_patches,
                    const uint32_t           num_attributes,
                    T*                       d_block_output)
{
    assert(X.get_num_attributes() == Y.get_num_attributes());

    uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t element_per_patch = d_element_per_patch[p_id];
        T              thread_val        = 0;
        for (uint16_t i = threadIdx.x; i < element_per_patch; i += blockSize) {
            for (uint32_t j = 0; j < num_attributes; ++j) {
                thread_val += X(p_id, i, j) * Y(p_id, i, j);
            }
        }

        cub_block_sum<T, blockSize>(thread_val, d_block_output);
    }
}

template <typename T>
__global__ void memset_attribute(const RXMeshAttribute<T> attr,
                                 const T                  value,
                                 const uint16_t*          d_element_per_patch,
                                 const uint32_t           num_patches,
                                 const uint32_t           num_attributes)
{
    uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t element_per_patch = d_element_per_patch[p_id];
        for (uint16_t i = threadIdx.x; i < element_per_patch; i += blockDim.x) {
            for (uint32_t j = 0; j < num_attributes; ++j) {
                attr(p_id, i, j) = value;
            }
        }
    }
}

}  // namespace detail
}  // namespace rxmesh