#pragma once
#include <cub/block/block_reduce.cuh>
#include "rxmesh/util/macros.h"
namespace rxmesh {

template <class T>
class RXMeshAttribute;


template <class T, uint32_t blockSize>
__global__ void rxmesh_attribute_norm2(const RXMeshAttribute<T> X,
                                       const uint32_t           attribute_id,
                                       T*                       d_block_output)
{
    uint32_t idx       = threadIdx.x + blockIdx.x * blockDim.x;
    T        threa_val = 0;
    if (idx < X.get_num_mesh_elements()) {
        threa_val = X(idx, attribute_id);
    }
    threa_val *= threa_val;


    typedef cub::BlockReduce<T, blockSize>       BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T block_sum = BlockReduce(temp_storage).Sum(threa_val);
    if (threadIdx.x == 0) {
        d_block_output[blockIdx.x] = block_sum;
    }
}


template <class T, uint32_t blockSize>
__global__ void rxmesh_attribute_dot(const RXMeshAttribute<T> X,
                                     const RXMeshAttribute<T> Y,
                                     const uint32_t           attribute_id,
                                     T*                       d_block_output)
{
    uint32_t idx       = threadIdx.x + blockIdx.x * blockDim.x;
    T        threa_val = 0;
    if (idx < X.get_num_mesh_elements()) {
        threa_val = X(idx, attribute_id) * Y(idx, attribute_id);
    }

    typedef cub::BlockReduce<T, blockSize>       BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T block_sum = BlockReduce(temp_storage).Sum(threa_val);
    if (threadIdx.x == 0) {
        d_block_output[blockIdx.x] = block_sum;
    }
}

template <class T>
__global__ void memset_attribute(const RXMeshAttribute<T> attr,
                                 const T                  value,
                                 const uint16_t*          d_element_per_patch,
                                 const uint32_t           num_patches,
                                 const uint32_t           num_attributes)
{
    uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        uint16_t element_per_patch = d_element_per_patch[p_id];
        for (uint32_t i = threadIdx.x; i < element_per_patch; i += blockDim.x) {
            for (uint32_t j = 0; j < num_attributes; ++j) {
                attr(p_id, i, j) = value;
            }
        }
    }
}

}  // namespace rxmesh