#pragma once
#include <cub/block/block_reduce.cuh>
#include "rxmesh/util/macros.h"
namespace RXMESH {

template <class T>
class RXMeshAttribute;

template <class T>
__global__ void rxmesh_attribute_axpy(const RXMeshAttribute<T> X,
                                      const T*                 alpha,
                                      RXMeshAttribute<T>       Y,
                                      const T*                 beta,
                                      const uint32_t attribute_id = INVALID32)
{
    // Y = alpha*X + beta*Y
    // if attribute is INVALID32, then the operation is applied to all
    // attribute (one thread per mesh element on all attribute)
    // otherwise, the operation is applied on only that attribute

    // alpha and beta should be of size attributes per element if attribute ==
    // INVALID32. Otherwise, they should point to a single variable

    assert(X.get_num_mesh_elements() == Y.get_num_mesh_elements());
    assert(X.get_num_attribute_per_element() ==
           Y.get_num_attribute_per_element());

    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < X.get_num_mesh_elements()) {

        if (attribute_id == INVALID32) {
            for (uint32_t attr = 0; attr < X.get_num_attribute_per_element();
                 ++attr) {
                Y(idx, attr) =
                    alpha[attr] * X(idx, attr) + beta[attr] * Y(idx, attr);
            }
        } else {
            Y(idx, attribute_id) = alpha[0] * X(idx, attribute_id) +
                                   beta[0] * Y(idx, attribute_id);
        }
    }
}


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
}  // namespace RXMESH