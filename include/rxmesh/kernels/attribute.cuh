#pragma once
#include <cub/block/block_reduce.cuh>
#include "rxmesh/util/macros.h"


namespace rxmesh {

template <typename T, typename HandleT>
class Attribute;

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

template <class T, uint32_t blockSize, typename HandleT>
__launch_bounds__(blockSize) __global__
    void norm2_kernel(const Attribute<T, HandleT> X,
                      const uint32_t              num_patches,
                      const uint32_t              num_attributes,
                      T*                          d_block_output,
                      uint32_t                    attribute_id)
{
    using LocalT = typename HandleT::LocalT;

    uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t element_per_patch = X.size(p_id);
        T              thread_val        = 0;
        for (uint16_t i = threadIdx.x; i < element_per_patch; i += blockSize) {
            if (X.get_patch_info(p_id).is_owned(LocalT(i)) &&
                !X.get_patch_info(p_id).is_deleted(LocalT(i))) {

                if (attribute_id != INVALID32) {
                    const T val = X(p_id, i, attribute_id);
                    thread_val += val * val;
                } else {
                    for (uint32_t j = 0; j < num_attributes; ++j) {
                        const T val = X(p_id, i, j);
                        thread_val += val * val;
                    }
                }
            }
        }

        cub_block_sum<T, blockSize>(thread_val, d_block_output);
    }
}


template <class T, uint32_t blockSize, typename HandleT>
__launch_bounds__(blockSize) __global__
    void dot_kernel(const Attribute<T, HandleT> X,
                    const Attribute<T, HandleT> Y,
                    const uint32_t              num_patches,
                    const uint32_t              num_attributes,
                    T*                          d_block_output,
                    uint32_t                    attribute_id)
{
    using LocalT = typename HandleT::LocalT;

    assert(X.get_num_attributes() == Y.get_num_attributes());

    uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t element_per_patch = X.size(p_id);
        T              thread_val        = 0;
        for (uint16_t i = threadIdx.x; i < element_per_patch; i += blockSize) {

            if (X.get_patch_info(p_id).is_owned(LocalT(i)) &&
                !X.get_patch_info(p_id).is_deleted(LocalT(i))) {

                if (attribute_id != INVALID32) {
                    thread_val +=
                        X(p_id, i, attribute_id) * Y(p_id, i, attribute_id);
                } else {
                    for (uint32_t j = 0; j < num_attributes; ++j) {
                        thread_val += X(p_id, i, j) * Y(p_id, i, j);
                    }
                }
            }
        }

        cub_block_sum<T, blockSize>(thread_val, d_block_output);
    }
}

template <typename HandleT, typename T>
struct CustomMaxPair
{
    __host__ __device__ CustomMaxPair()
    {
        default_val = (std::numeric_limits<T>::lowest());
    }

    __device__ __forceinline__ cub::KeyValuePair<HandleT, T> operator()(
        const cub::KeyValuePair<HandleT, T>& a,
        const cub::KeyValuePair<HandleT, T>& b) const
    {
        return (b.value > a.value) ? b : a;
    }
    T default_val;
};

template <typename HandleT, typename T>
struct CustomMinPair
{
    __host__ __device__ CustomMinPair()
    {
        default_val = (std::numeric_limits<T>::max());
    }
    __device__ __forceinline__ cub::KeyValuePair<HandleT, T> operator()(
        const cub::KeyValuePair<HandleT, T>& a,
        const cub::KeyValuePair<HandleT, T>& b) const
    {
        return (b.value < a.value) ? b : a;
    }
    T default_val;
};
template <typename HandleT, typename T>
struct CustomSum  // for the purpose of doing the summation with cub since the
                  // cub::KeyValuePair<,> type needs this to do the reduction
                  // operation to determine number of bytes
{
    __device__ __host__ cub::KeyValuePair<HandleT, T> operator()(
        const cub::KeyValuePair<HandleT, T>& a,
        const cub::KeyValuePair<HandleT, T>& b) const
    {
        return cub::KeyValuePair<HandleT, T>(a.key, a.value + b.value);
    }
};

template <class T, uint32_t blockSize, typename HandleT, typename Operation>
__launch_bounds__(blockSize) __global__
    void arg_minmax_kernel(const Attribute<T, HandleT> X,
                    uint32_t                    attribute_id,
                    Operation                   op, //can be either max or min operation
                    const uint32_t              num_patches,
                    const uint32_t              num_attributes,
                    cub::KeyValuePair<HandleT, T>*  d_block_output)
{
    using LocalT = typename HandleT::LocalT;

    assert(X.get_num_attributes() == 1); //we can only take arg max for a scalar attribute

    uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t element_per_patch = X.size(p_id);
        cub::KeyValuePair<HandleT, T> thread_val;
        thread_val.value = op.default_val;
        thread_val.key   = HandleT(p_id, threadIdx.x);
        for (uint16_t i = threadIdx.x; i < element_per_patch; i += blockSize) {

            if (X.get_patch_info(p_id).is_owned(LocalT(i)) &&
                !X.get_patch_info(p_id).is_deleted(LocalT(i))) {

                if (attribute_id != INVALID32 ) 
                {
                    HandleT handle(p_id, i);
                    cub::KeyValuePair<HandleT, T> current_pair(handle, X(p_id, i, attribute_id));
                    thread_val = op(thread_val, current_pair);
                }
                else {
                    for (uint32_t j = 0; j < num_attributes; ++j) 
                    {
                        HandleT handle(p_id, i);
                        cub::KeyValuePair<HandleT, T> current_pair(handle, X(p_id, i, j));
                        thread_val = op(thread_val, current_pair);
                    }
                }
            }
        }
        typedef cub::BlockReduce<cub::KeyValuePair<HandleT, T>, blockSize> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        cub::KeyValuePair<HandleT, T> block_aggregate = BlockReduce(temp_storage).Reduce(thread_val, op);
        if (threadIdx.x == 0) 
        {
            d_block_output[blockIdx.x] = block_aggregate;
        }
    }
}



template <class T, uint32_t blockSize, typename ReductionOp, typename HandleT>
__launch_bounds__(blockSize) __global__
    void generic_reduce(const Attribute<T, HandleT> X,
                        const uint32_t              num_patches,
                        const uint32_t              num_attributes,
                        T*                          d_block_output,
                        ReductionOp                 reduction_op,
                        T                           init,
                        uint32_t                    attribute_id)
{
    using LocalT = typename HandleT::LocalT;

    uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t element_per_patch = X.size(p_id);
        T              thread_val        = init;
        for (uint16_t i = threadIdx.x; i < element_per_patch; i += blockSize) {
            if (X.get_patch_info(p_id).is_owned(LocalT(i)) &&
                !X.get_patch_info(p_id).is_deleted(LocalT(i))) {
                if (attribute_id != INVALID32) {
                    const T val = X(p_id, i, attribute_id);
                    thread_val  = reduction_op(thread_val, val);
                } else {
                    for (uint32_t j = 0; j < num_attributes; ++j) {
                        const T val = X(p_id, i, j);
                        thread_val  = reduction_op(thread_val, val);
                    }
                }
            }
        }
        typedef cub::BlockReduce<T, blockSize>       BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        T block_aggregate =
            BlockReduce(temp_storage).Reduce(thread_val, reduction_op);
        if (threadIdx.x == 0) {
            d_block_output[blockIdx.x] = block_aggregate;
        }
    }
}


template <typename T, typename HandleT>
__global__ void memset_attribute(const Attribute<T, HandleT> attr,
                                 const T                     value,
                                 const uint32_t              num_patches,
                                 const uint32_t              num_attributes)
{
    uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t element_per_patch = attr.capacity(p_id);
        for (uint16_t i = threadIdx.x; i < element_per_patch; i += blockDim.x) {
            for (uint32_t j = 0; j < num_attributes; ++j) {
                attr(p_id, i, j) = value;
            }
        }
    }
}

}  // namespace detail
}  // namespace rxmesh