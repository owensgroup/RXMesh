#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/kernels/attribute.cuh"

namespace rxmesh {

/**
 * @brief This class is used to compute different reduction operations on
 * Attribute. To create a new ReduceHandle, use create_reduce_handle()
 * from Attribute
 * @tparam T The type of the attribute
 */
template <typename T, typename HandleT>
class ReduceHandle
{

   public:
    using HandleType = HandleT;
    using Type       = T;

    ReduceHandle()                    = default;
    ReduceHandle(const ReduceHandle&) = default;

    /**
     * @brief Constructor which allocates internal memory used in all reduce
     * operations
     * @param attr one of Attribute used for subsequent reduction
     * operations
     */
    ReduceHandle(const Attribute<T, HandleT>& attr)
        : m_max_num_patches(attr.m_max_num_patches)
    {
        CUDA_ERROR(
            cudaMalloc(&m_d_reduce_1st_stage, m_max_num_patches * sizeof(T)));

        CUDA_ERROR(cudaMalloc(&m_d_reduce_2nd_stage, sizeof(T)));

        m_d_reduce_temp_storage = NULL;
        cub::DeviceReduce::Sum(m_d_reduce_temp_storage,
                               m_reduce_temp_storage_bytes,
                               m_d_reduce_1st_stage,
                               m_d_reduce_2nd_stage,
                               m_max_num_patches);

        CUDA_ERROR(
            cudaMalloc(&m_d_reduce_temp_storage, m_reduce_temp_storage_bytes));
    }

    ~ReduceHandle()
    {
        GPU_FREE(m_d_reduce_1st_stage);
        GPU_FREE(m_d_reduce_2nd_stage);
        GPU_FREE(m_d_reduce_temp_storage);
        m_reduce_temp_storage_bytes = 0;
    }

    /**
     * @brief compute dot product between two input attributes and return the
     * output on the host
     * @param attr1 first input attribute
     * @param attr2 second input attribute
     * @param attribute_id specific attribute ID to compute its dot product.
     * Default is INVALID32 which compute dot product for all attributes
     * @param stream stream to run the computation on
     * @return the output of dot product on the host
     */
    T dot(const Attribute<T, HandleT>& attr1,
          const Attribute<T, HandleT>& attr2,
          uint32_t                     attribute_id = INVALID32,
          cudaStream_t                 stream       = NULL)
    {
        if ((attr1.get_allocated() & DEVICE) != DEVICE ||
            (attr2.get_allocated() & DEVICE) != DEVICE) {
            RXMESH_ERROR(
                "ReduceHandle::dot() input attributes to should be "
                "allocated on the device");
        }

        detail::dot_kernel<T, attr1.m_block_size>
            <<<m_max_num_patches, attr1.m_block_size, 0, stream>>>(
                attr1,
                attr2,                
                m_max_num_patches,
                attr1.get_num_attributes(),
                m_d_reduce_1st_stage,
                attribute_id);

        return reduce_2nd_stage(stream, cub::Sum(), 0);
    }

    /**
     * @brief compute L2 norm between two input attributes and return the output
     * on the host
     * @param attr input attribute
     * @param attribute_id specific attribute ID to compute its norm2. Default
     * is INVALID32 which compute norm2 for all attributes
     * @param stream stream to run the computation on
     * @return the output of L2 norm on the host
     */
    T norm2(const Attribute<T, HandleT>& attr,
            uint32_t                     attribute_id = INVALID32,
            cudaStream_t                 stream       = NULL)
    {
        if ((attr.get_allocated() & DEVICE) != DEVICE) {
            RXMESH_ERROR(
                "ReduceHandle::norm2() input attribute to should be "
                "allocated on the device");
        }

        detail::norm2_kernel<T, attr.m_block_size>
            <<<m_max_num_patches, attr.m_block_size, 0, stream>>>(
                attr,                
                m_max_num_patches,
                attr.get_num_attributes(),
                m_d_reduce_1st_stage,
                attribute_id);

        return std::sqrt(reduce_2nd_stage(stream, cub::Sum(), 0));
    }

    /**
     * @brief performn generic reduction operations on an input attribute
     * @tparam ReductionOp type of the binary reduction functor having member T
     * operator()(const T &a, const T &b)
     * @param attr input attribute
     * @param reduction_op the binary reduction functor. It is possible to use
     * CUB built-in reduction functor like cub::Max(), cub::Sum(). An example of
     * user-defined:
     *
     * struct CustomMin
     * {
     *     template <typename T>
     *     __device__ __forceinline__ T operator()(const T& a, const T& b) const
     *     {
     *         return (b < a) ? b : a;
     *     }
     * };
     * Read more about reduction from CUB doc
     * https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html
     * @param init initial value for reduction. This should be the "neutral"
     * value for the reduction operations e.g., 0 for sum, 1 for multiplication,
     * 0 for max on uint32_t
     * @param attribute_id specific attribute ID to compute its reduction.
     * Default is INVALID32 which compute reduction for all attributes
     * @param stream stream to run the computation on
     * @return the reduced output on the host
     */

    template <typename ReductionOp>
    T reduce(const Attribute<T, HandleT>& attr,
             ReductionOp                  reduction_op,
             T                            init,
             uint32_t                     attribute_id = INVALID32,
             cudaStream_t                 stream       = NULL)
    {
        if ((attr.get_allocated() & DEVICE) != DEVICE) {
            RXMESH_ERROR(
                "ReduceHandle::reduce() input attribute to should be "
                "allocated on the device");
        }

        detail::generic_reduce<T, attr.m_block_size>
            <<<m_max_num_patches, attr.m_block_size, 0, stream>>>(
                attr,                
                m_max_num_patches,
                attr.get_num_attributes(),
                m_d_reduce_1st_stage,
                reduction_op,
                init,
                attribute_id);

        return reduce_2nd_stage(stream, reduction_op, init);
    }

   private:
    template <typename ReductionOp>
    T reduce_2nd_stage(cudaStream_t stream, ReductionOp reduction_op, T init)
    {
        T h_output = 0;

        cub::DeviceReduce::Reduce(m_d_reduce_temp_storage,
                                  m_reduce_temp_storage_bytes,
                                  m_d_reduce_1st_stage,
                                  m_d_reduce_2nd_stage,
                                  m_max_num_patches,
                                  reduction_op,
                                  init,
                                  stream);

        CUDA_ERROR(cudaMemcpyAsync(&h_output,
                                   m_d_reduce_2nd_stage,
                                   sizeof(T),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_ERROR(cudaStreamSynchronize(stream));

        return h_output;
    }

    size_t   m_reduce_temp_storage_bytes;
    T*       m_d_reduce_1st_stage;
    T*       m_d_reduce_2nd_stage;
    void*    m_d_reduce_temp_storage;
    uint32_t m_max_num_patches;
};

template <class T>
using VertexReduceHandle = ReduceHandle<T, VertexHandle>;

template <class T>
using EdgeReduceHandle = ReduceHandle<T, EdgeHandle>;

template <class T>
using FaceReduceHandle = ReduceHandle<T, FaceHandle>;

}  // namespace rxmesh