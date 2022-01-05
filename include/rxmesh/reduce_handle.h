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
template <typename T>
class ReduceHandle
{

   public:
    ReduceHandle()                    = default;
    ReduceHandle(const ReduceHandle&) = default;

    /**
     * @brief Constructor which allocates internal memory used in all reduce
     * operations
     * @param attr one of Attribute used for subsequent reduction
     * operations
     */
    ReduceHandle(const Attribute<T>& attr) : m_num_patches(attr.m_num_patches)
    {
        CUDA_ERROR(
            cudaMalloc(&m_d_reduce_1st_stage, m_num_patches * sizeof(T)));

        CUDA_ERROR(cudaMalloc(&m_d_reduce_2nd_stage, sizeof(T)));

        m_d_reduce_temp_storage = NULL;
        cub::DeviceReduce::Sum(m_d_reduce_temp_storage,
                               m_reduce_temp_storage_bytes,
                               m_d_reduce_1st_stage,
                               m_d_reduce_2nd_stage,
                               m_num_patches);

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
     * @param stream stream to run the computation on
     * @return the output of dot product on the host
     */
    T dot(const Attribute<T>& attr1,
          const Attribute<T>& attr2,
          cudaStream_t        stream = NULL)
    {
        if ((attr1.get_allocated() & DEVICE) != DEVICE ||
            (attr2.get_allocated() & DEVICE) != DEVICE) {
            RXMESH_ERROR(
                "ReduceHandle::dot() input attributes to should be "
                "allocated on the device");
        }

        detail::dot_kernel<T, attr1.m_block_size>
            <<<m_num_patches, attr1.m_block_size, 0, stream>>>(
                attr1,
                attr2,
                attr1.m_d_element_per_patch,
                m_num_patches,
                attr1.get_num_attributes(),
                m_d_reduce_1st_stage);

        return reduce_2nd_stage(stream);
    }

    /**
     * @brief compute L2 norm between two input attributes and return the output
     * on the host
     * @param attr input attribute
     * @param stream stream to run the computation on
     * @return the output of L2 norm on the host
     */
    T norm2(const Attribute<T>& attr, cudaStream_t stream = NULL)
    {
        if ((attr.get_allocated() & DEVICE) != DEVICE) {
            RXMESH_ERROR(
                "ReduceHandle::norm2() input attribute to should be "
                "allocated on the device");
        }

        detail::norm2_kernel<T, attr.m_block_size>
            <<<m_num_patches, attr.m_block_size, 0, stream>>>(
                attr,
                attr.m_d_element_per_patch,
                m_num_patches,
                attr.get_num_attributes(),
                m_d_reduce_1st_stage);

        return std::sqrt(reduce_2nd_stage(stream));
    }


   private:
    T reduce_2nd_stage(cudaStream_t stream)
    {
        T h_output = 0;

        cub::DeviceReduce::Sum(m_d_reduce_temp_storage,
                               m_reduce_temp_storage_bytes,
                               m_d_reduce_1st_stage,
                               m_d_reduce_2nd_stage,
                               m_num_patches,
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
    uint32_t m_num_patches;
};
}  // namespace rxmesh