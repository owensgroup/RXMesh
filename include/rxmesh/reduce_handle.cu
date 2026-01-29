#include "rxmesh/reduce_handle.h"

namespace rxmesh {

template <typename T, typename HandleT>
ReduceHandle<T, HandleT>::ReduceHandle(const uint32_t num_patches)
    : m_max_num_patches(num_patches)
{
    size_t type_size = std::max(sizeof(T), sizeof(KeyValue));

    CUDA_ERROR(
        cudaMalloc(&m_d_reduce_1st_stage, m_max_num_patches * type_size));

    CUDA_ERROR(cudaMalloc(&m_d_reduce_2nd_stage, type_size));

    T*     ptr_t        = NULL;
    size_t temp_bytes_t = 0;
    cub::DeviceReduce::Sum(ptr_t,
                           temp_bytes_t,
                           m_d_reduce_1st_stage,
                           m_d_reduce_2nd_stage,
                           m_max_num_patches);

    KeyValue* ptr_p        = NULL;
    size_t    temp_bytes_p = 0;
    cub::DeviceReduce::Reduce(
        ptr_p,
        temp_bytes_p,
        reinterpret_cast<KeyValue*>(m_d_reduce_1st_stage),
        reinterpret_cast<KeyValue*>(m_d_reduce_2nd_stage),
        m_max_num_patches,
        detail::ArgMaxOp<HandleT, T>(),
        KeyValue(HandleT(), std::numeric_limits<T>::lowest()));

    m_d_reduce_temp_storage = NULL;

    m_reduce_temp_storage_bytes = std::max(temp_bytes_p, temp_bytes_t);

    CUDA_ERROR(cudaMalloc((void**)&m_d_reduce_temp_storage,
                          m_reduce_temp_storage_bytes));
}

template <typename T, typename HandleT>
ReduceHandle<T, HandleT>::~ReduceHandle()
{
    GPU_FREE(m_d_reduce_1st_stage);
    GPU_FREE(m_d_reduce_2nd_stage);
    GPU_FREE(m_d_reduce_temp_storage);
    m_reduce_temp_storage_bytes = 0;
}

template <typename T, typename HandleT>
T ReduceHandle<T, HandleT>::dot(const Attribute<T, HandleT>& attr1,
                                const Attribute<T, HandleT>& attr2,
                                uint32_t                     attribute_id,
                                cudaStream_t                 stream)
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

    return reduce_2nd_stage<T>(stream, cub::Sum(), 0);
}

template <typename T, typename HandleT>
T ReduceHandle<T, HandleT>::norm2(const Attribute<T, HandleT>& attr,
                                  uint32_t                     attribute_id,
                                  cudaStream_t                 stream)
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

    return std::sqrt(reduce_2nd_stage<T>(stream, cub::Sum(), 0));
}

template <typename T, typename HandleT>
typename ReduceHandle<T, HandleT>::KeyValue ReduceHandle<T, HandleT>::arg_max(
    const Attribute<T, HandleT>& attr,
    uint32_t                     attribute_id,
    cudaStream_t                 stream)
{
    if ((attr.get_allocated() & DEVICE) != DEVICE) {
        RXMESH_ERROR(
            "ReduceHandle::arg_max() input attribute to should be "
            "allocated on the device");
    }

    detail::ArgMaxOp<HandleT, T> max_pair;

    detail::arg_minmax_kernel<T, attr.m_block_size, HandleT>
        <<<m_max_num_patches, attr.m_block_size, 0, stream>>>(
            attr,
            attribute_id,
            max_pair,
            m_max_num_patches,
            attr.get_num_attributes(),
            reinterpret_cast<KeyValue*>(m_d_reduce_1st_stage));

    KeyValue init(HandleT(), max_pair.default_val());

    return reduce_2nd_stage<KeyValue>(
        stream, detail::ArgMaxOp<HandleT, T>(), init);
}

template <typename T, typename HandleT>
typename ReduceHandle<T, HandleT>::KeyValue ReduceHandle<T, HandleT>::arg_min(
    const Attribute<T, HandleT>& attr,
    uint32_t                     attribute_id,
    cudaStream_t                 stream)
{
    if ((attr.get_allocated() & DEVICE) != DEVICE) {
        RXMESH_ERROR(
            "ReduceHandle::arg_min() input attribute to should be "
            "allocated on the device");
    }

    detail::ArgMinOp<HandleT, T> min_pair;

    detail::arg_minmax_kernel<T, attr.m_block_size, HandleT>
        <<<m_max_num_patches, attr.m_block_size, 0, stream>>>(
            attr,
            attribute_id,
            min_pair,
            m_max_num_patches,
            attr.get_num_attributes(),
            reinterpret_cast<KeyValue*>(m_d_reduce_1st_stage));

    KeyValue init(HandleT(), min_pair.default_val());

    return reduce_2nd_stage<KeyValue>(
        stream, detail::ArgMinOp<HandleT, T>(), init);
}

// Explicit instantiations
#define RXMESH_REDUCE_HANDLE_INSTANTIATE(T)       \
    template class ReduceHandle<T, VertexHandle>; \
    template class ReduceHandle<T, EdgeHandle>;   \
    template class ReduceHandle<T, FaceHandle>;

RXMESH_REDUCE_HANDLE_INSTANTIATE(float)
RXMESH_REDUCE_HANDLE_INSTANTIATE(double)
RXMESH_REDUCE_HANDLE_INSTANTIATE(uint32_t)
RXMESH_REDUCE_HANDLE_INSTANTIATE(int32_t)
RXMESH_REDUCE_HANDLE_INSTANTIATE(uint8_t)
RXMESH_REDUCE_HANDLE_INSTANTIATE(int8_t)

#undef RXMESH_REDUCE_HANDLE_INSTANTIATE

}  // namespace rxmesh
