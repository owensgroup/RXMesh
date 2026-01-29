#include <cstring>
#include <memory>
#include <vector>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/attribute.h"
#include "rxmesh/handle.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/types.h"

#include <glm/gtx/norm.hpp>

#include "thrust/complex.h"

namespace rxmesh {

namespace detail {
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

template <class T, typename HandleT>
Attribute<T, HandleT>::Attribute()
    : AttributeBase(),
      m_rxmesh(nullptr),
      m_h_patches_info(nullptr),
      m_d_patches_info(nullptr),
      m_name(nullptr),
      m_num_attributes(0),
      m_allocated(LOCATION_NONE),
      m_h_attr(nullptr),
      m_h_ptr_on_device(nullptr),
      m_d_attr(nullptr),
      m_max_num_patches(0),
      m_layout(AoS),
      m_memory_mega_bytes(0)
{
    this->m_name    = (char*)malloc(sizeof(char) * 1);
    this->m_name[0] = '\0';
}

template <class T, typename HandleT>
Attribute<T, HandleT>::Attribute(const char*    name,
                                 const uint32_t num_attributes,
                                 locationT      location,
                                 const layoutT  layout,
                                 RXMeshStatic*  rxmesh)
    : AttributeBase(),
      m_rxmesh(rxmesh),
      m_h_patches_info(rxmesh->m_h_patches_info),
      m_d_patches_info(rxmesh->m_d_patches_info),
      m_name(nullptr),
      m_num_attributes(num_attributes),
      m_allocated(LOCATION_NONE),
      m_h_attr(nullptr),
      m_h_ptr_on_device(nullptr),
      m_d_attr(nullptr),
      m_max_num_patches(rxmesh->get_max_num_patches()),
      m_layout(layout),
      m_memory_mega_bytes(0)
{
    if (name != nullptr) {
        this->m_name = (char*)malloc(sizeof(char) * (strlen(name) + 1));
        strcpy(this->m_name, name);
    }

    if (m_rxmesh->get_num_patches() == 0) {
        return;
    }

    allocate(location);
}

template <class T, typename HandleT>
T& Attribute<T, HandleT>::operator()(size_t i, size_t j)
{
    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        return this->operator()(m_rxmesh->map_to_local_vertex(i), j);
    }
    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        return this->operator()(m_rxmesh->map_to_local_edge(i), j);
    }
    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        return this->operator()(m_rxmesh->map_to_local_face(i), j);
    }
}

template <class T, typename HandleT>
T& Attribute<T, HandleT>::operator()(size_t i, size_t j) const
{
    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        return this->operator()(m_rxmesh->map_to_local_vertex(i), j);
    }
    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        return this->operator()(m_rxmesh->map_to_local_edge(i), j);
    }
    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        return this->operator()(m_rxmesh->map_to_local_face(i), j);
    }
}

template <class T, typename HandleT>
size_t Attribute<T, HandleT>::rows() const
{
    return size();
}

template <class T, typename HandleT>
size_t Attribute<T, HandleT>::cols() const
{
    return this->get_num_attributes();
}

template <class T, typename HandleT>
uint32_t Attribute<T, HandleT>::size() const
{
    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        return m_rxmesh->get_num_vertices();
    }

    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        return m_rxmesh->get_num_edges();
    }

    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        return m_rxmesh->get_num_faces();
    }
}

template <class T, typename HandleT>
template <int Order>
std::shared_ptr<DenseMatrix<T, Order>> Attribute<T, HandleT>::to_matrix() const
{
    std::shared_ptr<DenseMatrix<T, Order>> mat =
        std::make_shared<DenseMatrix<T, Order>>(
            *m_rxmesh, rows(), cols(), LOCATION_ALL);

    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        m_rxmesh->for_each_vertex(HOST, [&](const VertexHandle vh) {
            for (uint32_t j = 0; j < cols(); ++j) {
                (*mat)(vh, j) = this->operator()(vh, j);
            }
        });
    }

    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        m_rxmesh->for_each_edge(HOST, [&](const EdgeHandle eh) {
            for (uint32_t j = 0; j < cols(); ++j) {
                (*mat)(eh, j) = this->operator()(eh, j);
            }
        });
    }

    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        m_rxmesh->for_each_face(HOST, [&](const FaceHandle fh) {
            for (uint32_t j = 0; j < cols(); ++j) {
                (*mat)(fh, j) = this->operator()(fh, j);
            }
        });
    }

    mat->move(HOST, DEVICE);

    return mat;
}

template <class T, typename HandleT>
template <int Order>
void Attribute<T, HandleT>::from_matrix(DenseMatrix<T, Order>* mat)
{
    assert(mat->rows() == rows());
    assert(mat->cols() == cols());

    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        m_rxmesh->for_each_vertex(HOST, [&](const VertexHandle vh) {
            for (uint32_t j = 0; j < cols(); ++j) {
                this->operator()(vh, j) = (*mat)(vh, j);
            }
        });
    }

    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        m_rxmesh->for_each_edge(HOST, [&](const EdgeHandle eh) {
            for (uint32_t j = 0; j < cols(); ++j) {
                this->operator()(eh, j) = (*mat)(eh, j);
            }
        });
    }

    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        m_rxmesh->for_each_face(HOST, [&](const FaceHandle fh) {
            for (uint32_t j = 0; j < cols(); ++j) {
                this->operator()(fh, j) = (*mat)(fh, j);
            }
        });
    }
}


template <class T, typename HandleT>
__host__ __device__ __forceinline__ uint32_t
Attribute<T, HandleT>::size(const uint32_t p) const
{
#ifdef __CUDA_ARCH__
    return m_d_patches_info[p].get_num_elements<HandleT>()[0];
#else
    return m_h_patches_info[p].get_num_elements<HandleT>()[0];
#endif
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ uint32_t
Attribute<T, HandleT>::capacity(const uint32_t p) const
{
#ifdef __CUDA_ARCH__
    return m_d_patches_info[p].get_capacity<HandleT>();
#else
    return m_h_patches_info[p].get_capacity<HandleT>();
#endif
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ const PatchInfo&
Attribute<T, HandleT>::get_patch_info(const uint32_t p) const
{
#ifdef __CUDA_ARCH__
    return m_d_patches_info[p];
#else
    return m_h_patches_info[p];
#endif
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ uint32_t
Attribute<T, HandleT>::pitch_x() const
{
    return (m_layout == AoS) ? m_num_attributes : 1;
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ uint32_t
Attribute<T, HandleT>::pitch_y(const uint32_t p) const
{
    return (m_layout == AoS) ? 1 : capacity(p);
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ uint32_t
Attribute<T, HandleT>::get_num_attributes() const
{
    return this->m_num_attributes;
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ locationT
Attribute<T, HandleT>::get_allocated() const
{
    return this->m_allocated;
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ layoutT
Attribute<T, HandleT>::get_layout() const
{
    return this->m_layout;
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ bool
Attribute<T, HandleT>::is_device_allocated() const
{
    return ((m_allocated & DEVICE) == DEVICE);
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ bool
Attribute<T, HandleT>::is_host_allocated() const
{
    return ((m_allocated & HOST) == HOST);
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ T& Attribute<T, HandleT>::operator()(
    const uint32_t p_id,
    const uint16_t local_id,
    const uint32_t attr) const
{
    assert(p_id < m_max_num_patches);
    assert(attr < m_num_attributes);

#ifdef __CUDA_ARCH__
    assert(local_id < capacity(p_id));
    return m_d_attr[p_id][local_id * pitch_x() + attr * pitch_y(p_id)];
#else
    assert(local_id < size(p_id));
    return m_h_attr[p_id][local_id * pitch_x() + attr * pitch_y(p_id)];
#endif
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ T& Attribute<T, HandleT>::operator()(
    const uint32_t p_id,
    const uint16_t local_id,
    const uint32_t attr)
{
    assert(p_id < m_max_num_patches);
    assert(attr < m_num_attributes);

#ifdef __CUDA_ARCH__
    assert(local_id < capacity(p_id));
    return m_d_attr[p_id][local_id * pitch_x() + attr * pitch_y(p_id)];
#else
    assert(local_id < size(p_id));
    return m_h_attr[p_id][local_id * pitch_x() + attr * pitch_y(p_id)];
#endif
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ T& Attribute<T, HandleT>::operator()(
    const HandleT  handle,
    const uint32_t attr) const
{
    auto pl = handle.unpack();
    return this->operator()(pl.first, pl.second, attr);
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ T& Attribute<T, HandleT>::operator()(
    const HandleT  handle,
    const uint32_t attr)
{
    auto pl = handle.unpack();
    return this->operator()(pl.first, pl.second, attr);
}

template <class T, typename HandleT>
__host__ __device__ __forceinline__ bool Attribute<T, HandleT>::is_empty() const
{
    return m_max_num_patches == 0;
}


template <class T, typename HandleT>
const char* Attribute<T, HandleT>::get_name() const
{
    return m_name;
}

template <class T, typename HandleT>
double Attribute<T, HandleT>::get_memory_mg() const
{
    return m_memory_mega_bytes;
}

template <class T, typename HandleT>
void Attribute<T, HandleT>::reset(const T      value,
                                  locationT    location,
                                  cudaStream_t stream)
{
    if (((location & DEVICE) == DEVICE) && is_device_allocated()) {
        const int threads = 256;
        detail::memset_attribute<T, HandleT>
            <<<m_rxmesh->get_num_patches(), threads, 0, stream>>>(
                *this, value, m_rxmesh->get_num_patches(), m_num_attributes);
    }

    if (((location & HOST) == HOST) && is_host_allocated()) {
#pragma omp parallel for
        for (int p = 0; p < static_cast<int>(m_rxmesh->get_num_patches());
             ++p) {
            for (uint32_t e = 0; e < capacity(p); ++e) {
                m_h_attr[p][e] = value;
            }
        }
    }
}

template <class T, typename HandleT>
void Attribute<T, HandleT>::move(locationT    source,
                                 locationT    target,
                                 cudaStream_t stream)
{
    if (source == target) {
        RXMESH_WARN(
            "Attribute::move() source ({}) and target ({}) "
            "are the same.",
            location_to_string(source),
            location_to_string(target));
        return;
    }

    if ((source == HOST || source == DEVICE) &&
        ((source & m_allocated) != source)) {
        RXMESH_ERROR(
            "Attribute::move() moving source is not valid"
            " because it was not allocated on source i.e., {}",
            location_to_string(source));
    }

    if (((target & HOST) == HOST || (target & DEVICE) == DEVICE) &&
        ((target & m_allocated) != target)) {
        RXMESH_WARN("Attribute::move() allocating target before moving to {}",
                    location_to_string(target));
        allocate(target);
    }

    if (m_rxmesh->get_num_patches() == 0) {
        return;
    }

    if (source == HOST && target == DEVICE) {
        for (uint32_t p = 0; p < m_rxmesh->get_num_patches(); ++p) {
            CUDA_ERROR(
                cudaMemcpyAsync(m_h_ptr_on_device[p],
                                m_h_attr[p],
                                sizeof(T) * capacity(p) * m_num_attributes,
                                cudaMemcpyHostToDevice,
                                stream));
        }
    } else if (source == DEVICE && target == HOST) {
        for (uint32_t p = 0; p < m_rxmesh->get_num_patches(); ++p) {
            CUDA_ERROR(
                cudaMemcpyAsync(m_h_attr[p],
                                m_h_ptr_on_device[p],
                                sizeof(T) * capacity(p) * m_num_attributes,
                                cudaMemcpyDeviceToHost,
                                stream));
        }
    }
}

template <class T, typename HandleT>
void Attribute<T, HandleT>::release(locationT location)
{
    if (((location & HOST) == HOST) && is_host_allocated()) {
        for (uint32_t p = 0; p < m_rxmesh->get_max_num_patches(); ++p) {
            free(m_h_attr[p]);
        }
        free(m_h_attr);
        m_h_attr    = nullptr;
        m_allocated = m_allocated & (~HOST);
    }

    if (((location & DEVICE) == DEVICE) && is_device_allocated()) {
        for (uint32_t p = 0; p < m_rxmesh->get_max_num_patches(); ++p) {
            GPU_FREE(m_h_ptr_on_device[p]);
        }
        GPU_FREE(m_d_attr);
        m_allocated = m_allocated & (~DEVICE);
    }
}

template <class T, typename HandleT>
void Attribute<T, HandleT>::copy_from(Attribute<T, HandleT>& source,
                                      locationT              source_flag,
                                      locationT              dst_flag,
                                      cudaStream_t           stream)
{
    if (source.m_layout != m_layout) {
        RXMESH_ERROR(
            "Attribute::copy_from() does not support copy from "
            "source of different layout!");
    }

    if ((source_flag & LOCATION_ALL) == LOCATION_ALL &&
        (dst_flag & LOCATION_ALL) != LOCATION_ALL) {
        RXMESH_ERROR("Attribute::copy_from() Invalid configuration!");
        return;
    }

    if (m_num_attributes != source.get_num_attributes()) {
        RXMESH_ERROR(
            "Attribute::copy_from() number of attributes is "
            "different!");
        return;
    }

    if (this->is_empty() || m_rxmesh->get_num_patches() == 0) {
        return;
    }

    if ((source_flag & HOST) == HOST && (dst_flag & HOST) == HOST) {
        if ((source.m_allocated & HOST) != HOST) {
            RXMESH_ERROR(
                "Attribute::copy_from() copying source is not valid"
                " because it was not allocated on host");
            return;
        }
        if ((m_allocated & HOST) != HOST) {
            RXMESH_ERROR(
                "Attribute::copy_from() copying source is not valid"
                " because location (this) was not allocated on host");
            return;
        }

        for (uint32_t p = 0; p < m_rxmesh->get_num_patches(); ++p) {
            std::memcpy(m_h_attr[p],
                        source.m_h_attr[p],
                        sizeof(T) * capacity(p) * m_num_attributes);
        }
    }

    if ((source_flag & DEVICE) == DEVICE && (dst_flag & DEVICE) == DEVICE) {
        if ((source.m_allocated & DEVICE) != DEVICE) {
            RXMESH_ERROR(
                "Attribute::copy_from() copying source is not valid"
                " because it was not allocated on device");
            return;
        }
        if ((m_allocated & DEVICE) != DEVICE) {
            RXMESH_ERROR(
                "Attribute::copy_from() copying source is not valid"
                " because location (this) was not allocated on device");
            return;
        }

        for (uint32_t p = 0; p < m_rxmesh->get_num_patches(); ++p) {
            CUDA_ERROR(
                cudaMemcpyAsync(m_h_ptr_on_device[p],
                                source.m_h_ptr_on_device[p],
                                sizeof(T) * capacity(p) * m_num_attributes,
                                cudaMemcpyDeviceToDevice,
                                stream));
        }
    }

    if ((source_flag & DEVICE) == DEVICE && (dst_flag & HOST) == HOST) {
        if ((source.m_allocated & DEVICE) != DEVICE) {
            RXMESH_ERROR(
                "Attribute::copy_from() copying source is not valid"
                " because it was not allocated on host");
            return;
        }
        if ((m_allocated & HOST) != HOST) {
            RXMESH_ERROR(
                "Attribute::copy_from() copying source is not valid"
                " because location (this) was not allocated on device");
            return;
        }

        for (uint32_t p = 0; p < m_rxmesh->get_num_patches(); ++p) {
            CUDA_ERROR(
                cudaMemcpyAsync(m_h_attr[p],
                                source.m_h_ptr_on_device[p],
                                sizeof(T) * capacity(p) * m_num_attributes,
                                cudaMemcpyDeviceToHost,
                                stream));
        }
    }

    if ((source_flag & HOST) == HOST && (dst_flag & DEVICE) == DEVICE) {
        if ((source.m_allocated & HOST) != HOST) {
            RXMESH_ERROR(
                "Attribute::copy_from() copying source is not valid"
                " because it was not allocated on device");
            return;
        }
        if ((m_allocated & DEVICE) != DEVICE) {
            RXMESH_ERROR(
                "Attribute::copy_from() copying source is not valid"
                " because location (this) was not allocated on host");
            return;
        }

        for (uint32_t p = 0; p < m_rxmesh->get_num_patches(); ++p) {
            CUDA_ERROR(
                cudaMemcpyAsync(m_h_ptr_on_device[p],
                                source.m_h_attr[p],
                                sizeof(T) * capacity(p) * m_num_attributes,
                                cudaMemcpyHostToDevice,
                                stream));
        }
    }
}

template <class T, typename HandleT>
template <int N>
__host__ __device__ __inline__ vec<T, N> Attribute<T, HandleT>::to_glm(
    const HandleT& handle) const
{
    assert(N <= get_num_attributes());

    vec<T, N> ret;

    for (int i = 0; i < N; ++i) {
        ret[i] = this->operator()(handle, i);
    }
    return ret;
}

template <class T, typename HandleT>
template <int N>
__host__ __device__ __inline__ void Attribute<T, HandleT>::from_glm(
    const HandleT&   handle,
    const vec<T, N>& in)
{
    assert(N <= get_num_attributes());

    for (int i = 0; i < N; ++i) {
        this->operator()(handle, i) = in[i];
    }
}

template <class T, typename HandleT>
template <int N>
__host__ __device__ __inline__ Eigen::Matrix<T, N, 1>
         Attribute<T, HandleT>::to_eigen(const HandleT& handle) const
{
    assert(N <= get_num_attributes());

    Eigen::Matrix<T, N, 1> ret;

    for (Eigen::Index i = 0; i < N; ++i) {
        ret[i] = this->operator()(handle, i);
    }
    return ret;
}

template <class T, typename HandleT>
template <int N>
__host__ __device__ __inline__ void Attribute<T, HandleT>::from_eigen(
    const HandleT&                handle,
    const Eigen::Matrix<T, N, 1>& in)
{
    assert(N <= get_num_attributes());

    for (Eigen::Index i = 0; i < N; ++i) {
        this->operator()(handle, i) = in[i];
    }
}


template <class T, typename HandleT>
void Attribute<T, HandleT>::allocate(locationT location)
{
    if (m_max_num_patches != 0) {
        if ((location & HOST) == HOST) {
            release(HOST);

            m_h_attr = static_cast<T**>(malloc(sizeof(T*) * m_max_num_patches));

            for (uint32_t p = 0; p < m_max_num_patches; ++p) {
                m_h_attr[p] = static_cast<T*>(
                    malloc(sizeof(T) * capacity(p) * m_num_attributes));
            }

            m_allocated = m_allocated | HOST;
        }

        if ((location & DEVICE) == DEVICE) {
            release(DEVICE);

            CUDA_ERROR(cudaMalloc((void**)&(m_d_attr),
                                  sizeof(T*) * m_max_num_patches));
            m_memory_mega_bytes +=
                BYTES_TO_MEGABYTES(sizeof(T*) * m_max_num_patches);

            m_h_ptr_on_device =
                static_cast<T**>(malloc(sizeof(T*) * m_max_num_patches));

            for (uint32_t p = 0; p < m_max_num_patches; ++p) {
                CUDA_ERROR(
                    cudaMalloc((void**)&(m_h_ptr_on_device[p]),
                               sizeof(T) * capacity(p) * m_num_attributes));

                m_memory_mega_bytes += BYTES_TO_MEGABYTES(
                    sizeof(T) * capacity(p) * m_num_attributes);
            }
            CUDA_ERROR(cudaMemcpy(m_d_attr,
                                  m_h_ptr_on_device,
                                  sizeof(T*) * m_max_num_patches,
                                  cudaMemcpyHostToDevice));
            m_allocated = m_allocated | DEVICE;
        }
    }
}


size_t AttributeContainer::size()
{
    return m_attr_container.size();
}

std::vector<std::string> AttributeContainer::get_attribute_names() const
{
    std::vector<std::string> names;
    for (size_t i = 0; i < m_attr_container.size(); ++i) {
        names.push_back(m_attr_container[i]->get_name());
    }
    return names;
}

template <typename AttrT>
std::shared_ptr<AttrT> AttributeContainer::add(const char*   name,
                                               uint32_t      num_attributes,
                                               locationT     location,
                                               layoutT       layout,
                                               RXMeshStatic* rxmesh)
{
    if (does_exist(name)) {
        RXMESH_WARN(
            "AttributeContainer::add() adding an attribute with "
            "name {} already exists!",
            std::string(name));
    }

    auto new_attr =
        std::make_shared<AttrT>(name, num_attributes, location, layout, rxmesh);
    m_attr_container.push_back(
        std::dynamic_pointer_cast<AttributeBase>(new_attr));

    return new_attr;
}

bool AttributeContainer::does_exist(const char* name)
{
    for (size_t i = 0; i < m_attr_container.size(); ++i) {
        if (!strcmp(m_attr_container[i]->get_name(), name)) {
            return true;
        }
    }
    return false;
}

void AttributeContainer::remove(const char* name)
{
    for (auto it = m_attr_container.begin(); it != m_attr_container.end();
         ++it) {

        if (!strcmp((*it)->get_name(), name)) {
            (*it)->release(LOCATION_ALL);
            m_attr_container.erase(it);
            break;
        }
    }
}


// Explicit instantiations

#define RXMESH_ATTRIBUTE_INST(T, HandleT) template class Attribute<T, HandleT>;

#define RXMESH_ATTRIBUTE_INST_ALL_HANDLES(T) \
    RXMESH_ATTRIBUTE_INST(T, VertexHandle)   \
    RXMESH_ATTRIBUTE_INST(T, EdgeHandle)     \
    RXMESH_ATTRIBUTE_INST(T, FaceHandle)

#define RXMESH_ATTR_CONTAINER_ADD_INST(AttrT)                       \
    template std::shared_ptr<AttrT> AttributeContainer::add<AttrT>( \
        const char*, uint32_t, locationT, layoutT, RXMeshStatic*);

#define RXMESH_ATTR_CONTAINER_ADD_INST_ALL_HANDLES(T)  \
    RXMESH_ATTR_CONTAINER_ADD_INST(VertexAttribute<T>) \
    RXMESH_ATTR_CONTAINER_ADD_INST(EdgeAttribute<T>)   \
    RXMESH_ATTR_CONTAINER_ADD_INST(FaceAttribute<T>)

#define RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(T) \
    RXMESH_ATTRIBUTE_INST_ALL_HANDLES(T)                   \
    RXMESH_ATTR_CONTAINER_ADD_INST_ALL_HANDLES(T)

RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(Eigen::Matrix3f)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(Eigen::Matrix2f)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(Eigen::Matrix3d)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(Eigen::Matrix2d)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(thrust::complex<float>)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(thrust::complex<double>)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(bool)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(unsigned short)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(char)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(int8_t)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(uint8_t)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(float)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(double)
//RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(int)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(uint32_t)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(int32_t)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(uint64_t)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(int64_t)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(VertexHandle)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(EdgeHandle)
RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES(FaceHandle)

#undef RXMESH_ATTRIBUTE_AND_CONTAINER_INST_ALL_HANDLES
#undef RXMESH_ATTR_CONTAINER_ADD_INST_ALL_HANDLES
#undef RXMESH_ATTR_CONTAINER_ADD_INST
#undef RXMESH_ATTRIBUTE_INST_ALL_HANDLES
#undef RXMESH_ATTRIBUTE_INST

#define RXMESH_ATTR_TM_INST(T, H)                                      \
    template std::shared_ptr<DenseMatrix<T, Eigen::ColMajor>>          \
    Attribute<T, H>::to_matrix<Eigen::ColMajor>() const;               \
    template std::shared_ptr<DenseMatrix<T, Eigen::RowMajor>>          \
                  Attribute<T, H>::to_matrix<Eigen::RowMajor>() const; \
    template void Attribute<T, H>::from_matrix<Eigen::ColMajor>(       \
        DenseMatrix<T, Eigen::ColMajor>*);                             \
    template void Attribute<T, H>::from_matrix<Eigen::RowMajor>(       \
        DenseMatrix<T, Eigen::RowMajor>*);

#define RXMESH_ATTR_TM_INST_ALL(T)       \
    RXMESH_ATTR_TM_INST(T, VertexHandle) \
    RXMESH_ATTR_TM_INST(T, EdgeHandle)   \
    RXMESH_ATTR_TM_INST(T, FaceHandle)

RXMESH_ATTR_TM_INST_ALL(float)
RXMESH_ATTR_TM_INST_ALL(double)
RXMESH_ATTR_TM_INST_ALL(int)
RXMESH_ATTR_TM_INST_ALL(uint32_t)
RXMESH_ATTR_TM_INST_ALL(char)

#undef RXMESH_ATTR_TM_INST_ALL
#undef RXMESH_ATTR_TM_INST

#define RXMESH_ATTR_GLM_EIGEN_INST(T, H, N)                                      \
    template vec<T, N> Attribute<T, H>::to_glm<N>(const H&) const;               \
    template void      Attribute<T, H>::from_glm<N>(const H&, const vec<T, N>&); \
    template Eigen::Matrix<T, N, 1> Attribute<T, H>::to_eigen<N>(const H&)       \
        const;                                                                   \
    template void Attribute<T, H>::from_eigen<N>(                                \
        const H&, const Eigen::Matrix<T, N, 1>&);

// GLM only fully defines vec<N,T> for N=1,2,3,4; N=6,8,12,16 are incomplete.
// Eigen supports arbitrary N. Only instantiate to_glm/from_glm for N=1..4.
#define RXMESH_ATTR_GLM_EIGEN_INST_N(T, H) \
    RXMESH_ATTR_GLM_EIGEN_INST(T, H, 1)    \
    RXMESH_ATTR_GLM_EIGEN_INST(T, H, 2)    \
    RXMESH_ATTR_GLM_EIGEN_INST(T, H, 3)    \
    RXMESH_ATTR_GLM_EIGEN_INST(T, H, 4)

#define RXMESH_ATTR_GLM_EIGEN_INST_ALL(T)         \
    RXMESH_ATTR_GLM_EIGEN_INST_N(T, VertexHandle) \
    RXMESH_ATTR_GLM_EIGEN_INST_N(T, EdgeHandle)   \
    RXMESH_ATTR_GLM_EIGEN_INST_N(T, FaceHandle)

RXMESH_ATTR_GLM_EIGEN_INST_ALL(char)
RXMESH_ATTR_GLM_EIGEN_INST_ALL(int8_t)
RXMESH_ATTR_GLM_EIGEN_INST_ALL(uint8_t)
RXMESH_ATTR_GLM_EIGEN_INST_ALL(float)
RXMESH_ATTR_GLM_EIGEN_INST_ALL(double)
//RXMESH_ATTR_GLM_EIGEN_INST_ALL(int)
RXMESH_ATTR_GLM_EIGEN_INST_ALL(uint32_t)
RXMESH_ATTR_GLM_EIGEN_INST_ALL(int32_t)
RXMESH_ATTR_GLM_EIGEN_INST_ALL(uint64_t)
RXMESH_ATTR_GLM_EIGEN_INST_ALL(int64_t)

#undef RXMESH_ATTR_GLM_EIGEN_INST_ALL
#undef RXMESH_ATTR_GLM_EIGEN_INST_N
#undef RXMESH_ATTR_GLM_EIGEN_INST

}  // namespace rxmesh
