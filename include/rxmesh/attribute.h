#pragma once

#include <assert.h>
#include <utility>

#include "rxmesh/handle.h"
#include "rxmesh/kernels/attribute.cuh"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/types.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/util.h"

#include "rxmesh/matrix/dense_matrix.cuh"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include <Eigen/Dense>

class RXMeshTest;


namespace rxmesh {

class RXMeshStatic;

/**
 * @brief Base untyped attributes used as an interface for attribute container
 */
class AttributeBase
{
    // our friend tester class
    friend class ::RXMeshTest;

   public:
    AttributeBase() = default;

    virtual const char* get_name() const = 0;

    virtual void release(locationT location = LOCATION_ALL) = 0;

    virtual ~AttributeBase() = default;
};

/**
 * @brief  Here we manage the attributes on top of the mesh. An attributes is
 * attached to mesh element (e.g., vertices, edges, or faces).
 * largely inspired by
 * https://github.com/gunrock/gunrock/blob/master/gunrock/util/array_utils.cuh
 * It is discouraged to use Attribute directly in favor of using
 * add_X_attributes() from RXMeshStatic where X is vertex, edge, or face. This
 * way, the user does not have to specify the number of mesh elements or
 * deallocate/release the Attribute (attribute garbage collection is managed by
 * RXMeshStatic)
 * @tparam T type of the attribute
 */
template <class T, typename HandleT>
class Attribute : public AttributeBase
{
    template <typename S, typename H>
    friend class ReduceHandle;

   public:
    using HandleType = HandleT;
    using Type       = T;

    /**
     * @brief Default constructor which initializes all pointers to nullptr
     */
    Attribute()
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

    /**
     * @brief Main constructor to be used by RXMeshStatic not directly by the
     * user
     * @param name of the attribute
     * @param element_per_patch number of elements owned per patch
     * @param num_attributes number of attribute per face
     * @param location where the attribute to be allocated
     * @param layout memory layout in case of num_attributes>1
     */
    explicit Attribute(const char*    name,
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


    T& operator()(size_t i, size_t j = 0)
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

    T& operator()(size_t i, size_t j = 0) const
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

    size_t rows() const
    {
        return size();
    }

    size_t cols() const
    {
        return this->get_num_attributes();
    }

    uint32_t size() const
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

    /**
     * @brief convert the attributes stored into a dense matrix where number of
     * rows represent the number of mesh elements of this attribute and number
     * of columns is the number of attributes
     */
    std::shared_ptr<DenseMatrix<T>> to_matrix() const
    {
        std::shared_ptr<DenseMatrix<T>> mat =
            std::make_shared<DenseMatrix<T>>(*m_rxmesh, rows(), cols());

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

    /**
     * @brief copy a dense matrix to this attribute. The copying happens on the
     * host side, i.e., we copy the content of mat which is on the host to this
     * attribute on the host side
     * @param mat
     */
    void from_matrix(DenseMatrix<T>* mat)
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


    /**
     * @brief get the number of elements in a patch. The element type
     * corresponds to the template HandleT
     * @param p the patch id
     */
    __host__ __device__ __forceinline__ uint32_t size(const uint32_t p) const
    {
#ifdef __CUDA_ARCH__
        return m_d_patches_info[p].get_num_elements<HandleT>()[0];
#else
        return m_h_patches_info[p].get_num_elements<HandleT>()[0];
#endif
    }

    /**
     * @brief get maximum number of elements in a patch. The element type
     * corresponds to the template HandleT
     * @param p the patch id
     */
    __host__ __device__ __forceinline__ uint32_t
    capacity(const uint32_t p) const
    {
#ifdef __CUDA_ARCH__
        return m_d_patches_info[p].get_capacity<HandleT>()[0];
#else
        return m_h_patches_info[p].get_capacity<HandleT>()[0];
#endif
    }


    __host__ __device__ __forceinline__ const PatchInfo& get_patch_info(
        const uint32_t p) const
    {
#ifdef __CUDA_ARCH__
        return m_d_patches_info[p];
#else
        return m_h_patches_info[p];
#endif
    }

    __host__ __device__ __forceinline__ uint32_t pitch_x() const
    {

        return (m_layout == AoS) ? m_num_attributes : 1;
    }

    __host__ __device__ __forceinline__ uint32_t pitch_y(const uint32_t p) const
    {

        return (m_layout == AoS) ? 1 : capacity(p);
    }

    Attribute(const Attribute& rhs) = default;

    virtual ~Attribute() = default;

    /**
     * @brief Get the name of the attribute
     */
    const char* get_name() const
    {
        return m_name;
    }

    /**
     * @brief return the amount of allocated memory in megabytes
     */
    double get_memory_mg() const
    {
        return m_memory_mega_bytes;
    }

    /**
     * @brief get the number of attributes per mesh element
     */
    __host__ __device__ __forceinline__ uint32_t get_num_attributes() const
    {
        return this->m_num_attributes;
    }

    /**
     * @brief Flag that indicates where the memory is allocated
     */
    __host__ __device__ __forceinline__ locationT get_allocated() const
    {
        return this->m_allocated;
    }

    /**
     * @brief return the memory layout
     */
    __host__ __device__ __forceinline__ layoutT get_layout() const
    {
        return this->m_layout;
    }

    /**
     * @brief Check if attribute is allocated on device
     */
    __host__ __device__ __forceinline__ bool is_device_allocated() const
    {
        return ((m_allocated & DEVICE) == DEVICE);
    }

    /**
     * @brief Check if attribute is allocated on host
     */
    __host__ __device__ __forceinline__ bool is_host_allocated() const
    {
        return ((m_allocated & HOST) == HOST);
    }

    /**
     * @brief Reset attribute to certain value
     * @param value to be set
     * @param location which location (device, host, or both) where attribute
     * will be set
     * @param stream in case of DEVICE, this is the stream that will be used to
     * launch the reset kernel
     */
    void reset(const T value, locationT location, cudaStream_t stream = NULL)
    {
        if (((location & DEVICE) == DEVICE) && is_device_allocated()) {
            const int threads = 256;
            detail::template memset_attribute<T>
                <<<m_rxmesh->get_num_patches(), threads, 0, stream>>>(
                    *this,
                    value,
                    m_rxmesh->get_num_patches(),
                    m_num_attributes);
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


    /**
     * @brief Copy memory from one location to another. If target is not
     * allocated, it will be allocated first before copying the memory.
     * @param source the source location
     * @param target the destination location
     * @param stream to be used to launch the kernel
     * TODO it is better to launch a kernel that do the memcpy than relying on
     * the host API from CUDA since all these small memcpy will be enqueued in
     * the same stream and so serialized
     */
    void move(locationT source, locationT target, cudaStream_t stream = NULL)
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
            RXMESH_WARN(
                "Attribute::move() allocating target before moving to {}",
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

    /**
     * @brief Release allocated memory in certain location
     * @param location where memory will be released
     */
    void release(locationT location = LOCATION_ALL)
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

    /**
     * @brief Deep copy from a source attribute. If source_flag and dst_flag are
     * both set to LOCATION_ALL, then we copy what is on host to host, and what
     * on device to device. If sourc_flag is set to HOST (or DEVICE) and
     * dst_flag is set to LOCATION_ALL, then we copy source's HOST (or
     * DEVICE) to both HOST and DEVICE. Setting source_flag to
     * LOCATION_ALL while dst_flag is NOT set to LOCATION_ALL is invalid
     * because we don't know which source to copy from
     * @param source attribute to copy from
     * @param source_flag defines where we will copy from
     * @param dst_flag defines where we will copy to
     * @param stream used to launch kernel/memcpy
     */
    void copy_from(Attribute<T, HandleT>& source,
                   locationT              source_flag,
                   locationT              dst_flag,
                   cudaStream_t           stream = NULL)
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

        // 1) copy from HOST to HOST
        if ((source_flag & HOST) == HOST && (dst_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "Attribute::copy_from() copying source is not valid"
                    " because it was not allocated on host");
                return;
            }
            if ((dst_flag & m_allocated) != dst_flag) {
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


        // 2) copy from DEVICE to DEVICE
        if ((source_flag & DEVICE) == DEVICE && (dst_flag & DEVICE) == DEVICE) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "Attribute::copy_from() copying source is not valid"
                    " because it was not allocated on device");
                return;
            }
            if ((dst_flag & m_allocated) != dst_flag) {
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


        // 3) copy from DEVICE to HOST
        if ((source_flag & DEVICE) == DEVICE && (dst_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "Attribute::copy_from() copying source is not valid"
                    " because it was not allocated on host");
                return;
            }
            if ((dst_flag & m_allocated) != dst_flag) {
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


        // 4) copy from HOST to DEVICE
        if ((source_flag & HOST) == HOST && (dst_flag & DEVICE) == DEVICE) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "Attribute::copy_from() copying source is not valid"
                    " because it was not allocated on device");
                return;
            }
            if ((dst_flag & m_allocated) != dst_flag) {
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

    /**
     * @brief Accessing an attribute using a handle to the mesh element
     * @param handle input handle
     * @param attr the attribute id
     * @return const reference to the attribute
     */
    __host__ __device__ __forceinline__ T& operator()(
        const HandleT  handle,
        const uint32_t attr = 0) const
    {
        auto pl = handle.unpack();
        return this->operator()(pl.first, pl.second, attr);
    }


    /**
     * @brief Accessing the attribute a glm vector. This is used for read only
     * since the return result is a copy.
     */
    template <int N>
    __host__ __device__ __inline__ vec<T, N> to_glm(const HandleT& handle) const
    {
        assert(N == get_num_attributes());

        vec<T, N> ret;

        for (int i = 0; i < N; ++i) {
            ret[i] = this->operator()(handle, i);
        }
        return ret;
    }


    /**
     * @brief Accessing the attribute a Eigen matrix. This is used for read only
     * since the return result is a copy.
     */
    template <int N>
    __host__ __device__ __inline__ Eigen::Matrix<T, N, 1> to_eigen(
        const HandleT& handle) const
    {
        assert(N == get_num_attributes());

        Eigen::Matrix<T, N, 1> ret;

        for (Eigen::Index i = 0; i < N; ++i) {
            ret[i] = this->operator()(handle, i);
        }
        return ret;
    }

    /**
     * @brief Accessing an attribute using a handle to the mesh element
     * @param handle input handle
     * @param attr the attribute id
     * @return non-const reference to the attribute
     */
    __host__ __device__ __forceinline__ T& operator()(const HandleT  handle,
                                                      const uint32_t attr = 0)
    {
        auto pl = handle.unpack();
        return this->operator()(pl.first, pl.second, attr);
    }

    /**
     * @brief Access the attribute value using patch and local index in the
     * patch. This is meant to be used by XXAttribute not directly by the user
     * @param p_id patch to be accessed
     * @param local_id the local id in the patch
     * @param attr the attribute id
     * @return const reference to the attribute
     */
    __host__ __device__ __forceinline__ T& operator()(const uint32_t p_id,
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

    /**
     * @brief Access the attribute value using patch and local index in the
     * patch. This is meant to be used by XXAttribute not directly by the user
     * @param p_id patch to be accessed
     * @param local_id the local id in the patch
     * @param attr the attribute id
     * @return non-const reference to the attribute
     */
    __host__ __device__ __forceinline__ T& operator()(const uint32_t p_id,
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

    /**
     * @brief Check if the attribute is empty
     */
    __host__ __device__ __forceinline__ bool is_empty() const
    {
        return m_max_num_patches == 0;
    }


   protected:
    /**
     * @brief allocate internal memory
     */
    void allocate(locationT location)
    {
        if (m_max_num_patches != 0) {

            if ((location & HOST) == HOST) {
                release(HOST);

                m_h_attr =
                    static_cast<T**>(malloc(sizeof(T*) * m_max_num_patches));

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

    RXMeshStatic*    m_rxmesh;
    const PatchInfo* m_h_patches_info;
    const PatchInfo* m_d_patches_info;
    char*            m_name;
    uint32_t         m_num_attributes;
    locationT        m_allocated;
    T**              m_h_attr;
    T**              m_h_ptr_on_device;
    T**              m_d_attr;
    uint32_t         m_max_num_patches;
    layoutT          m_layout;
    double           m_memory_mega_bytes;

    constexpr static uint32_t m_block_size = 256;
};

template <class T>
using VertexAttribute = Attribute<T, VertexHandle>;

template <class T>
using EdgeAttribute = Attribute<T, EdgeHandle>;

template <class T>
using FaceAttribute = Attribute<T, FaceHandle>;

/**
 * @brief Attribute container used to manage a collection of attributes by
 * RXMeshStatic
 */
class AttributeContainer
{
   public:
    /**
     * @brief Default constructor
     */
    AttributeContainer() = default;

    /**
     * @brief Destructor which releases all attribute managed by this container
     */
    virtual ~AttributeContainer()
    {
        while (!m_attr_container.empty()) {
            m_attr_container.back()->release();
            m_attr_container.pop_back();
        }
    }

    /**
     * @brief Number of attribute managed by this container
     */
    size_t size()
    {
        return m_attr_container.size();
    }

    /**
     * @brief get a list of name of the attributes managed by this container
     * @return
     */
    std::vector<std::string> get_attribute_names() const
    {
        std::vector<std::string> names;
        for (size_t i = 0; i < m_attr_container.size(); ++i) {
            names.push_back(m_attr_container[i]->get_name());
        }
        return names;
    }

    /**
     * @brief add a new attribute to be managed by this container
     * @tparam AttrT attribute type
     * @param name unique name given to the attribute
     * @param element_per_patch number of mesh element owned by each patch
     * @param num_attributes number of attributes per mesh element
     * @param location where the attributes will be allocated
     * @param layout memory layout in case of num_attributes > 1
     * @return a shared pointer to the attribute
     */
    template <typename AttrT>
    std::shared_ptr<AttrT> add(const char*   name,
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

        auto new_attr = std::make_shared<AttrT>(
            name, num_attributes, location, layout, rxmesh);
        m_attr_container.push_back(
            std::dynamic_pointer_cast<AttributeBase>(new_attr));

        return new_attr;
    }

    /**
     * @brief Check if an attribute exists
     * @param name of the attribute
     */
    bool does_exist(const char* name)
    {
        for (size_t i = 0; i < m_attr_container.size(); ++i) {
            if (!strcmp(m_attr_container[i]->get_name(), name)) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief remove an attribute and release its memory
     * @param name of the attribute
     */
    void remove(const char* name)
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

   private:
    std::vector<std::shared_ptr<AttributeBase>> m_attr_container;
};

}  // namespace rxmesh