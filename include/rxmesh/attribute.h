#pragma once

#include <assert.h>
#include <utility>

#include "rxmesh/handle.h"
#include "rxmesh/kernels/attribute.cuh"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/types.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/util.h"
#include "rxmesh/util/vector.h"

class RXMeshTest;


namespace rxmesh {


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
 * add_X_attributes() from RXMeshStatic.
 * @tparam T type of the attribute
 */
template <class T>
class Attribute : public AttributeBase
{
    template <typename S>
    friend class ReduceHandle;

   public:
    Attribute()
        : AttributeBase(),
          m_name(nullptr),
          m_num_attributes(0),
          m_allocated(LOCATION_NONE),
          m_h_attr(nullptr),
          m_h_ptr_on_device(nullptr),
          m_d_attr(nullptr),
          m_num_patches(0),
          m_d_element_per_patch(nullptr),
          m_h_element_per_patch(nullptr),
          m_layout(AoS)
    {

        this->m_name    = (char*)malloc(sizeof(char) * 1);
        this->m_name[0] = '\0';
    }

    Attribute(const char* name)
        : AttributeBase(),
          m_name(nullptr),
          m_num_attributes(0),
          m_allocated(LOCATION_NONE),
          m_h_attr(nullptr),
          m_h_ptr_on_device(nullptr),
          m_d_attr(nullptr),
          m_num_patches(0),
          m_d_element_per_patch(nullptr),
          m_h_element_per_patch(nullptr),
          m_layout(AoS)
    {
        if (name != nullptr) {
            this->m_name = (char*)malloc(sizeof(char) * (strlen(name) + 1));
            strcpy(this->m_name, name);
        }
    }

    Attribute(const Attribute& rhs) = default;

    virtual ~Attribute() = default;


    const char* get_name() const
    {
        return m_name;
    }

    __host__ __device__ __forceinline__ uint32_t get_num_attributes() const
    {
        return this->m_num_attributes;
    }

    __host__ __device__ __forceinline__ locationT get_allocated() const
    {
        return this->m_allocated;
    }

    __host__ __device__ __forceinline__ bool is_device_allocated() const
    {
        return ((m_allocated & DEVICE) == DEVICE);
    }

    __host__ __device__ __forceinline__ bool is_host_allocated() const
    {
        return ((m_allocated & HOST) == HOST);
    }


    void reset(const T value, locationT location, cudaStream_t stream = NULL)
    {
        if ((location & DEVICE) == DEVICE) {

            assert((m_allocated & DEVICE) == DEVICE);

            const int threads = 256;
            detail::template memset_attribute<T>
                <<<m_num_patches, threads, 0, stream>>>(*this,
                                                        value,
                                                        m_d_element_per_patch,
                                                        m_num_patches,
                                                        m_num_attributes);
        }


        if ((location & HOST) == HOST) {
            assert((m_allocated & HOST) == HOST);
#pragma omp parallel for
            for (int p = 0; p < static_cast<int>(m_num_patches); ++p) {
                for (int e = 0; e < m_h_element_per_patch[p]; ++e) {
                    m_h_attr[p][e] = value;
                }
            }
        }
    }


    void init(const std::vector<uint16_t>& element_per_patch,
              const uint32_t               num_attributes,
              locationT                    location = LOCATION_ALL,
              const layoutT                layout   = AoS)
    {
        release();
        m_num_patches    = element_per_patch.size();
        m_num_attributes = num_attributes;
        m_layout         = layout;

        if (m_num_patches == 0) {
            return;
        }

        allocate(element_per_patch.data(), location);
    }


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
            allocate(m_h_element_per_patch, target);
        }

        if (this->m_num_patches == 0) {
            return;
        }

        if (source == HOST && target == DEVICE) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_ptr_on_device[p],
                    m_h_attr[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyHostToDevice,
                    stream));
            }
        } else if (source == DEVICE && target == HOST) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_attr[p],
                    m_h_ptr_on_device[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyDeviceToHost,
                    stream));
            }
        }
    }


    void release(locationT location = LOCATION_ALL)
    {
        if (((location & HOST) == HOST) && ((m_allocated & HOST) == HOST)) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                free(m_h_attr[p]);
            }
            free(m_h_attr);
            m_h_attr = nullptr;
            free(m_h_element_per_patch);
            m_h_element_per_patch = nullptr;
            m_allocated           = m_allocated & (~HOST);
        }

        if (((location & DEVICE) == DEVICE) &&
            ((m_allocated & DEVICE) == DEVICE)) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                GPU_FREE(m_h_ptr_on_device[p]);
            }
            GPU_FREE(m_d_attr);
            GPU_FREE(m_d_element_per_patch);
            m_allocated = m_allocated & (~DEVICE);
        }
    }


    void copy_from(Attribute<T>& source,
                   locationT     source_flag,
                   locationT     location_flag,
                   cudaStream_t  stream = NULL)
    {
        // Deep copy from source. The source_flag defines where we will copy
        // from. The location_flag defines where we will copy to.

        // if source_flag and location_flag are both set to LOCATION_ALL, then
        // we copy what is on host to host, and what on location to location

        // If sourc_flag is set to HOST (or DEVICE) and location_flag is set to
        // LOCATION_ALL, then we copy source's HOST (or DEVICE) to both HOST
        // and DEVICE in location

        // Setting source_flag to LOCATION_ALL while location_flag is Not set to
        // LOCATION_ALL is invalid because we don't know which source to copy
        // from

        if (source.m_layout != m_layout) {
            RXMESH_ERROR(
                "Attribute::copy_from() does not support copy from "
                "source of different layout!");
        }

        if ((source_flag & LOCATION_ALL) == LOCATION_ALL &&
            (location_flag & LOCATION_ALL) != LOCATION_ALL) {
            RXMESH_ERROR("Attribute::copy_from() Invalid configuration!");
        }

        if (m_num_attributes != source.get_num_attributes()) {
            RXMESH_ERROR(
                "Attribute::copy_from() number of attributes is "
                "different!");
        }

        if (this->is_empty() || this->m_num_patches == 0) {
            return;
        }

        // 1) copy from HOST to HOST
        if ((source_flag & HOST) == HOST && (location_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "Attribute::copy() copying source is not valid"
                    " because it was not allocated on host");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "Attribute::copy() copying source is not valid"
                    " because location (this) was not allocated on host");
            }

            for (uint32_t p = 0; p < m_num_patches; ++p) {
                assert(m_h_element_per_patch[p] ==
                       source.m_h_element_per_patch[p]);
                std::memcpy(
                    m_h_ptr_on_device[p],
                    source.m_h_ptr_on_device[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes);
            }
        }


        // 2) copy from DEVICE to DEVICE
        if ((source_flag & DEVICE) == DEVICE &&
            (location_flag & DEVICE) == DEVICE) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "Attribute::copy() copying source is not valid"
                    " because it was not allocated on device");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "Attribute::copy() copying source is not valid"
                    " because location (this) was not allocated on device");
            }

            for (uint32_t p = 0; p < m_num_patches; ++p) {
                assert(m_h_element_per_patch[p] ==
                       source.m_h_element_per_patch[p]);
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_ptr_on_device[p],
                    source.m_h_ptr_on_device[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
        }


        // 3) copy from DEVICE to HOST
        if ((source_flag & DEVICE) == DEVICE &&
            (location_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "Attribute::copy() copying source is not valid"
                    " because it was not allocated on host");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "Attribute::copy() copying source is not valid"
                    " because location (this) was not allocated on device");
            }


            for (uint32_t p = 0; p < m_num_patches; ++p) {
                assert(m_h_element_per_patch[p] ==
                       source.m_h_element_per_patch[p]);
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_attr[p],
                    source.m_h_ptr_on_device[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyDeviceToHost,
                    stream));
            }
        }


        // 4) copy from HOST to DEVICE
        if ((source_flag & HOST) == HOST &&
            (location_flag & DEVICE) == DEVICE) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "Attribute::copy() copying source is not valid"
                    " because it was not allocated on device");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "Attribute::copy() copying source is not valid"
                    " because location (this) was not allocated on host");
            }


            for (uint32_t p = 0; p < m_num_patches; ++p) {
                assert(m_h_element_per_patch[p] ==
                       source.m_h_element_per_patch[p]);
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_ptr_on_device[p],
                    source.m_h_attr[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyHostToDevice,
                    stream));
            }
        }
    }

    __host__ __device__ __forceinline__ T& operator()(const uint32_t patch_id,
                                                      const uint16_t local_id,
                                                      const uint32_t attr) const
    {
        assert(patch_id < m_num_patches);
        assert(attr < m_num_attributes);

        const uint32_t pitch_x = (m_layout == AoS) ? m_num_attributes : 1;
#ifdef __CUDA_ARCH__
        const uint32_t pitch_y =
            (m_layout == AoS) ? 1 : m_d_element_per_patch[patch_id];
        return m_d_attr[patch_id][local_id * pitch_x + attr * pitch_y];
#else
        const uint32_t pitch_y =
            (m_layout == AoS) ? 1 : m_h_element_per_patch[patch_id];
        return m_h_attr[patch_id][local_id * pitch_x + attr * pitch_y];
#endif
    }

    __host__ __device__ __forceinline__ T& operator()(const uint32_t patch_id,
                                                      const uint16_t local_id,
                                                      const uint32_t attr)
    {
        assert(patch_id < m_num_patches);
        assert(attr < m_num_attributes);

        const uint32_t pitch_x = (m_layout == AoS) ? m_num_attributes : 1;
#ifdef __CUDA_ARCH__
        const uint32_t pitch_y =
            (m_layout == AoS) ? 1 : m_d_element_per_patch[patch_id];
        return m_d_attr[patch_id][local_id * pitch_x + attr * pitch_y];
#else
        const uint32_t pitch_y =
            (m_layout == AoS) ? 1 : m_h_element_per_patch[patch_id];
        return m_h_attr[patch_id][local_id * pitch_x + attr * pitch_y];
#endif
    }


    __host__ __device__ __forceinline__ bool is_empty() const
    {
        return m_num_patches == 0;
    }


   private:
    void allocate(const uint16_t* element_per_patch, locationT location)
    {

        if (m_num_patches != 0) {

            if ((location & HOST) == HOST) {
                release(HOST);
                m_h_element_per_patch = static_cast<uint16_t*>(
                    malloc(sizeof(uint16_t) * m_num_patches));

                m_h_attr = static_cast<T**>(malloc(sizeof(T*) * m_num_patches));

                std::memcpy(m_h_element_per_patch,
                            element_per_patch,
                            sizeof(uint16_t) * m_num_patches);

                for (uint32_t p = 0; p < m_num_patches; ++p) {
                    m_h_attr[p] = static_cast<T*>(malloc(
                        sizeof(T) * element_per_patch[p] * m_num_attributes));
                }

                m_allocated = m_allocated | HOST;
            }

            if ((location & DEVICE) == DEVICE) {
                release(DEVICE);

                m_h_element_per_patch = static_cast<uint16_t*>(
                    malloc(sizeof(uint16_t) * m_num_patches));

                std::memcpy(m_h_element_per_patch,
                            element_per_patch,
                            sizeof(uint16_t) * m_num_patches);

                CUDA_ERROR(cudaMalloc((void**)&(m_d_element_per_patch),
                                      sizeof(uint16_t) * m_num_patches));


                CUDA_ERROR(cudaMalloc((void**)&(m_d_attr),
                                      sizeof(T*) * m_num_patches));
                m_h_ptr_on_device =
                    static_cast<T**>(malloc(sizeof(T*) * m_num_patches));

                CUDA_ERROR(cudaMemcpy(m_d_element_per_patch,
                                      element_per_patch,
                                      sizeof(uint16_t) * m_num_patches,
                                      cudaMemcpyHostToDevice));

                for (uint32_t p = 0; p < m_num_patches; ++p) {
                    CUDA_ERROR(cudaMalloc((void**)&(m_h_ptr_on_device[p]),
                                          sizeof(T) * m_h_element_per_patch[p] *
                                              m_num_attributes));
                }
                CUDA_ERROR(cudaMemcpy(m_d_attr,
                                      m_h_ptr_on_device,
                                      sizeof(T*) * m_num_patches,
                                      cudaMemcpyHostToDevice));
                m_allocated = m_allocated | DEVICE;
            }
        }
    }


    char*     m_name;
    uint32_t  m_num_attributes;
    locationT m_allocated;
    T**       m_h_attr;
    T**       m_h_ptr_on_device;
    T**       m_d_attr;
    uint32_t  m_num_patches;
    uint16_t* m_d_element_per_patch;
    uint16_t* m_h_element_per_patch;
    layoutT   m_layout;

    constexpr static uint32_t m_block_size = 256;
};

/**
 * @brief Attributes for faces
 * @tparam T the attribute type
 */
template <class T>
class FaceAttribute : public Attribute<T>
{
   public:
    FaceAttribute() = default;

    FaceAttribute(const char*                  name,
                  const std::vector<uint16_t>& face_per_patch,
                  const uint32_t               num_attributes,
                  locationT                    location,
                  const layoutT                layout,
                  const PatchInfo*             h_patches_info,
                  const PatchInfo*             d_patches_info)
        : Attribute<T>(name),
          m_h_patches_info(h_patches_info),
          m_d_patches_info(d_patches_info)
    {
        this->init(face_per_patch, num_attributes, location, layout);
    }

    __host__ __device__ __forceinline__ T& operator()(const FaceHandle f_handle,
                                                      const uint32_t attr) const
    {

#ifdef __CUDA_ARCH__
        auto pl = f_handle.unpack(m_d_patches_info);
#else
        auto pl = f_handle.unpack(m_h_patches_info);
#endif
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }

    __host__ __device__ __forceinline__ T& operator()(
        const FaceHandle f_handle) const
    {
        return (*this)(f_handle, 0);
    }

    __host__ __device__ __forceinline__ T& operator()(const FaceHandle f_handle,
                                                      const uint32_t   attr)
    {

#ifdef __CUDA_ARCH__
        auto pl = f_handle.unpack(m_d_patches_info);
#else
        auto pl = f_handle.unpack(m_h_patches_info);
#endif
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }

    __host__ __device__ __forceinline__ T& operator()(const FaceHandle f_handle)
    {
        return (*this)(f_handle, 0);
    }

   private:
    const PatchInfo* m_h_patches_info;
    const PatchInfo* m_d_patches_info;
};


/**
 * @brief Attributes for edges
 * @tparam T the attribute type
 */
template <class T>
class EdgeAttribute : public Attribute<T>
{
   public:
    EdgeAttribute() = default;

    EdgeAttribute(const char*                  name,
                  const std::vector<uint16_t>& edge_per_patch,
                  const uint32_t               num_attributes,
                  locationT                    location,
                  const layoutT                layout,
                  const PatchInfo*             h_patches_info,
                  const PatchInfo*             d_patches_info)
        : Attribute<T>(name),
          m_h_patches_info(h_patches_info),
          m_d_patches_info(d_patches_info)
    {
        this->init(edge_per_patch, num_attributes, location, layout);
    }

    __host__ __device__ __forceinline__ T& operator()(const EdgeHandle e_handle,
                                                      const uint32_t attr) const
    {
#ifdef __CUDA_ARCH__
        auto pl = e_handle.unpack(m_d_patches_info);
#else
        auto pl = e_handle.unpack(m_h_patches_info);
#endif
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }

    __host__ __device__ __forceinline__ T& operator()(
        const EdgeHandle e_handle) const
    {
        return (*this)(e_handle, 0);
    }

    __host__ __device__ __forceinline__ T& operator()(const EdgeHandle e_handle,
                                                      const uint32_t   attr)
    {
#ifdef __CUDA_ARCH__
        auto pl = e_handle.unpack(m_d_patches_info);
#else
        auto pl = e_handle.unpack(m_h_patches_info);
#endif
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }

    __host__ __device__ __forceinline__ T& operator()(const EdgeHandle e_handle)
    {
        return (*this)(e_handle, 0);
    }

   private:
    const PatchInfo* m_h_patches_info;
    const PatchInfo* m_d_patches_info;
};


/**
 * @brief Attributes for vertices
 * @tparam T the attribute type
 */
template <class T>
class VertexAttribute : public Attribute<T>
{
   public:
    VertexAttribute() = default;

    VertexAttribute(const char*                  name,
                    const std::vector<uint16_t>& vertex_per_patch,
                    const uint32_t               num_attributes,
                    locationT                    location,
                    const layoutT                layout,
                    const PatchInfo*             h_patches_info,
                    const PatchInfo*             d_patches_info)
        : Attribute<T>(name),
          m_h_patches_info(h_patches_info),
          m_d_patches_info(d_patches_info)
    {
        this->init(vertex_per_patch, num_attributes, location, layout);
    }


    __host__ __device__ __forceinline__ T& operator()(
        const VertexHandle v_handle,
        const uint32_t     attr) const
    {
#ifdef __CUDA_ARCH__
        auto pl = v_handle.unpack(m_d_patches_info);
#else
        auto pl = v_handle.unpack(m_h_patches_info);
#endif
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }

    __host__ __device__ __forceinline__ T& operator()(
        const VertexHandle v_handle) const
    {
        return (*this)(v_handle, 0);
    }

    __host__ __device__ __forceinline__ T& operator()(
        const VertexHandle v_handle,
        const uint32_t     attr)
    {
#ifdef __CUDA_ARCH__
        auto pl = v_handle.unpack(m_d_patches_info);
#else
        auto pl = v_handle.unpack(m_h_patches_info);
#endif
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }

    __host__ __device__ __forceinline__ T& operator()(
        const VertexHandle v_handle)
    {
        return (*this)(v_handle, 0);
    }

   private:
    const PatchInfo* m_h_patches_info;
    const PatchInfo* m_d_patches_info;
};

/**
 * @brief Attribute container used to managing attributes from RXMeshStatic
 */
class AttributeContainer
{
   public:
    AttributeContainer() = default;

    virtual ~AttributeContainer()
    {
        while (!m_attr_container.empty()) {
            m_attr_container.back()->release();
            m_attr_container.pop_back();
        }
    }

    size_t size()
    {
        return m_attr_container.size();
    }

    std::vector<std::string> get_attribute_names() const
    {
        std::vector<std::string> names;
        for (size_t i = 0; i < m_attr_container.size(); ++i) {
            names.push_back(m_attr_container[i]->get_name());
        }
        return names;
    }

    template <typename AttrT>
    std::shared_ptr<AttrT> add(const char*            name,
                               std::vector<uint16_t>& element_per_patch,
                               uint32_t               num_attributes,
                               locationT              location,
                               layoutT                layout,
                               const PatchInfo*       h_patches_info,
                               const PatchInfo*       d_patches_info)
    {
        if (does_exist(name)) {
            RXMESH_WARN(
                "AttributeContainer::add() adding an attribute with "
                "name {} already exists!",
                std::string(name));
        }

        auto new_attr = std::make_shared<AttrT>(name,
                                                element_per_patch,
                                                num_attributes,
                                                location,
                                                layout,
                                                h_patches_info,
                                                d_patches_info);
        m_attr_container.push_back(
            std::dynamic_pointer_cast<AttributeBase>(new_attr));

        return new_attr;
    }

    bool does_exist(const char* name)
    {
        for (size_t i = 0; i < m_attr_container.size(); ++i) {
            if (!strcmp(m_attr_container[i]->get_name(), name)) {
                return true;
            }
        }
        return false;
    }

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