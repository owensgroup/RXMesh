#pragma once

#include <assert.h>
#include <utility>

#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/rxmesh_attribute.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/rxmesh_types.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/util.h"
#include "rxmesh/util/vector.h"

class RXMeshTest;


namespace rxmesh {

// Flags for where the attributes array resides
using locationT = uint32_t;
enum : locationT
{
    LOCATION_NONE = 0x00,
    HOST          = 0x01,
    DEVICE        = 0x02,
    LOCATION_ALL  = 0x0F,
};

// The memory layout
using layoutT = uint32_t;
enum : layoutT
{
    AoS = 0x00,
    SoA = 0x01,
};

static std::string location_to_string(locationT location)
{
    std::string str = "";
    if ((location & HOST) == HOST) {
        str = (str == "" ? "" : " ") + std::string("HOST");
    }
    if ((location & DEVICE) == DEVICE) {
        str = (str == "" ? "" : " ") + std::string("DEVICE");
    }
    if (str == "") {
        str = "NONE";
    }
    return str;
}


/**
 * @brief Base untyped attributes used as an interface for attribute container
 */
class RXMeshAttributeBase
{
    // our friend tester class
    friend class ::RXMeshTest;

   public:
    RXMeshAttributeBase() = default;

    // virtual RXMeshAttributeBase* clone() const = 0;

    virtual const char* get_name() const = 0;

    virtual void release(locationT target) = 0;

    virtual void release_v1(locationT location = LOCATION_ALL) = 0;

    virtual ~RXMeshAttributeBase() = default;
};

/**
 * @brief  Here we manage the attributes on top of the mesh. An attributes is
 * attached to mesh element (e.g., vertices, edges, or faces).
 * largely inspired by
 * https://github.com/gunrock/gunrock/blob/master/gunrock/util/array_utils.cuh
 * It is discouraged to use RXMeshAttribute directly in favor of using
 * add_X_attributes() from RXMeshStatic.
 * @tparam T type of the attribute
 */
template <class T>
class RXMeshAttribute : public RXMeshAttributeBase
{
    template <typename S>
    friend class ReduceHandle;

   public:
    RXMeshAttribute()
        : RXMeshAttributeBase(),
          m_name(nullptr),
          m_num_mesh_elements(0),
          m_num_attributes(0),
          m_allocated(LOCATION_NONE),
          m_h_attr(nullptr),
          m_d_attr(nullptr),
          m_h_attr_v1(nullptr),
          m_d_attr_v1(nullptr),
          m_h_ptr_on_device(nullptr),
          m_num_patches(0),
          m_d_element_per_patch(nullptr),
          m_h_element_per_patch(nullptr),
          m_layout(AoS)
    {

        this->m_name    = (char*)malloc(sizeof(char) * 1);
        this->m_name[0] = '\0';
    }

    RXMeshAttribute(const char* name)
        : m_name(nullptr),
          m_num_mesh_elements(0),
          m_num_attributes(0),
          m_allocated(LOCATION_NONE),
          m_h_attr(nullptr),
          m_d_attr(nullptr),
          m_h_attr_v1(nullptr),
          m_d_attr_v1(nullptr),
          m_h_ptr_on_device(nullptr),
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

    RXMeshAttribute(const RXMeshAttribute& rhs) = default;

    virtual ~RXMeshAttribute() = default;


    const char* get_name() const
    {
        return m_name;
    }

    __host__ __device__ __forceinline__ uint32_t get_num_mesh_elements() const
    {
        return this->m_num_mesh_elements;
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

            const int      threads = 256;
            const uint32_t total   = m_num_attributes * m_num_mesh_elements;
            memset<T><<<(total + threads - 1) / threads, threads, 0, stream>>>(
                m_d_attr, value, total);
            CUDA_ERROR(cudaDeviceSynchronize());
            CUDA_ERROR(cudaGetLastError());
        }


        if ((location & HOST) == HOST) {
            assert((m_allocated & HOST) == HOST);
            for (uint32_t i = 0; i < m_num_mesh_elements * m_num_attributes;
                 ++i) {
                m_h_attr[i] = value;
            }
        }
    }

    void reset_v1(const T value, locationT location, cudaStream_t stream = NULL)
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
                    m_h_attr_v1[p][e] = value;
                }
            }
        }
    }

    void init(uint32_t   num_elements,
              uint32_t   num_attributes,
              locationT  location        = DEVICE,
              layoutT    layout          = AoS,
              const bool with_axpy_alloc = true)
    {
        release();
        m_allocated               = LOCATION_NONE;
        this->m_num_mesh_elements = num_elements;
        this->m_num_attributes    = num_attributes;
        if (num_elements == 0) {
            return;
        }
        allocate(location);
        m_layout = layout;

        // if (!m_is_reduce_allocated && with_reduce_alloc) {
        //    allocate_reduce_temp_storage(0);
        //}
    }

    void init_v1(const std::vector<uint16_t>& element_per_patch,
                 const uint32_t               num_attributes,
                 locationT                    location = LOCATION_ALL,
                 const layoutT                layout   = AoS)
    {
        release_v1();
        m_num_patches    = element_per_patch.size();
        m_num_attributes = num_attributes;
        m_layout         = layout;

        if (m_num_patches == 0) {
            return;
        }

        allocate_v1(element_per_patch.data(), location);
    }

    void move(locationT source, locationT target)
    {
        if (source == target) {
            return;
        }

        if ((source == HOST || source == DEVICE) &&
            ((source & m_allocated) != source)) {
            RXMESH_ERROR(
                "RXMeshAttribute::move() moving source is not valid"
                " because it was not allocated on source");
        }

        if (((target & HOST) == HOST || (target & DEVICE) == DEVICE) &&
            ((target & m_allocated) != target)) {
            RXMESH_WARN(
                "RXMeshAttribute::move() allocating target before moving to {}",
                location_to_string(target));
            allocate(target);
        }

        if (this->m_num_mesh_elements == 0) {
            return;
        }

        if (source == HOST && target == DEVICE) {
            CUDA_ERROR(
                cudaMemcpy(m_d_attr,
                           m_h_attr,
                           sizeof(T) * m_num_mesh_elements * m_num_attributes,
                           cudaMemcpyHostToDevice));

        } else if (source == DEVICE && target == HOST) {
            CUDA_ERROR(
                cudaMemcpy(m_h_attr,
                           m_d_attr,
                           sizeof(T) * m_num_mesh_elements * m_num_attributes,
                           cudaMemcpyDeviceToHost));
        }
    }

    void move_v1(locationT source, locationT target, cudaStream_t stream = NULL)
    {
        if (source == target) {
            RXMESH_WARN(
                "RXMeshAttribute::move_v1() source ({}) and target ({}) "
                "are the same.",
                location_to_string(source),
                location_to_string(target));
            return;
        }

        if ((source == HOST || source == DEVICE) &&
            ((source & m_allocated) != source)) {
            RXMESH_ERROR(
                "RXMeshAttribute::move_v1() moving source is not valid"
                " because it was not allocated on source i.e., {}",
                location_to_string(source));
        }

        if (((target & HOST) == HOST || (target & DEVICE) == DEVICE) &&
            ((target & m_allocated) != target)) {
            RXMESH_WARN(
                "RXMeshAttribute::move() allocating target before moving to {}",
                location_to_string(target));
            allocate_v1(m_h_element_per_patch, target);
        }

        if (this->m_num_patches == 0) {
            return;
        }

        if (source == HOST && target == DEVICE) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_ptr_on_device[p],
                    m_h_attr_v1[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyHostToDevice,
                    stream));
            }
        } else if (source == DEVICE && target == HOST) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_attr_v1[p],
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
            free(m_h_attr);
            m_h_attr    = nullptr;
            m_allocated = m_allocated & (~HOST);
        }

        if (((location & DEVICE) == DEVICE) &&
            ((m_allocated & DEVICE) == DEVICE)) {
            GPU_FREE(m_d_attr);
            m_allocated = m_allocated & (~DEVICE);
        }

        /*if (location == LOCATION_ALL || m_allocated == 0) {
            if (m_is_reduce_allocated) {
                for (uint32_t i = 0; i < m_num_attributes; ++i) {
                    GPU_FREE(m_d_reduce_temp_storage[i]);
                    GPU_FREE(m_norm2_temp_buffer[i]);
                    GPU_FREE(m_d_reduce_output[i]);
                    CUDA_ERROR(cudaStreamDestroy(m_reduce_streams[i]));
                }
                m_is_reduce_allocated = false;
                free(m_reduce_streams);
                free(m_d_reduce_output);
                free(m_norm2_temp_buffer);
                free(m_d_reduce_temp_storage);
            }
        }*/
    }

    void release_v1(locationT location = LOCATION_ALL)
    {
        if (((location & HOST) == HOST) && ((m_allocated & HOST) == HOST)) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                free(m_h_attr_v1[p]);
            }
            free(m_h_attr_v1);
            m_h_attr_v1 = nullptr;
            free(m_h_element_per_patch);
            m_h_element_per_patch = nullptr;
            m_allocated           = m_allocated & (~HOST);
        }

        if (((location & DEVICE) == DEVICE) &&
            ((m_allocated & DEVICE) == DEVICE)) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                GPU_FREE(m_h_ptr_on_device[p]);
            }
            GPU_FREE(m_d_attr_v1);
            GPU_FREE(m_d_element_per_patch);
            m_allocated = m_allocated & (~DEVICE);
        }
    }

    void copy(RXMeshAttribute<T>& source,
              locationT           source_flag,
              locationT           location_flag)
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
                "RXMeshAttribute::copy() does not support copy from source of "
                "different layout!");
        }

        if ((source_flag & LOCATION_ALL) == LOCATION_ALL &&
            (location_flag & LOCATION_ALL) != LOCATION_ALL) {
            RXMESH_ERROR("RXMeshAttribute::copy() Invalid configuration!");
        }

        if (source.get_num_mesh_elements() != m_num_mesh_elements) {
            RXMESH_ERROR(
                "RXMeshAttribute::copy() source has different size than "
                "location!");
        }

        // 1) copy from HOST to HOST
        if ((source_flag & HOST) == HOST && (location_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on host");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because location (this) was not allocated on host");
            }

            std::memcpy((void*)m_h_attr,
                        source.m_h_attr,
                        m_num_mesh_elements * m_num_attributes * sizeof(T));
        }


        // 2) copy from DEVICE to DEVICE
        if ((source_flag & DEVICE) == DEVICE &&
            (location_flag & DEVICE) == DEVICE) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on device");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because location (this) was not allocated on device");
            }

            CUDA_ERROR(
                cudaMemcpy(m_d_attr,
                           source.m_d_attr,
                           m_num_mesh_elements * m_num_attributes * sizeof(T),
                           cudaMemcpyDeviceToDevice));
        }


        // 3) copy from DEVICE to HOST
        if ((source_flag & DEVICE) == DEVICE &&
            (location_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on host");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because location (this) was not allocated on device");
            }

            CUDA_ERROR(
                cudaMemcpy(m_h_attr,
                           source.m_d_attr,
                           m_num_mesh_elements * m_num_attributes * sizeof(T),
                           cudaMemcpyDeviceToHost));
        }


        // 4) copy from HOST to DEVICE
        if ((source_flag & HOST) == HOST &&
            (location_flag & DEVICE) == DEVICE) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on device");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because location (this) was not allocated on host");
            }

            CUDA_ERROR(
                cudaMemcpy(m_d_attr,
                           source.m_h_attr,
                           m_num_mesh_elements * m_num_attributes * sizeof(T),
                           cudaMemcpyHostToDevice));
        }
    }


    void copy_from(RXMeshAttribute<T>& source,
                   locationT           source_flag,
                   locationT           location_flag,
                   cudaStream_t        stream = NULL)
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
                "RXMeshAttribute::copy_v1() does not support copy from source "
                "of "
                "different layout!");
        }

        if ((source_flag & LOCATION_ALL) == LOCATION_ALL &&
            (location_flag & LOCATION_ALL) != LOCATION_ALL) {
            RXMESH_ERROR("RXMeshAttribute::copy() Invalid configuration!");
        }

        if (source.get_num_mesh_elements() != m_num_mesh_elements) {
            RXMESH_ERROR(
                "RXMeshAttribute::copy() source has different size than "
                "location!");
        }

        if (this->is_empty() || this->m_num_patches == 0) {
            return;
        }

        // 1) copy from HOST to HOST
        if ((source_flag & HOST) == HOST && (location_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on host");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because location (this) was not allocated on host");
            }

            for (uint32_t p = 0; p < m_num_patches; ++p) {
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
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on device");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because location (this) was not allocated on device");
            }

            for (uint32_t p = 0; p < m_num_patches; ++p) {
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
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on host");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because location (this) was not allocated on device");
            }


            for (uint32_t p = 0; p < m_num_patches; ++p) {
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_attr_v1[p],
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
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on device");
            }
            if ((location_flag & m_allocated) != location_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because location (this) was not allocated on host");
            }


            for (uint32_t p = 0; p < m_num_patches; ++p) {
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_ptr_on_device[p],
                    source.m_h_attr_v1[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyHostToDevice,
                    stream));
            }
        }
    }

    // TODO remove
    __host__ __device__ __forceinline__ T& operator()(const uint32_t idx,
                                                      const uint32_t attr)
    {

        assert(attr < m_num_attributes);
        assert(idx < m_num_mesh_elements);

        const uint32_t pitch_x = (m_layout == AoS) ? m_num_attributes : 1;
        const uint32_t pitch_y = (m_layout == AoS) ? 1 : m_num_mesh_elements;

#ifdef __CUDA_ARCH__
        return m_d_attr[idx * pitch_x + attr * pitch_y];

#else
        return m_h_attr[idx * pitch_x + attr * pitch_y];
#endif
    }

    // TODO remove
    __host__ __device__ __forceinline__ T& operator()(const uint32_t idx)
    {
        // for m_num_attributes =1
        assert(m_num_attributes == 1);
        assert(idx < m_num_mesh_elements);
        return (*this)(idx, 0);
    }

    // TODO remove
    __host__ __device__ __forceinline__ T& operator()(const uint32_t idx,
                                                      const uint32_t attr) const
    {

        assert(attr < m_num_attributes);
        assert(idx < m_num_mesh_elements);

        const uint32_t pitch_x = (m_layout == AoS) ? m_num_attributes : 1;
        const uint32_t pitch_y = (m_layout == AoS) ? 1 : m_num_mesh_elements;

#ifdef __CUDA_ARCH__
        return m_d_attr[idx * pitch_x + attr * pitch_y];

#else
        return m_h_attr[idx * pitch_x + attr * pitch_y];
#endif
    }

    // TODO remove
    __host__ __device__ __forceinline__ T& operator()(const uint32_t idx) const
    {
        // for m_num_attributes =1
        assert(m_num_attributes == 1);
        assert(idx < m_num_mesh_elements);
        return (*this)(idx, 0);
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
        return m_d_attr_v1[patch_id][local_id * pitch_x + attr * pitch_y];
#else
        const uint32_t pitch_y =
            (m_layout == AoS) ? 1 : m_h_element_per_patch[patch_id];
        return m_h_attr_v1[patch_id][local_id * pitch_x + attr * pitch_y];
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
        return m_d_attr_v1[patch_id][local_id * pitch_x + attr * pitch_y];
#else
        const uint32_t pitch_y =
            (m_layout == AoS) ? 1 : m_h_element_per_patch[patch_id];
        return m_h_attr_v1[patch_id][local_id * pitch_x + attr * pitch_y];
#endif
    }


    __host__ __device__ __forceinline__ bool is_empty() const
    {
        return m_num_patches == 0;
    }


   private:
    void allocate(locationT location)
    {

        if ((location & HOST) == HOST) {
            release(HOST);
            if (m_num_mesh_elements != 0) {
                m_h_attr = (T*)malloc(sizeof(T) * m_num_mesh_elements *
                                      m_num_attributes);
                if (!m_h_attr) {
                    RXMESH_ERROR(
                        " RXMeshAttribute::allocate() allocation on {} failed "
                        "with #mesh_elemnts = {} and #attributes per element = "
                        "{}" +
                            location_to_string(HOST),
                        m_num_mesh_elements,
                        m_num_attributes);
                }
            }
            m_allocated = m_allocated | HOST;
        }


        if ((location & DEVICE) == DEVICE) {
            release(DEVICE);
            if (m_num_mesh_elements != 0) {
                CUDA_ERROR(cudaMalloc(
                    (void**)&(m_d_attr),
                    sizeof(T) * m_num_mesh_elements * m_num_attributes));
            }
            m_allocated = m_allocated | DEVICE;
        }
    }

    void allocate_v1(const uint16_t* element_per_patch, locationT location)
    {

        if (m_num_patches != 0) {

            if ((location & HOST) == HOST) {
                release_v1(HOST);
                m_h_element_per_patch = static_cast<uint16_t*>(
                    malloc(sizeof(uint16_t) * m_num_patches));

                m_h_attr_v1 =
                    static_cast<T**>(malloc(sizeof(T*) * m_num_patches));

                std::memcpy(m_h_element_per_patch,
                            element_per_patch,
                            sizeof(uint16_t) * m_num_patches);

                for (uint32_t p = 0; p < m_num_patches; ++p) {
                    m_h_attr_v1[p] = static_cast<T*>(malloc(
                        sizeof(T) * element_per_patch[p] * m_num_attributes));
                }

                m_allocated = m_allocated | HOST;
            }

            if ((location & DEVICE) == DEVICE) {
                release_v1(DEVICE);

                m_h_element_per_patch = static_cast<uint16_t*>(
                    malloc(sizeof(uint16_t) * m_num_patches));

                std::memcpy(m_h_element_per_patch,
                            element_per_patch,
                            sizeof(uint16_t) * m_num_patches);

                CUDA_ERROR(cudaMalloc((void**)&(m_d_element_per_patch),
                                      sizeof(uint16_t) * m_num_patches));


                CUDA_ERROR(cudaMalloc((void**)&(m_d_attr_v1),
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
                CUDA_ERROR(cudaMemcpy(m_d_attr_v1,
                                      m_h_ptr_on_device,
                                      sizeof(T*) * m_num_patches,
                                      cudaMemcpyHostToDevice));
                m_allocated = m_allocated | DEVICE;
            }
        }
    }


    char*     m_name;
    uint32_t  m_num_mesh_elements;
    uint32_t  m_num_attributes;
    locationT m_allocated;
    T*        m_h_attr;
    T*        m_d_attr;
    T**       m_h_attr_v1;
    T**       m_h_ptr_on_device;
    T**       m_d_attr_v1;
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
class RXMeshFaceAttribute : public RXMeshAttribute<T>
{
   public:
    RXMeshFaceAttribute() = default;

    RXMeshFaceAttribute(const char*                  name,
                        const std::vector<uint16_t>& face_per_patch,
                        const uint32_t               num_attributes,
                        locationT                    location,
                        const layoutT                layout,
                        const PatchInfo*             h_patches_info,
                        const PatchInfo*             d_patches_info)
        : RXMeshAttribute<T>(name),
          m_h_patches_info(h_patches_info),
          m_d_patches_info(d_patches_info)
    {
        this->init_v1(face_per_patch, num_attributes, location, layout);
    }

    __host__ __device__ __forceinline__ T& operator()(const FaceHandle f_handle,
                                                      const uint32_t attr) const
    {
#ifdef __CUDA_ARCH__
        auto pl = m_d_patches_info[f_handle.m_patch_id].get_patch_and_local_id(
            f_handle);
#else
        auto pl = m_h_patches_info[f_handle.m_patch_id].get_patch_and_local_id(
            f_handle);
#endif
        return RXMeshAttribute<T>::operator()(pl.first, pl.second, attr);
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
        auto pl = m_d_patches_info[f_handle.m_patch_id].get_patch_and_local_id(
            f_handle);
#else
        auto pl = m_h_patches_info[f_handle.m_patch_id].get_patch_and_local_id(
            f_handle);
#endif
        return RXMeshAttribute<T>::operator()(pl.first, pl.second, attr);
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
class RXMeshEdgeAttribute : public RXMeshAttribute<T>
{
   public:
    RXMeshEdgeAttribute() = default;

    RXMeshEdgeAttribute(const char*                  name,
                        const std::vector<uint16_t>& edge_per_patch,
                        const uint32_t               num_attributes,
                        locationT                    location,
                        const layoutT                layout,
                        const PatchInfo*             h_patches_info,
                        const PatchInfo*             d_patches_info)
        : RXMeshAttribute<T>(name),
          m_h_patches_info(h_patches_info),
          m_d_patches_info(d_patches_info)
    {
        this->init_v1(edge_per_patch, num_attributes, location, layout);
    }

    __host__ __device__ __forceinline__ T& operator()(const EdgeHandle e_handle,
                                                      const uint32_t attr) const
    {
#ifdef __CUDA_ARCH__
        auto pl = m_d_patches_info[e_handle.m_patch_id].get_patch_and_local_id(
            e_handle);
#else
        auto pl = m_h_patches_info[e_handle.m_patch_id].get_patch_and_local_id(
            e_handle);
#endif
        return RXMeshAttribute<T>::operator()(pl.first, pl.second, attr);
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
        auto pl = m_d_patches_info[e_handle.m_patch_id].get_patch_and_local_id(
            e_handle);
#else
        auto pl = m_h_patches_info[e_handle.m_patch_id].get_patch_and_local_id(
            e_handle);
#endif
        return RXMeshAttribute<T>::operator()(pl.first, pl.second, attr);
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
class RXMeshVertexAttribute : public RXMeshAttribute<T>
{
   public:
    RXMeshVertexAttribute() = default;

    RXMeshVertexAttribute(const char*                  name,
                          const std::vector<uint16_t>& vertex_per_patch,
                          const uint32_t               num_attributes,
                          locationT                    location,
                          const layoutT                layout,
                          const PatchInfo*             h_patches_info,
                          const PatchInfo*             d_patches_info)
        : RXMeshAttribute<T>(name),
          m_h_patches_info(h_patches_info),
          m_d_patches_info(d_patches_info)
    {
        this->init_v1(vertex_per_patch, num_attributes, location, layout);
    }

    // TODO remove
    __host__ __device__ __forceinline__ T& operator()(const uint32_t idx,
                                                      const uint32_t attr) const
    {
        return (*this)(idx, attr);
    }

    // TODO remove
    __host__ __device__ __forceinline__ T& operator()(const uint32_t idx,
                                                      const uint32_t attr)
    {
        return (*this)(idx, attr);
    }

    __host__ __device__ __forceinline__ T& operator()(
        const VertexHandle v_handle,
        const uint32_t     attr) const
    {
#ifdef __CUDA_ARCH__
        auto pl = m_d_patches_info[v_handle.m_patch_id].get_patch_and_local_id(
            v_handle);
#else
        auto pl = m_h_patches_info[v_handle.m_patch_id].get_patch_and_local_id(
            v_handle);
#endif
        return RXMeshAttribute<T>::operator()(pl.first, pl.second, attr);
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
        auto pl = m_d_patches_info[v_handle.m_patch_id].get_patch_and_local_id(
            v_handle);
#else
        auto pl = m_h_patches_info[v_handle.m_patch_id].get_patch_and_local_id(
            v_handle);
#endif
        return RXMeshAttribute<T>::operator()(pl.first, pl.second, attr);
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
class RXMeshAttributeContainer
{
   public:
    RXMeshAttributeContainer() = default;

    virtual ~RXMeshAttributeContainer()
    {
        while (!m_attr_container.empty()) {
            m_attr_container.back()->release_v1();
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
                "RXMeshAttributeContainer::add() adding an attribute with "
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
            std::dynamic_pointer_cast<RXMeshAttributeBase>(new_attr));

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
    std::vector<std::shared_ptr<RXMeshAttributeBase>> m_attr_container;
};

}  // namespace rxmesh