#pragma once

#include <assert.h>
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/rxmesh_attribute.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/rxmesh_types.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/util.h"
#include "rxmesh/util/vector.h"

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

// Reduce ops
using reduceOpT = uint32_t;
enum : reduceOpT
{
    SUM   = 0x00,
    MAX   = 0x01,
    MIN   = 0X02,
    NORM2 = 0X04,  // L2 norm squared
    DOT   = 0x08,  // dot product

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
   public:
    RXMeshAttributeBase() = default;

    // virtual RXMeshAttributeBase* clone() const = 0;

    virtual const char* get_name() const = 0;

    virtual void release(locationT location) = 0;

    virtual void release_v1() = 0;

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
   public:
    RXMeshAttribute()
        : RXMeshAttributeBase(),
          m_name(nullptr),
          m_num_mesh_elements(0),
          m_num_attributes(0),
          m_allocated(LOCATION_NONE),
          m_is_allocated(false),
          m_h_attr(nullptr),
          m_d_attr(nullptr),
          m_attr(nullptr),
          m_num_patches(0),
          m_element_per_patch(nullptr),
          m_layout(AoS),
          d_axpy_alpha(nullptr),
          d_axpy_beta(nullptr),
          m_is_axpy_allocated(false),
          m_is_reduce_allocated(false),
          m_reduce_temp_storage_bytes(0),
          m_d_reduce_temp_storage(nullptr),
          m_d_reduce_output(nullptr),
          m_reduce_streams(nullptr),
          m_norm2_temp_buffer(nullptr)
    {

        this->m_name    = (char*)malloc(sizeof(char) * 1);
        this->m_name[0] = '\0';
    }

    RXMeshAttribute(const char* name)
        : m_name(nullptr),
          m_num_mesh_elements(0),
          m_num_attributes(0),
          m_allocated(LOCATION_NONE),
          m_is_allocated(false),
          m_h_attr(nullptr),
          m_d_attr(nullptr),
          m_attr(nullptr),
          m_num_patches(0),
          m_element_per_patch(nullptr),
          m_layout(AoS),
          d_axpy_alpha(nullptr),
          d_axpy_beta(nullptr),
          m_is_axpy_allocated(false),
          m_is_reduce_allocated(false),
          m_reduce_temp_storage_bytes(0)
    {

        if (name != nullptr) {
            this->m_name = (char*)malloc(sizeof(char) * (strlen(name) + 1));
            strcpy(this->m_name, name);
        }
    }

    RXMeshAttribute(const RXMeshAttribute& rhs) = default;

    virtual ~RXMeshAttribute() = default;

    void set_name(std::string name)
    {
        free(this->m_name);
        this->m_name = (char*)malloc(sizeof(char) * name.length() + 1);
        strcpy(this->m_name, name.c_str());
    }

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

    __host__ __device__ __forceinline__ T* get_pointer(locationT location) const
    {

        if (location == DEVICE) {
            return m_d_attr;
        }
        if (location == HOST) {
            return m_h_attr;
        }
        return nullptr;
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

    void init(uint32_t   num_elements,
              uint32_t   num_attributes,
              locationT  location          = DEVICE,
              layoutT    layout            = AoS,
              const bool with_axpy_alloc   = true,
              const bool with_reduce_alloc = true)
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

        if (!m_is_axpy_allocated && with_axpy_alloc) {
            CUDA_ERROR(cudaMalloc((void**)&d_axpy_alpha,
                                  m_num_attributes * sizeof(T)));
            CUDA_ERROR(
                cudaMalloc((void**)&d_axpy_beta, m_num_attributes * sizeof(T)));
            m_is_axpy_allocated = true;
        }

        if (!m_is_reduce_allocated && with_reduce_alloc) {
            allocate_reduce_temp_storage(0);
        }
    }

    void init_v1(const std::vector<uint16_t>& element_per_patch,
                 const uint32_t               num_attributes,
                 const layoutT                layout            = AoS,
                 const bool                   with_axpy_alloc   = true,
                 const bool                   with_reduce_alloc = true)
    {
        release();
        m_num_patches    = element_per_patch.size();
        m_num_attributes = num_attributes;
        m_layout         = layout;

        if (m_num_patches == 0) {
            return;
        }

        allocate_v1(element_per_patch);

        if (!m_is_axpy_allocated && with_axpy_alloc) {
            CUDA_ERROR(cudaMalloc((void**)&d_axpy_alpha,
                                  m_num_attributes * sizeof(T)));
            CUDA_ERROR(
                cudaMalloc((void**)&d_axpy_beta, m_num_attributes * sizeof(T)));
            m_is_axpy_allocated = true;
        }

        if (!m_is_reduce_allocated && with_reduce_alloc) {
            allocate_reduce_temp_storage(0);
        }
    }

    void move(locationT source, locationT location)
    {
        if (source == location) {
            return;
        }

        if ((source == HOST || source == DEVICE) &&
            ((source & m_allocated) != source)) {
            RXMESH_ERROR(
                "RXMeshAttribute::move() moving source is not valid"
                " because it was not allocated on source");
        }

        if (((location & HOST) == HOST || (location & DEVICE) == DEVICE) &&
            ((location & m_allocated) != location)) {
            allocate(location);
        }

        if (this->m_num_mesh_elements == 0) {
            return;
        }

        if (source == HOST && location == DEVICE) {
            CUDA_ERROR(
                cudaMemcpy(m_d_attr,
                           m_h_attr,
                           sizeof(T) * m_num_mesh_elements * m_num_attributes,
                           cudaMemcpyHostToDevice));

        } else if (source == DEVICE && location == HOST) {
            CUDA_ERROR(
                cudaMemcpy(m_h_attr,
                           m_d_attr,
                           sizeof(T) * m_num_mesh_elements * m_num_attributes,
                           cudaMemcpyDeviceToHost));
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

        if (location == LOCATION_ALL || m_allocated == 0) {
            m_num_mesh_elements = 0;

            if (m_is_axpy_allocated) {
                GPU_FREE(d_axpy_alpha);
                GPU_FREE(d_axpy_beta);
                m_is_axpy_allocated = false;
            }
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
        }
    }

    void release_v1()
    {
        if (m_is_allocated) {
            GPU_FREE(m_d_attr);
            GPU_FREE(m_element_per_patch);

            m_is_allocated = false;

            m_num_mesh_elements = 0;
            m_num_patches       = 0;
            m_num_attributes    = 0;

            if (m_is_axpy_allocated) {
                GPU_FREE(d_axpy_alpha);
                GPU_FREE(d_axpy_beta);
                m_is_axpy_allocated = false;
            }

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

    void change_layout(locationT location)
    {
        // Only supporting HOST location
        // If location is HOST, then the layout change only for the HOST
        // the user then can copy the data to the DEVICE.
        // To change the layout of data in the DEVICE, it should be copied first
        // to the HOST, change layout, and then copy back to the DEVICE

        // Only make sense when number of attributes is >1
        if (m_num_attributes > 1) {

            if ((location & m_allocated) != location) {
                RXMESH_ERROR(
                    "RXMeshAttribute::change_layout() changing layout {} is "
                    "not valid because it was not allocated",
                    location_to_string(location));
                return;
            }

            if ((location & HOST) != HOST) {
                RXMESH_ERROR(
                    "RXMeshAttribute::change_layout() changing layout {} is "
                    "not valid because it is not supported",
                    location_to_string(location));
                return;
            }

            if ((location & HOST) == HOST) {
                const uint32_t size = m_num_mesh_elements * m_num_attributes;
                const uint32_t num_cols =
                    (m_layout == AoS) ? m_num_attributes : m_num_mesh_elements;
                in_place_matrix_transpose(
                    m_h_attr, m_h_attr + size, uint64_t(num_cols));

                m_layout = (m_layout == SoA) ? AoS : SoA;
            }
        }
    }

    template <uint32_t N>
    void axpy(const RXMeshAttribute<T>& X,
              const Vector<N, T>        alpha,
              const Vector<N, T>        beta,
              const locationT           location     = DEVICE,
              const uint32_t            attribute_id = INVALID32,
              cudaStream_t              stream       = NULL)
    {
        // Implements
        // Y = alpha*X + beta*Y
        // where Y is *this.
        // alpha and beta is passed as vector so different values can be applied
        // to each attribute.
        // if attribute == INVALID32, then axpy is applied on all attributes
        // and alpha (and beta) should be of size m_num_attributes.
        // Otherwise axpy will be only applied on the given attribute number
        //(should be less than m_num_attributes) and alpha (and
        // beta) should be of size one
        // location tells on which side (host to device) the operation
        // will run

        const uint32_t num_attribute =
            (attribute_id == INVALID32) ? m_num_attributes : 1;
        assert(N >= num_attribute);

        if ((location & DEVICE) == DEVICE) {

            const uint32_t blocks =
                DIVIDE_UP(m_num_mesh_elements, m_block_size);

            CUDA_ERROR(cudaMemcpyAsync(d_axpy_alpha,
                                       (void*)&alpha,
                                       sizeof(Vector<N, T>),
                                       cudaMemcpyHostToDevice,
                                       stream));
            CUDA_ERROR(cudaMemcpyAsync(d_axpy_beta,
                                       (void*)&beta,
                                       sizeof(Vector<N, T>),
                                       cudaMemcpyHostToDevice,
                                       stream));

            rxmesh_attribute_axpy<T><<<blocks, m_block_size, 0, stream>>>(
                X, d_axpy_alpha, *this, d_axpy_beta, attribute_id);

            cudaStreamSynchronize(stream);
        }
        if ((location & HOST) == HOST) {
            for (uint32_t i = 0; i < m_num_mesh_elements; ++i) {
                for (uint32_t j = 0; j < m_num_attributes; ++j) {
                    (*this)(i, j) =
                        alpha[j] * X(i, j) + beta[j] * (*this)(i, j);
                }
            }
        }
    }

    template <uint32_t N>
    void reduce(Vector<N, T>&             h_output,
                const reduceOpT           op,
                const RXMeshAttribute<T>* other    = nullptr,
                const locationT           location = DEVICE)
    {
        if (N < m_num_attributes) {
            RXMESH_ERROR(
                "RXMeshAttribute::reduce() the output Vector size should be "
                ">= the number of attributes per mesh element. Output "
                "Vector size = {}, number of attributes per mesh element = {}",
                N,
                m_num_attributes);
        }


        if ((location & DEVICE) == DEVICE) {
            if (m_layout != SoA) {
                RXMESH_ERROR(
                    "RXMeshAttribute::reduce is not supported for non SoA "
                    "layouts on the device");
            }
            for (uint32_t i = 0; i < m_num_attributes; ++i) {
                switch (op) {
                    case SUM: {
                        cub::DeviceReduce::Sum(
                            m_d_reduce_temp_storage[i],
                            m_reduce_temp_storage_bytes,
                            m_d_attr + i * m_num_mesh_elements,
                            m_d_reduce_output[i],
                            m_num_mesh_elements,
                            m_reduce_streams[i]);
                        break;
                    }
                    case MAX: {
                        cub::DeviceReduce::Max(
                            m_d_reduce_temp_storage[i],
                            m_reduce_temp_storage_bytes,
                            m_d_attr + i * m_num_mesh_elements,
                            m_d_reduce_output[i],
                            m_num_mesh_elements,
                            m_reduce_streams[i]);
                        break;
                    }
                    case MIN: {
                        cub::DeviceReduce::Min(
                            m_d_reduce_temp_storage[i],
                            m_reduce_temp_storage_bytes,
                            m_d_attr + i * m_num_mesh_elements,
                            m_d_reduce_output[i],
                            m_num_mesh_elements,
                            m_reduce_streams[i]);
                        break;
                    }
                    case NORM2: {
                        uint32_t num_blocks =
                            DIVIDE_UP(m_num_mesh_elements, m_block_size);
                        // 1st pass
                        rxmesh_attribute_norm2<T, m_block_size>
                            <<<num_blocks,
                               m_block_size,
                               0,
                               m_reduce_streams[i]>>>(
                                *this, i, m_norm2_temp_buffer[i]);

                        // 2nd pass
                        cub::DeviceReduce::Sum(m_d_reduce_temp_storage[i],
                                               m_reduce_temp_storage_bytes,
                                               m_norm2_temp_buffer[i],
                                               m_d_reduce_output[i],
                                               num_blocks,
                                               m_reduce_streams[i]);
                        break;
                    }
                    case DOT: {
                        if (other == nullptr) {
                            RXMESH_ERROR(
                                "RXMeshAttribute::reduce other can not be "
                                "nullptr for dot product");
                        }
                        uint32_t num_blocks =
                            DIVIDE_UP(m_num_mesh_elements, m_block_size);
                        // 1st pass
                        rxmesh_attribute_dot<T, m_block_size>
                            <<<num_blocks,
                               m_block_size,
                               0,
                               m_reduce_streams[i]>>>(
                                *this, *other, i, m_norm2_temp_buffer[i]);

                        // 2nd pass
                        cub::DeviceReduce::Sum(m_d_reduce_temp_storage[i],
                                               m_reduce_temp_storage_bytes,
                                               m_norm2_temp_buffer[i],
                                               m_d_reduce_output[i],
                                               num_blocks,
                                               m_reduce_streams[i]);
                        break;
                    }
                    default: {
                        RXMESH_ERROR(
                            "RXMeshAttribute::reduce is not supported for the "
                            "given operation");
                        break;
                    }
                }
                CUDA_ERROR(cudaStreamSynchronize(m_reduce_streams[i]));
                CUDA_ERROR(cudaMemcpy(&h_output[i],
                                      m_d_reduce_output[i],
                                      sizeof(T),
                                      cudaMemcpyDeviceToHost));
            }
        }

        if ((location & HOST) == HOST) {
            for (uint32_t j = 0; j < m_num_attributes; ++j) {
                for (uint32_t i = 0; i < m_num_mesh_elements; ++i) {
                    h_output[i] = 0;
                    if (op == MAX || op == MIN) {
                        h_output[i] = (*this)(i, j);
                    }

                    switch (op) {
                        case SUM: {
                            h_output[i] += (*this)(i, j);
                            break;
                        }
                        case MAX: {
                            h_output[i] = std::max(h_output[i], (*this)(i, j));
                            break;
                        }
                        case MIN: {
                            h_output[i] = std::min(h_output[i], (*this)(i, j));
                            break;
                        }
                        case NORM2: {
                            h_output[i] += (*this)(i, j) * (*this)(i, j);
                            break;
                        }
                        case DOT: {
                            if (other == nullptr) {
                                RXMESH_ERROR(
                                    "RXMeshAttribute::reduce other can not be "
                                    "nullptr for dot product");
                            }
                            h_output[i] += (*this)(i, j) * (*other)(i, j);
                        }
                        default:
                            break;
                    }
                }
            }
        }
    }

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx,
                                                      uint32_t attr)
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

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx)
    {
        // for m_num_attributes =1
        assert(m_num_attributes == 1);
        assert(idx < m_num_mesh_elements);
        return (*this)(idx, 0);
    }

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx,
                                                      uint32_t attr) const
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

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx) const
    {
        // for m_num_attributes =1
        assert(m_num_attributes == 1);
        assert(idx < m_num_mesh_elements);
        return (*this)(idx, 0);
    }


    __host__ __device__ __forceinline__ T* operator->() const
    {
#ifdef __CUDA_ARCH__
        return m_d_attr;
#else
        return m_h_attr;
#endif
    }

    RXMeshAttribute& operator=(const RXMeshAttribute& rhs)
    {
        if (rhs.m_name != nullptr) {
            this->m_name =
                (char*)malloc(sizeof(char) * (strlen(rhs.m_name) + 1));
            strcpy(this->m_name, rhs.m_name);
        }
        m_num_mesh_elements = rhs.m_num_mesh_elements;
        m_num_attributes    = rhs.m_num_attributes;
        m_allocated         = rhs.m_allocated;
        if (rhs.is_device_allocated()) {
            if (rhs.m_num_mesh_elements != 0) {
                uint64_t num_bytes =
                    sizeof(T) * rhs.m_num_mesh_elements * rhs.m_num_attributes;
                CUDA_ERROR(cudaMalloc((void**)&m_d_attr, num_bytes));
                CUDA_ERROR(cudaMemcpy(m_d_attr,
                                      rhs.m_d_attr,
                                      num_bytes,
                                      cudaMemcpyDeviceToDevice));
            }
        }
        if (rhs.is_host_allocated()) {
            if (rhs.m_num_mesh_elements != 0) {
                uint64_t num_bytes =
                    sizeof(T) * rhs.m_num_mesh_elements * rhs.m_num_attributes;
                m_h_attr = (T*)malloc(num_bytes);
                std::memcpy((void*)m_h_attr, rhs.m_h_attr, num_bytes);
            }
        }
        m_layout            = rhs.m_layout;
        m_is_axpy_allocated = rhs.m_is_axpy_allocated;
        if (rhs.m_is_axpy_allocated) {
            CUDA_ERROR(cudaMalloc((void**)&d_axpy_alpha,
                                  m_num_attributes * sizeof(T)));
            CUDA_ERROR(
                cudaMalloc((void**)&d_axpy_beta, m_num_attributes * sizeof(T)));
        }
        m_is_reduce_allocated = rhs.m_is_reduce_allocated;
        if (rhs.m_is_reduce_allocated) {
            allocate_reduce_temp_storage(m_is_reduce_allocated);
        }
        return *this;
    }

    __host__ __device__ __forceinline__ bool is_empty() const
    {
#ifdef __CUDA_ARCH__

        return (m_d_attr == nullptr) ? true : false;
#else
        return (m_h_attr == nullptr) ? true : false;

#endif
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


    void allocate_v1(const std::vector<uint16_t>& element_per_patch)
    {

        release_v1();

        if (m_num_patches != 0) {
            CUDA_ERROR(cudaMallocManaged((void**)&(m_attr),
                                         sizeof(T*) * m_num_patches));

            CUDA_ERROR(cudaMallocManaged((void**)&(m_element_per_patch),
                                         sizeof(uint16_t) * m_num_patches));

            for (uint32_t p = 0; p < m_num_patches; ++p) {
                m_element_per_patch[p] = element_per_patch[p];
                CUDA_ERROR(cudaMallocManaged((void**)&(m_attr[p]),
                                             sizeof(T) * element_per_patch[p]));
            }
        }

        m_is_allocated = true;
    }

    void allocate_reduce_temp_storage(size_t reduce_temp_bytes)
    {

        // Reduce operations are either SUM, MIN, MAX, or NORM2
        // NORM2 produce is done in two passes, the first pass uses cub
        // device API to multiply the input and then store in a temp buffer
        // (every CUDA block outputs a single value) which then is used for
        // the second pass using cub host API The other three operations
        // uses only cub host API. cub host API requires temp buffer which
        // is taken as the max of what NORM2 requires and the other three
        // operations.

        // NORM2 temp buffer (to store the per-block output)
        uint32_t num_blocks = DIVIDE_UP(m_num_mesh_elements, m_block_size);
        m_norm2_temp_buffer = (T**)malloc(sizeof(T*) * m_num_attributes);
        if (!m_norm2_temp_buffer) {
            RXMESH_ERROR(
                "RXMeshAttribute::allocate_reduce_temp_storage() could not "
                "allocate m_norm2_temp_buffer.");
        }
        for (uint32_t i = 0; i < m_num_attributes; ++i) {
            CUDA_ERROR(
                cudaMalloc(&m_norm2_temp_buffer[i], sizeof(T) * num_blocks));
        }

        m_d_reduce_output = (T**)malloc(sizeof(T*) * m_num_attributes);
        if (!m_d_reduce_output) {
            RXMESH_ERROR(
                "RXMeshAttribute::allocate_reduce_temp_storage() could not "
                "allocate m_d_reduce_output.");
        }
        m_d_reduce_temp_storage =
            (void**)malloc(sizeof(void*) * m_num_attributes);
        if (!m_d_reduce_temp_storage) {
            RXMESH_ERROR(
                "RXMeshAttribute::allocate_reduce_temp_storage() could not "
                "allocate m_d_reduce_temp_storage.");
        }
        m_reduce_streams =
            (cudaStream_t*)malloc(sizeof(cudaStream_t) * m_num_attributes);
        if (!m_d_reduce_output) {
            RXMESH_ERROR(
                "RXMeshAttribute::init() could not allocate "
                "m_reduce_streams.");
        }
        if (reduce_temp_bytes == 0) {
            // get the num bytes for cub device-wide reduce
            size_t norm2_temp_bytes(0), other_reduce_temp_bytes(0);
            T*     d_out(NULL);
            m_d_reduce_temp_storage[0] = NULL;
            cub::DeviceReduce::Sum(m_d_reduce_temp_storage[0],
                                   norm2_temp_bytes,
                                   m_d_attr,
                                   d_out,
                                   num_blocks);
            cub::DeviceReduce::Sum(m_d_reduce_temp_storage[0],
                                   other_reduce_temp_bytes,
                                   m_d_attr,
                                   d_out,
                                   m_num_mesh_elements);
            m_reduce_temp_storage_bytes =
                std::max(norm2_temp_bytes, other_reduce_temp_bytes);
        }

        for (uint32_t i = 0; i < m_num_attributes; ++i) {
            CUDA_ERROR(cudaMalloc(&m_d_reduce_temp_storage[i],
                                  m_reduce_temp_storage_bytes));
            CUDA_ERROR(cudaMalloc(&m_d_reduce_output[i], sizeof(T)));
            CUDA_ERROR(cudaStreamCreate(&m_reduce_streams[i]));
        }
    }

    char*     m_name;
    uint32_t  m_num_mesh_elements;
    uint32_t  m_num_attributes;
    locationT m_allocated;
    bool      m_is_allocated;
    T*        m_h_attr;
    T*        m_d_attr;
    T**       m_attr;
    uint32_t  m_num_patches;
    uint16_t* m_element_per_patch;
    layoutT   m_layout;

    constexpr static uint32_t m_block_size = 256;

    // temp array for alpha and beta parameters of axpy allocated on the device
    T *  d_axpy_alpha, *d_axpy_beta;
    bool m_is_axpy_allocated;

    // temp array for reduce operations
    bool          m_is_reduce_allocated;
    size_t        m_reduce_temp_storage_bytes;
    void**        m_d_reduce_temp_storage;
    T**           m_d_reduce_output;
    cudaStream_t* m_reduce_streams;
    T**           m_norm2_temp_buffer;
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
                        const layoutT                layout,
                        const bool                   with_axpy_alloc,
                        const bool                   with_reduce_alloc)
        : RXMeshAttribute<T>(name)
    {
        this->init_v1(face_per_patch,
                      num_attributes,
                      layout,
                      with_axpy_alloc,
                      with_reduce_alloc);
    }

    __host__ __device__ __forceinline__ T& operator()(FaceHandle fid,
                                                      uint32_t   attr) const
    {
        return 0;
    }

    __host__ __device__ __forceinline__ T& operator()(FaceHandle fid) const
    {
        return (*this)(fid, 0);
    }

    __host__ __device__ __forceinline__ T& operator()(FaceHandle fid,
                                                      uint32_t   attr)
    {
        return 0;
    }

    __host__ __device__ __forceinline__ T& operator()(FaceHandle fid)
    {
        return (*this)(fid, 0);
    }

   private:
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
                        const layoutT                layout,
                        const bool                   with_axpy_alloc,
                        const bool                   with_reduce_alloc)
        : RXMeshAttribute<T>(name)
    {
        this->init_v1(edge_per_patch,
                      num_attributes,
                      layout,
                      with_axpy_alloc,
                      with_reduce_alloc);
    }

    __host__ __device__ __forceinline__ T& operator()(EdgeHandle fid,
                                                      uint32_t   attr) const
    {
        return 0;
    }

    __host__ __device__ __forceinline__ T& operator()(EdgeHandle fid) const
    {
        return (*this)(fid, 0);
    }

    __host__ __device__ __forceinline__ T& operator()(EdgeHandle fid,
                                                      uint32_t   attr)
    {
        return 0;
    }

    __host__ __device__ __forceinline__ T& operator()(EdgeHandle fid)
    {
        return (*this)(fid, 0);
    }

   private:
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
                          const layoutT                layout,
                          const bool                   with_axpy_alloc,
                          const bool                   with_reduce_alloc)
        : RXMeshAttribute<T>(name)
    {
        this->init_v1(vertex_per_patch,
                      num_attributes,
                      layout,
                      with_axpy_alloc,
                      with_reduce_alloc);
    }

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx) const
    {
        return (*this)(idx, 0);
    }

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx)
    {
        return (*this)(idx, 0);
    }

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx,
                                                      uint32_t attr) const
    {
        return (*this)(idx, attr);
    }

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx,
                                                      uint32_t attr)
    {
        return (*this)(idx, attr);
    }

    __host__ __device__ __forceinline__ T& operator()(VertexHandle fid,
                                                      uint32_t     attr) const
    {
        return 0;
    }

    __host__ __device__ __forceinline__ T& operator()(VertexHandle fid) const
    {
        return (*this)(fid, 0);
    }

    __host__ __device__ __forceinline__ T& operator()(VertexHandle fid,
                                                      uint32_t     attr)
    {
        return 0;
    }

    __host__ __device__ __forceinline__ T& operator()(VertexHandle fid)
    {
        return (*this)(fid, 0);
    }
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
                               layoutT                layout,
                               const bool             with_axpy_alloc,
                               const bool             with_reduce_alloc)
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
                                                layout,
                                                with_axpy_alloc,
                                                with_reduce_alloc);
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