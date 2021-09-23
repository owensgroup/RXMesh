#pragma once

#include <assert.h>
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/rxmesh_attribute.cuh"
#include "rxmesh/kernels/util.cuh"
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

static std::string location_to_string(locationT target)
{
    std::string str = "";
    if ((target & HOST) == HOST) {
        str = (str == "" ? "" : " ") + std::string("HOST");
    }
    if ((target & DEVICE) == DEVICE) {
        str = (str == "" ? "" : " ") + std::string("DEVICE");
    }
    if (str == "") {
        str = "NONE";
    }
    return str;
}

template <class T>
class RXMeshAttribute
{
    // Here we manage the attributes on top of the mesh. An attributes is
    // attached to mesh element (e.g., vertices, edges, or faces). The user
    // is expected to declare as many attributes as expected to be used
    // during the lifetime of RXMesh

    // largely inspired by
    // https://github.com/gunrock/gunrock/blob/master/gunrock/util/array_utils.cuh


   public:
    RXMeshAttribute()
        : m_name(nullptr),
          m_num_mesh_elements(0),
          m_num_attribute_per_element(0),
          m_allocated(LOCATION_NONE),
          m_h_attr(nullptr),
          m_d_attr(nullptr),
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
        allocate(0, LOCATION_NONE);
        m_pitch.x = 0;
        m_pitch.y = 0;
    }

    RXMeshAttribute(const char* const name)
        : m_name(nullptr),
          m_num_mesh_elements(0),
          m_num_attribute_per_element(0),
          m_allocated(LOCATION_NONE),
          m_h_attr(nullptr),
          m_d_attr(nullptr),
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
        allocate(0, LOCATION_NONE);
        m_pitch.x = 0;
        m_pitch.y = 0;
    }


    void set_name(std::string name)
    {
        free(this->m_name);
        this->m_name = (char*)malloc(sizeof(char) * name.length() + 1);
        strcpy(this->m_name, name.c_str());
    }

    __host__ __device__ __forceinline__ uint32_t get_num_mesh_elements() const
    {
        return this->m_num_mesh_elements;
    }

    __host__ __device__ __forceinline__ uint32_t
    get_num_attribute_per_element() const
    {
        return this->m_num_attribute_per_element;
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

    __host__ __device__ __forceinline__ T* get_pointer(locationT target) const
    {

        if (target == DEVICE) {
            return m_d_attr;
        }
        if (target == HOST) {
            return m_h_attr;
        }
        return nullptr;
    }

    void reset(const T value, locationT target, cudaStream_t stream = NULL)
    {

        if ((target & DEVICE) == DEVICE) {

            assert((m_allocated & DEVICE) == DEVICE);

            const int      threads = 256;
            const uint32_t total =
                m_num_attribute_per_element * m_num_mesh_elements;
            memset<T><<<(total + threads - 1) / threads, threads, 0, stream>>>(
                m_d_attr, value, total);
            CUDA_ERROR(cudaDeviceSynchronize());
            CUDA_ERROR(cudaGetLastError());
        }


        if ((target & HOST) == HOST) {
            assert((m_allocated & HOST) == HOST);
            for (uint32_t i = 0;
                 i < m_num_mesh_elements * m_num_attribute_per_element;
                 ++i) {
                m_h_attr[i] = value;
            }
        }
    }

    void init(uint32_t   num_elements,
              uint32_t   num_attributes_per_elements,
              locationT  target            = DEVICE,
              layoutT    layout            = AoS,
              const bool with_axpy_alloc   = true,
              const bool with_reduce_alloc = true)
    {
        release();
        m_allocated                       = LOCATION_NONE;
        this->m_num_mesh_elements         = num_elements;
        this->m_num_attribute_per_element = num_attributes_per_elements;
        if (num_elements == 0) {
            return;
        }
        allocate(num_elements, target);
        m_layout = layout;
        set_pitch();

        if (!m_is_axpy_allocated && with_axpy_alloc) {
            CUDA_ERROR(cudaMalloc((void**)&d_axpy_alpha,
                                  m_num_attribute_per_element * sizeof(T)));
            CUDA_ERROR(cudaMalloc((void**)&d_axpy_beta,
                                  m_num_attribute_per_element * sizeof(T)));
            m_is_axpy_allocated = true;
        }

        if (!m_is_reduce_allocated && with_reduce_alloc) {
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
            m_norm2_temp_buffer =
                (T**)malloc(sizeof(T*) * m_num_attribute_per_element);
            if (!m_norm2_temp_buffer) {
                RXMESH_ERROR(
                    "RXMeshAttribute::init() could not allocate "
                    "m_norm2_temp_buffer.");
            }
            for (uint32_t i = 0; i < m_num_attribute_per_element; ++i) {
                CUDA_ERROR(cudaMalloc(&m_norm2_temp_buffer[i],
                                      sizeof(T) * num_blocks));
            }

            m_d_reduce_output =
                (T**)malloc(sizeof(T*) * m_num_attribute_per_element);
            if (!m_d_reduce_output) {
                RXMESH_ERROR(
                    "RXMeshAttribute::init() could not allocate "
                    "m_d_reduce_output.");
            }
            m_d_reduce_temp_storage =
                (void**)malloc(sizeof(void*) * m_num_attribute_per_element);
            if (!m_d_reduce_temp_storage) {
                RXMESH_ERROR(
                    "RXMeshAttribute::init() could not allocate "
                    "m_d_reduce_temp_storage.");
            }
            m_reduce_streams = (cudaStream_t*)malloc(
                sizeof(cudaStream_t) * m_num_attribute_per_element);
            if (!m_d_reduce_output) {
                RXMESH_ERROR(
                    "RXMeshAttribute::init() could not allocate "
                    "m_reduce_streams.");
            }
            {  // get the num bytes for cub device-wide reduce
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

            for (uint32_t i = 0; i < m_num_attribute_per_element; ++i) {
                CUDA_ERROR(cudaMalloc(&m_d_reduce_temp_storage[i],
                                      m_reduce_temp_storage_bytes));
                CUDA_ERROR(cudaMalloc(&m_d_reduce_output[i], sizeof(T)));
                CUDA_ERROR(cudaStreamCreate(&m_reduce_streams[i]));
            }
        }
    }

    void allocate(uint32_t num_mesh_elements, locationT target = DEVICE)
    {

        if ((target & HOST) == HOST) {
            release(HOST);
            if (num_mesh_elements != 0) {
                m_h_attr = (T*)malloc(sizeof(T) * num_mesh_elements *
                                      m_num_attribute_per_element);
                if (!m_h_attr) {
                    RXMESH_ERROR(
                        " RXMeshAttribute::allocate() allocation on {} failed "
                        "with #mesh_elemnts = {} and #attributes per element = "
                        "{}" +
                            location_to_string(HOST),
                        num_mesh_elements,
                        m_num_attribute_per_element);
                }
            }
            m_allocated = m_allocated | HOST;
        }


        if ((target & DEVICE) == DEVICE) {
            release(DEVICE);
            if (num_mesh_elements != 0) {
                CUDA_ERROR(cudaMalloc((void**)&(m_d_attr),
                                      sizeof(T) * num_mesh_elements *
                                          m_num_attribute_per_element));
            }
            m_allocated = m_allocated | DEVICE;
        }
        this->m_num_mesh_elements = num_mesh_elements;
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
            allocate(this->m_num_mesh_elements, target);
        }

        if (this->m_num_mesh_elements == 0) {
            return;
        }

        if (source == HOST && target == DEVICE) {
            CUDA_ERROR(cudaMemcpy(
                m_d_attr,
                m_h_attr,
                sizeof(T) * m_num_mesh_elements * m_num_attribute_per_element,
                cudaMemcpyHostToDevice));

        } else if (source == DEVICE && target == HOST) {
            CUDA_ERROR(cudaMemcpy(
                m_h_attr,
                m_d_attr,
                sizeof(T) * m_num_mesh_elements * m_num_attribute_per_element,
                cudaMemcpyDeviceToHost));
        }
    }

    void release(locationT target = LOCATION_ALL)
    {

        if (((target & HOST) == HOST) && ((m_allocated & HOST) == HOST)) {
            free(m_h_attr);
            m_h_attr    = nullptr;
            m_allocated = m_allocated & (~HOST);
        }

        if (((target & DEVICE) == DEVICE) &&
            ((m_allocated & DEVICE) == DEVICE)) {
            GPU_FREE(m_d_attr);
            m_allocated = m_allocated & (~DEVICE);
        }

        if (target == LOCATION_ALL || m_allocated == 0) {
            m_num_mesh_elements = 0;
            m_pitch.x           = 0;
            m_pitch.y           = 0;

            if (m_is_axpy_allocated) {
                GPU_FREE(d_axpy_alpha);
                GPU_FREE(d_axpy_beta);
                m_is_axpy_allocated = false;
            }
            if (m_is_reduce_allocated) {
                for (uint32_t i = 0; i < m_num_attribute_per_element; ++i) {
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
              locationT           target_flag)
    {
        // Deep copy from source. The source_flag defines where we will copy
        // from. The target_flag defines where we will copy to.

        // if source_flag and target_flag are both set to LOCATION_ALL, then we
        // copy what is on host to host, and what on target to target

        // If sourc_flag is set to HOST (or DEVICE) and target_flag is set to
        // LOCATION_ALL, then we copy source's HOST (or DEVICE) to both HOST
        // and DEVICE in target

        // Setting source_flag to LOCATION_ALL while target_flag is Not set to
        // LOCATION_ALL is invalid because we don't know which source to copy
        // from

        if (source.m_layout != m_layout) {
            RXMESH_ERROR(
                "RXMeshAttribute::copy() does not support copy from source of "
                "different layout!");
        }

        if ((source_flag & LOCATION_ALL) == LOCATION_ALL &&
            (target_flag & LOCATION_ALL) != LOCATION_ALL) {
            RXMESH_ERROR("RXMeshAttribute::copy() Invalid configuration!");
        }

        if (source.get_num_mesh_elements() != m_num_mesh_elements) {
            RXMESH_ERROR(
                "RXMeshAttribute::copy() source has different size than "
                "target!");
        }

        // 1) copy from HOST to HOST
        if ((source_flag & HOST) == HOST && (target_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on host");
            }
            if ((target_flag & m_allocated) != target_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because target (this) was not allocated on host");
            }

            std::memcpy(
                (void*)m_h_attr,
                source.m_h_attr,
                m_num_mesh_elements * m_num_attribute_per_element * sizeof(T));
        }


        // 2) copy from DEVICE to DEVICE
        if ((source_flag & DEVICE) == DEVICE &&
            (target_flag & DEVICE) == DEVICE) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on device");
            }
            if ((target_flag & m_allocated) != target_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because target (this) was not allocated on device");
            }

            CUDA_ERROR(cudaMemcpy(
                m_d_attr,
                source.m_d_attr,
                m_num_mesh_elements * m_num_attribute_per_element * sizeof(T),
                cudaMemcpyDeviceToDevice));
        }


        // 3) copy from DEVICE to HOST
        if ((source_flag & DEVICE) == DEVICE && (target_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on host");
            }
            if ((target_flag & m_allocated) != target_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because target (this) was not allocated on device");
            }

            CUDA_ERROR(cudaMemcpy(
                m_h_attr,
                source.m_d_attr,
                m_num_mesh_elements * m_num_attribute_per_element * sizeof(T),
                cudaMemcpyDeviceToHost));
        }


        // 4) copy from HOST to DEVICE
        if ((source_flag & HOST) == HOST && (target_flag & DEVICE) == DEVICE) {
            if ((source_flag & source.m_allocated) != source_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because it was not allocated on device");
            }
            if ((target_flag & m_allocated) != target_flag) {
                RXMESH_ERROR(
                    "RXMeshAttribute::copy() copying source is not valid"
                    " because target (this) was not allocated on host");
            }

            CUDA_ERROR(cudaMemcpy(
                m_d_attr,
                source.m_h_attr,
                m_num_mesh_elements * m_num_attribute_per_element * sizeof(T),
                cudaMemcpyHostToDevice));
        }
    }

    void change_layout(locationT target)
    {
        // Only supporting HOST target
        // If target is HOST, then the layout change only for the HOST
        // the user then can copy the data to the DEVICE.
        // To change the layout of data in the DEVICE, it should be copied first
        // to the HOST, change layout, and then copy back to the DEVICE

        // Only make sense when number of attributes is >1
        if (m_num_attribute_per_element > 1) {

            if ((target & m_allocated) != target) {
                RXMESH_ERROR(
                    "RXMeshAttribute::change_layout() changing layout {} is "
                    "not valid because it was not allocated",
                    location_to_string(target));
                return;
            }

            if ((target & HOST) != HOST) {
                RXMESH_ERROR(
                    "RXMeshAttribute::change_layout() changing layout {} is "
                    "not valid because it is not supported",
                    location_to_string(target));
                return;
            }

            if ((target & HOST) == HOST) {
                const uint32_t size =
                    m_num_mesh_elements * m_num_attribute_per_element;
                const uint32_t num_cols = (m_layout == AoS) ?
                                              m_num_attribute_per_element :
                                              m_num_mesh_elements;
                in_place_matrix_transpose(
                    m_h_attr, m_h_attr + size, uint64_t(num_cols));

                m_layout = (m_layout == SoA) ? AoS : SoA;
                set_pitch();
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
        // and alpha (and beta) should be of size m_num_attribute_per_element.
        // Otherwise axpy will be only applied on the given attribute number
        //(should be less than m_num_attribute_per_element) and alpha (and
        // beta) should be of size one
        // location tells on which side (host to device) the operation
        // will run

        const uint32_t num_attribute =
            (attribute_id == INVALID32) ? m_num_attribute_per_element : 1;
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
                for (uint32_t j = 0; j < m_num_attribute_per_element; ++j) {
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
        if (N < m_num_attribute_per_element) {
            RXMESH_ERROR(
                "RXMeshAttribute::reduce() the output Vector size should be "
                ">= the number of attributes per mesh element. Output "
                "Vector size = {}, number of attributes per mesh element = {}",
                N,
                m_num_attribute_per_element);
        }


        if ((location & DEVICE) == DEVICE) {
            if (m_layout != SoA) {
                RXMESH_ERROR(
                    "RXMeshAttribute::reduce is not supported for non SoA "
                    "layouts on the device");
            }
            for (uint32_t i = 0; i < m_num_attribute_per_element; ++i) {
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
            for (uint32_t j = 0; j < m_num_attribute_per_element; ++j) {
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

        assert(attr < m_num_attribute_per_element);
        assert(idx < m_num_mesh_elements);
        assert(m_pitch.x > 0 && m_pitch.y > 0);

#ifdef __CUDA_ARCH__
        return m_d_attr[idx * m_pitch.x + attr * m_pitch.y];
#else
        return m_h_attr[idx * m_pitch.x + attr * m_pitch.y];
#endif
    }

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx)
    {
        // for m_num_attribute_per_element =1

        assert(m_num_attribute_per_element == 1);
        assert(idx < m_num_mesh_elements);

#ifdef __CUDA_ARCH__
        return m_d_attr[idx];
#else
        return m_h_attr[idx];
#endif
    }

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx,
                                                      uint32_t attr) const
    {

        assert(attr < m_num_attribute_per_element);
        assert(idx < m_num_mesh_elements);

#ifdef __CUDA_ARCH__
        return m_d_attr[idx * m_pitch.x + attr * m_pitch.y];
#else
        return m_h_attr[idx * m_pitch.x + attr * m_pitch.y];
#endif
    }

    __host__ __device__ __forceinline__ T& operator()(uint32_t idx) const
    {
        // for m_num_attribute_per_element =1

        assert(m_num_attribute_per_element == 1);
        assert(idx < m_num_mesh_elements);

#ifdef __CUDA_ARCH__
        return m_d_attr[idx];
#else
        return m_h_attr[idx];
#endif
    }

    __host__ __device__ __forceinline__ T* operator->() const
    {
#ifdef __CUDA_ARCH__
        return m_d_attr;
#else
        return m_h_attr;
#endif
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
    void set_pitch()
    {
        if (m_layout == AoS) {
            m_pitch.x = m_num_attribute_per_element;
            m_pitch.y = 1;
        } else if (m_layout == SoA) {
            m_pitch.x = 1;
            m_pitch.y = m_num_mesh_elements;
        } else {
            RXMESH_ERROR("RXMeshAttribute::set_pitch() unknown layout");
        }
    }

    char*     m_name;
    uint32_t  m_num_mesh_elements;
    uint32_t  m_num_attribute_per_element;
    locationT m_allocated;
    T*        m_h_attr;
    T*        m_d_attr;
    layoutT   m_layout;
    // to index: id*m_pitch.x + attr*m_pitch.y
    uint2 m_pitch;

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
}  // namespace rxmesh