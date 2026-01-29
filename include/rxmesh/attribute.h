#pragma once

#include <assert.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rxmesh/handle.h"
#include "rxmesh/patch_info.h"
#include "rxmesh/types.h"

#include "rxmesh/matrix/dense_matrix.h"

#include <Eigen/Dense>

class RXMeshTest;

namespace rxmesh {

class RXMeshStatic;

/**
 * @brief Base untyped attributes used as an interface for attribute container
 */
class AttributeBase
{
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
 * @tparam HandleT One of VertexHandle, EdgeHandle, or FaceHandle
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
    Attribute();

    /**
     * @brief Main constructor to be used by RXMeshStatic not directly by the
     * user
     * @param name of the attribute
     * @param num_attributes number of attribute per face
     * @param location where the attribute to be allocated
     * @param layout memory layout in case of num_attributes>1
     * @param rxmesh pointer to the RXMeshStatic instance
     */
    explicit Attribute(const char*    name,
                       const uint32_t num_attributes,
                       locationT      location,
                       const layoutT  layout,
                       RXMeshStatic*  rxmesh);

    T& operator()(size_t i, size_t j = 0);

    T& operator()(size_t i, size_t j = 0) const;

    /** @brief Number of rows (mesh elements) */
    size_t rows() const;

    /** @brief Number of columns (attributes per element) */
    size_t cols() const;

    /** @brief Total number of mesh elements (vertices, edges, or faces) */
    uint32_t size() const;

    /**
     * @brief convert the attributes stored into a dense matrix where number of
     * rows represent the number of mesh elements of this attribute and number
     * of columns is the number of attributes
     * @tparam Order storage order (Eigen::ColMajor or Eigen::RowMajor)
     */
    template <int Order = Eigen::ColMajor>
    std::shared_ptr<DenseMatrix<T, Order>> to_matrix() const;

    /**
     * @brief copy a dense matrix to this attribute. The copying happens on the
     * host side, i.e., we copy the content of mat which is on the host to this
     * attribute on the host side
     * @tparam Order storage order of the input matrix
     * @param mat dense matrix to copy from
     */
    template <int Order>
    void from_matrix(DenseMatrix<T, Order>* mat);

    /**
     * @brief get the number of elements in a patch. The element type
     * corresponds to the template HandleT
     * @param p the patch id
     */
    __host__ __device__ __inline__ uint32_t size(const uint32_t p) const;

    /**
     * @brief get maximum number of elements in a patch. The element type
     * corresponds to the template HandleT
     * @param p the patch id
     */
    __host__ __device__ __inline__ uint32_t
    capacity(const uint32_t p) const;

    /**
     * @brief Get patch info for patch p
     */
    __host__ __device__ __inline__ const PatchInfo& get_patch_info(
        const uint32_t p) const;


    __host__ __device__ __inline__ uint32_t pitch_x() const;


    __host__ __device__ __inline__ uint32_t
    pitch_y(const uint32_t p) const;

    Attribute(const Attribute& rhs) = default;

    virtual ~Attribute() = default;

    /**
     * @brief Get the name of the attribute
     */
    const char* get_name() const;

    /**
     * @brief return the amount of allocated memory in megabytes
     */
    double get_memory_mg() const;

    /**
     * @brief get the number of attributes per mesh element
     */
    __host__ __device__ __inline__ uint32_t get_num_attributes() const;

    /**
     * @brief Flag that indicates where the memory is allocated
     */
    __host__ __device__ __inline__ locationT get_allocated() const;

    /**
     * @brief return the memory layout
     */
    __host__ __device__ __inline__ layoutT get_layout() const;

    /**
     * @brief Check if attribute is allocated on device
     */
    __host__ __device__ __inline__ bool is_device_allocated() const;

    /**
     * @brief Check if attribute is allocated on host
     */
    __host__ __device__ __inline__ bool is_host_allocated() const;

    /**
     * @brief Reset attribute to certain value
     * @param value value to set
     * @param location which location (device, host, or both) where attribute
     * will be set
     * @param stream in case of DEVICE, this is the stream that will be used to
     * launch the reset kernel
     */
    void reset(const T value, locationT location, cudaStream_t stream = NULL);

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
    void move(locationT source, locationT target, cudaStream_t stream = NULL);

    /**
     * @brief Release allocated memory in certain location
     * @param location where memory will be released
     */
    void release(locationT location = LOCATION_ALL);

    /**
     * @brief Deep copy from a source attribute. If source_flag and dst_flag are
     * both set to LOCATION_ALL, then we copy what is on host to host, and what
     * on device to device. If source_flag is set to HOST (or DEVICE) and
     * dst_flag is set to LOCATION_ALL, then we copy source's HOST (or
     * DEVICE) to both HOST and DEVICE.
     * @param source attribute to copy from
     * @param source_flag defines where we will copy from
     * @param dst_flag defines where we will copy to
     * @param stream used to launch kernel/memcpy
     */
    void copy_from(Attribute<T, HandleT>& source,
                   locationT              source_flag,
                   locationT              dst_flag,
                   cudaStream_t           stream = NULL);

    /**
     * @brief Accessing an attribute using a handle to the mesh element
     * @param handle input handle
     * @param attr the attribute id
     * @return const reference to the attribute
     */
    __host__ __device__ __inline__ T& operator()(
        const HandleT  handle,
        const uint32_t attr = 0) const;

    /**
     * @brief Accessing the attribute as a glm vector. This is used for read
     * only since the return result is a copy.
     * @tparam N dimension of the vector
     */
    template <int N>
    __host__ __device__ __inline__ vec<T, N> to_glm(
        const HandleT& handle) const;

    /**
     * @brief store a glm vector in this attribute. The size of glm vector
     * should match the number of attributes in this attribute
     * @tparam N dimension of the vector
     */
    template <int N>
    __host__ __device__ __inline__ void from_glm(const HandleT&   handle,
                                                 const vec<T, N>& in);

    /**
     * @brief Accessing the attribute as an Eigen matrix. This is used for read
     * only since the return result is a copy.
     * @tparam N dimension of the vector
     */
    template <int N>
    __host__ __device__ __inline__ Eigen::Matrix<T, N, 1> to_eigen(
        const HandleT& handle) const;

    /**
     * @brief store an Eigen (small) vector in this attribute. The size of Eigen
     * vector should match the number of attributes in this attribute
     * @tparam N dimension of the vector
     */
    template <int N>
    __host__ __device__ __inline__ void from_eigen(
        const HandleT&                handle,
        const Eigen::Matrix<T, N, 1>& in);

    /**
     * @brief Accessing an attribute using a handle to the mesh element
     * @param handle input handle
     * @param attr the attribute id
     * @return non-const reference to the attribute
     */
    __host__ __device__ __inline__ T& operator()(const HandleT  handle,
                                                      const uint32_t attr = 0);

    /**
     * @brief Access the attribute value using patch and local index in the
     * patch. This is meant to be used by XXAttribute not directly by the user
     * @param p_id patch to be accessed
     * @param local_id the local id in the patch
     * @param attr the attribute id
     * @return const reference to the attribute
     */
    __host__ __device__ __inline__ T& operator()(
        const uint32_t p_id,
        const uint16_t local_id,
        const uint32_t attr) const;

    /**
     * @brief Access the attribute value using patch and local index in the
     * patch. This is meant to be used by XXAttribute not directly by the user
     * @param p_id patch to be accessed
     * @param local_id the local id in the patch
     * @param attr the attribute id
     * @return non-const reference to the attribute
     */
    __host__ __device__ __inline__ T& operator()(const uint32_t p_id,
                                                      const uint16_t local_id,
                                                      const uint32_t attr);

    /**
     * @brief Check if the attribute is empty
     */
    __host__ __device__ __inline__ bool is_empty() const;

   protected:
    /**
     * @brief allocate internal memory
     */
    void allocate(locationT location);

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
    AttributeContainer() = default;

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
    size_t size();

    /**
     * @brief get a list of name of the attributes managed by this container
     */
    std::vector<std::string> get_attribute_names() const;

    /**
     * @brief add a new attribute to be managed by this container
     * @tparam AttrT attribute type
     * @param name unique name given to the attribute
     * @param num_attributes number of attributes per mesh element
     * @param location where the attributes will be allocated
     * @param layout memory layout in case of num_attributes > 1
     * @param rxmesh pointer to the RXMeshStatic instance
     * @return a shared pointer to the attribute
     */
    template <typename AttrT>
    std::shared_ptr<AttrT> add(const char*   name,
                               uint32_t      num_attributes,
                               locationT     location,
                               layoutT       layout,
                               RXMeshStatic* rxmesh);

    /**
     * @brief Check if an attribute exists
     * @param name of the attribute
     */
    bool does_exist(const char* name);

    /**
     * @brief remove an attribute and release its memory
     * @param name of the attribute
     */
    void remove(const char* name);

   private:
    std::vector<std::shared_ptr<AttributeBase>> m_attr_container;
};

}  // namespace rxmesh
