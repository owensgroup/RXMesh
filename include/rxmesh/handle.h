#pragma once
#include <stdint.h>
#include <string>
#include "rxmesh/local.h"
#include "rxmesh/util/macros.h"

namespace rxmesh {

namespace detail {
/**
 * @brief Return unique index of the local mesh element composed by the
 * patch id and the local index
 *
 * @param local_id the local within-patch mesh element id
 * @param patch_id the patch owning the mesh element
 * @return
 */
constexpr __device__ __host__ __forceinline__ uint64_t
unique_id(const uint16_t local_id, const uint32_t patch_id)
{
    uint64_t ret = patch_id;
    ret          = (ret << 32);
    ret |= local_id;
    return ret;
}

/**
 * @brief unpack a 64 uint to its high and low 32 bits. The low 32 bit are
 * casted to 16 bit. This is used to convert the unique id to its local id (16
 * low bit) and patch id (high 32 bit)
 * @param uid unique id
 * @return a std::pair storing the patch id and local id
 */
constexpr __device__ __host__ __forceinline__ std::pair<uint32_t, uint16_t>
                                              unpack(uint64_t uid)
{
    uint16_t local_id = uid & ((1 << 16) - 1);
    uint32_t patch_id = uid >> 32;
    return std::make_pair(patch_id, local_id);
}

}  // namespace detail

/**
 * @brief vertex identifier. It is a unique handle for each vertex equipped with
 * operator==. It can be used to access mesh (vertex) attributes
 */
struct VertexHandle
{
    using LocalT = LocalVertexT;

    /**
     * @brief Default constructor
     */
    constexpr __device__ __host__ VertexHandle() : m_handle(INVALID64)
    {
    }

    /**
     * @brief Constructor with known (packed) handle
     */
    constexpr __device__ __host__ VertexHandle(uint64_t handle)
        : m_handle(handle)
    {
    }

    /**
     * @brief Constructor meant to be used internally by RXMesh and
     * query_dispatcher
     * @param patch_id the patch where the vertex belongs
     * @param vertex_local_id the vertex local index within the patch
     */
    constexpr __device__ __host__ VertexHandle(uint32_t     patch_id,
                                               LocalVertexT vertex_local_id)
        : m_handle(detail::unique_id(vertex_local_id.id, patch_id))
    {
    }

    /**
     * @brief Operator ==
     */
    constexpr __device__ __host__ __inline__ bool operator==(
        const VertexHandle& rhs) const
    {
        return m_handle == rhs.m_handle;
    }

    /**
     * @brief Operator !=
     */
    constexpr __device__ __host__ __inline__ bool operator!=(
        const VertexHandle& rhs) const
    {
        return !(*this == rhs);
    }

    /**
     * @brief Check if the vertex is valid i.e., has been initialized by RXMesh
     */
    constexpr __device__ __host__ __inline__ bool is_valid() const
    {
        return m_handle != INVALID64;
    }

    /**
     * @brief The unique identifier that represents the vertex
     */
    constexpr __device__ __host__ __inline__ uint64_t unique_id() const
    {
        return m_handle;
    }

    /**
     * @brief Unpack the handle to its patch id and vertex local index within
     * the patch
     */
    constexpr __device__ __host__ __inline__ std::pair<uint32_t, uint16_t>
                         unpack() const
    {
        return detail::unpack(m_handle);
    }

    /**
     * @brief return the patch id of this handle
     */
    constexpr __device__ __host__ __inline__ uint32_t patch_id() const
    {
        return unpack().first;
    }


    /**
     * @brief return the local index stored in this handle
     */
    constexpr __device__ __host__ __inline__ uint16_t local_id() const
    {
        return unpack().second;
    }

   protected:
    uint64_t m_handle;
};


/**
 * @brief edge identifier. It is a unique handle for each edge equipped with
 * operator==. It can be used to access mesh (edge) attributes
 */
struct EdgeHandle
{
    using LocalT = LocalEdgeT;

    /**
     * @brief Default constructor
     */
    constexpr __device__ __host__ EdgeHandle() : m_handle(INVALID64)
    {
    }

    /**
     * @brief Constructor with known (packed) handle
     */
    constexpr __device__ __host__ EdgeHandle(uint64_t handle) : m_handle(handle)
    {
    }

    /**
     * @brief Constructor meant to be used internally by RXMesh and
     * query_dispatcher
     * @param patch_id the patch where the edge belongs
     * @param edge_local_id the edge local index within the patch
     */
    constexpr __device__ __host__ EdgeHandle(uint32_t   patch_id,
                                             LocalEdgeT edge_local_id)
        : m_handle(detail::unique_id(edge_local_id.id, patch_id))
    {
    }

    /**
     * @brief Operator ==
     */
    constexpr __device__ __host__ __inline__ bool operator==(
        const EdgeHandle& rhs) const
    {
        return m_handle == rhs.m_handle;
    }


    /**
     * @brief Operator !=
     */
    constexpr __device__ __host__ __inline__ bool operator!=(
        const EdgeHandle& rhs) const
    {
        return !(*this == rhs);
    }

    /**
     * @brief Check if the edge is valid i.e., has been initialized by RXMesh
     */
    constexpr __device__ __host__ __inline__ bool is_valid() const
    {
        return m_handle != INVALID64;
    }

    /**
     * @brief The unique identifier that represents the edge
     */
    constexpr __device__ __host__ __inline__ uint64_t unique_id() const
    {
        return m_handle;
    }

    /**
     * @brief Unpack the handle to its patch id and edge local index within
     * the patch
     */
    constexpr __device__ __host__ __inline__ std::pair<uint32_t, uint16_t>
                         unpack() const
    {
        return detail::unpack(m_handle);
    }

    /**
     * @brief return the patch id of this handle
     */
    constexpr __device__ __host__ __inline__ uint32_t patch_id() const
    {
        return unpack().first;
    }

    /**
     * @brief return the local index stored in this handle
     */
    constexpr __device__ __host__ __inline__ uint16_t local_id() const
    {
        return unpack().second;
    }

   protected:
    uint64_t m_handle;
};

/**
 * @brief directed edges identifier. It is a unique handle for each edge
 * equipped with operator==. It is possible to extract the underlying EdgeHandle
 * of the directed edge. The idea is that every edge in the mesh could be split
 * into two edges of opposite direction (similar to half-edge data structure).
 */
struct DEdgeHandle
{
    using LocalT = LocalEdgeT;

    /**
     * @brief Default constructor
     */
    constexpr __device__ __host__ DEdgeHandle() : m_handle(INVALID64)
    {
    }

    /**
     * @brief Constructor meant to be used internally by RXMesh and
     * query_dispatcher
     * @param patch_id the patch where the edge belongs
     * @param edge_local_id the undirected edge local index within the patch
     * @param dir direction of the edge; 1 if flipped, 0 otherwise
     */
    constexpr __device__ __host__ DEdgeHandle(uint32_t   patch_id,
                                              LocalEdgeT edge_local_id,
                                              flag_t     dir)
        : m_handle(detail::unique_id((edge_local_id.id << 1) | dir, patch_id))
    {
    }


    /**
     * @brief Constructor meant to be used internally by RXMesh and
     * query_dispatcher
     * @param patch_id the patch where the edge belongs
     * @param edge_local_id the directed edge local index which has the
     * direction embedded in it i.e., in the first bit
     */
    constexpr __device__ __host__ DEdgeHandle(uint32_t   patch_id,
                                              LocalEdgeT edge_local_id)
        : m_handle(detail::unique_id(edge_local_id.id, patch_id))
    {
    }

    /**
     * @brief get a DEdgeHandle by flipping this directed edge
     */
    constexpr __device__ __host__ __inline__ DEdgeHandle get_flip_dedge() const
    {
        // extract and flip the lowest bit
        return DEdgeHandle(patch_id(), local_id(), ((unpack().second & 1) ^ 1));
    }

    /**
     * @brief extract the underlying undirected edge
     */
    constexpr __device__ __host__ __inline__ EdgeHandle get_edge_handle() const
    {
        return {patch_id(), local_id()};
    }


    /**
     * @brief Operator == compare this directed edge to an undirected one
     */
    constexpr __device__ __host__ __inline__ bool operator==(
        const EdgeHandle& rhs) const
    {
        return patch_id() == rhs.patch_id() && local_id() == rhs.local_id();
    }


    /**
     * @brief Operator !=
     */
    constexpr __device__ __host__ __inline__ bool operator!=(
        const EdgeHandle& rhs) const
    {
        return !(*this == rhs);
    }

    /**
     * @brief Operator == compare this directed edge to another directed one. To
     * directed edge are equal if both has the same underlying undirected edge
     * and have the same direction
     */
    constexpr __device__ __host__ __inline__ bool operator==(
        const DEdgeHandle& rhs) const
    {
        return m_handle == rhs.m_handle;
    }

    /**
     * @brief Operator !=
     */
    constexpr __device__ __host__ __inline__ bool operator!=(
        const DEdgeHandle& rhs) const
    {
        return !(*this == rhs);
    }

    /**
     * @brief Check if the edge is valid i.e., has been initialized by RXMesh
     */
    constexpr __device__ __host__ __inline__ bool is_valid() const
    {
        return m_handle != INVALID64;
    }

    /**
     * @brief The unique identifier that represents the edge
     */
    constexpr __device__ __host__ __inline__ uint64_t unique_id() const
    {
        return m_handle;
    }

    /**
     * @brief Unpack the handle to its patch id and edge local index within
     * the patch
     */
    constexpr __device__ __host__ __inline__ std::pair<uint32_t, uint16_t>
                         unpack() const
    {
        return detail::unpack(m_handle);
    }

    /**
     * @brief return the patch id of this handle
     */
    constexpr __device__ __host__ __inline__ uint32_t patch_id() const
    {
        return unpack().first;
    }

    /**
     * @brief return the local index stored in this handle
     */
    constexpr __device__ __host__ __inline__ uint16_t local_id() const
    {
        return unpack().second >> 1;
    }

   protected:
    uint64_t m_handle;
};

/**
 * @brief face identifier. It is a unique handle for each face equipped with
 * operator==. It can be used to access mesh (face) attributes
 */
struct FaceHandle
{
    using LocalT = LocalFaceT;

    /**
     * @brief Default constructor
     */
    constexpr __device__ __host__ FaceHandle() : m_handle(INVALID64)
    {
    }

    /**
     * @brief Constructor with known (packed) handle
     */
    constexpr __device__ __host__ FaceHandle(uint64_t handle) : m_handle(handle)
    {
    }

    /**
     * @brief Constructor meant to be used internally by RXMesh and
     * query_dispatcher
     * @param patch_id the patch where the face belongs
     * @param vertex_local_id the face local index within the patch
     */
    constexpr __device__ __host__ FaceHandle(uint32_t   patch_id,
                                             LocalFaceT face_local_id)
        : m_handle(detail::unique_id(face_local_id.id, patch_id))
    {
    }

    /**
     * @brief Operator ==
     */
    constexpr __device__ __host__ __inline__ bool operator==(
        const FaceHandle& rhs) const
    {
        return m_handle == rhs.m_handle;
    }

    /**
     * @brief Operator !=
     */
    constexpr __device__ __host__ __inline__ bool operator!=(
        const FaceHandle& rhs) const
    {
        return !(*this == rhs);
    }

    /**
     * @brief Check if the face is valid i.e., has been initialized by RXMesh
     */
    constexpr __device__ __host__ __inline__ bool is_valid() const
    {
        return m_handle != INVALID64;
    }

    /**
     * @brief The unique identifier that represents the face
     */
    constexpr __device__ __host__ __inline__ uint64_t unique_id() const
    {
        return m_handle;
    }

    /**
     * @brief Unpack the handle to its patch id and face local index within
     * the patch
     */
    constexpr __device__ __host__ __inline__ std::pair<uint32_t, uint16_t>
                         unpack() const
    {
        return detail::unpack(m_handle);
    }

    /**
     * @brief return the patch id of this handle
     */
    constexpr __device__ __host__ __inline__ uint32_t patch_id() const
    {
        return unpack().first;
    }

    /**
     * @brief return the local index stored in this handle
     */
    constexpr __device__ __host__ __inline__ uint16_t local_id() const
    {
        return unpack().second;
    }

   protected:
    uint64_t m_handle;
};
}  // namespace rxmesh