#pragma once

#include <type_traits>

#include "rxmesh/handle.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

// Return the underlying type of a DiffHandle which could be a Scalar<T> where T
// is a passive type or a passive type (e.g., float, double)
#define ACTIVE_TYPE(H) typename std::decay_t<decltype(H)>::Active

/**
 * @brief element identifier used in diff problems to switch between active and
 * passive mode to run the user-defined objective functions
 */
template <typename ActiveT, typename HandleT>
struct DiffHandle : public HandleT
{
    using Handle                   = HandleT;
    using LocalT                   = typename Handle::LocalT;
    constexpr static bool IsActive = is_scalar_v<ActiveT>;
    using Active                   = ActiveT;


    /**
     * @brief Default constructor
     */
    constexpr __device__ __host__ DiffHandle() : Handle()
    {
    }

    /**
     * @brief Constructor from HandleT
     */
    constexpr __device__ __host__ DiffHandle(HandleT handle)
        : DiffHandle(handle.unique_id())
    {
    }

    /**
     * @brief Constructor with known (packed) handle
     */
    explicit constexpr __device__ __host__ DiffHandle(uint64_t handle)
        : Handle(handle)
    {
    }

    /**
     * @brief Constructor meant to be used internally by RXMesh
     * @param patch_id the patch where the vertex belongs
     * @param vertex_local_id the vertex local index within the patch
     */
    constexpr __device__ __host__ DiffHandle(uint32_t patch_id,
                                             LocalT   vertex_local_id)
        : Handle(patch_id, vertex_local_id)
    {
    }

    /**
     * @brief Operator ==
     */
    constexpr __device__ __host__ __inline__ bool operator==(
        const DiffHandle& rhs) const
    {
        return this->m_handle == rhs.m_handle;
    }

    /**
     * @brief Operator !=
     */
    constexpr __device__ __host__ __inline__ bool operator!=(
        const DiffHandle& rhs) const
    {
        return !(*this == rhs);
    }

    constexpr __device__ __host__ __inline__ bool is_active() const
    {
        return IsActive;
    }
};

template <typename ActiveT>
using DiffVertexHandle = DiffHandle<ActiveT, VertexHandle>;

template <typename ActiveT>
using DiffEdgeHandle = DiffHandle<ActiveT, EdgeHandle>;

template <typename ActiveT>
using DiffFaceHandle = DiffHandle<ActiveT, FaceHandle>;

}  // namespace rxmesh