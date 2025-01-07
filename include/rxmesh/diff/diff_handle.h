#pragma once

#include <type_traits>

#include "rxmesh/handle.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

#define ACTIVE_TYPE(H) typename std::decay_t<decltype(H)>::Active

/**
 * @brief element identifier used in diff problems to switch between active and
 * passive mode to run the user-defined objective functions
 */
template <typename ActiveT, typename HandleT>
struct DiffHandle : public HandleT
{
    using Handle  = HandleT;
    using LocalT  = typename Handle::LocalT;
    bool IsActive = is_scalar_v<ActiveT>;
    using Active  = ActiveT;


    /**
     * @brief Default constructor
     */
    constexpr __device__ __host__ DiffHandle() : Handle()
    {
    }

    /**
     * @brief Constructor with known (packed) handle
     */
    constexpr __device__ __host__ DiffHandle(uint64_t handle) : Handle(handle)
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
};
}  // namespace rxmesh