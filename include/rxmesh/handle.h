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
uint64_t __device__ __host__ __forceinline__ unique_id(const uint16_t local_id,
                                                       const uint32_t patch_id)
{
    uint64_t ret = patch_id;
    ret          = (ret << 32);
    ret |= local_id;
    return ret;
}

/**
 * @brief unpack a 64 uint its high and low 32 bits. The low 32 bit are casted
 * to 16 bit. This is unused to convert the unique id to its local id (16 low
 * bit) and patch id (high 32 bit)
 * @param uid unique id
 * @return a std::pair storing the patch id and local id
 */
std::pair<uint32_t, uint16_t> __device__ __host__ __forceinline__
unpack(uint64_t uid)
{
    uint16_t local_id = uid & ((1 << 16) - 1);
    uint32_t patch_id = uid >> 32;
    return std::make_pair(patch_id, local_id);
}

}  // namespace detail

/**
 * @brief vertices identifier
 */
struct VertexHandle
{
    using LocalT = LocalVertexT;


    __device__ __host__ VertexHandle()
        : m_handle(INVALID64)/*, m_patch_info(nullptr)*/
    {
    }

    /*__device__ __host__ VertexHandle(const PatchInfo* patch_info,
                                     LocalVertexT     vertex_local_id)
        : m_handle(detail::unique_id(vertex_local_id.id, patch_info->patch_id)),
          m_patch_info(patch_info)
    {
    }*/

    __device__ __host__ VertexHandle(uint32_t     patch_id,
                                     LocalVertexT vertex_local_id)
        : m_handle(detail::unique_id(vertex_local_id.id, patch_id))/*,
          m_patch_info(nullptr)*/
    {
    }

    bool __device__ __host__ __inline__ operator==(
        const VertexHandle& rhs) const
    {
        return m_handle == rhs.m_handle;
    }

    bool __device__ __host__ __inline__ operator!=(
        const VertexHandle& rhs) const
    {
        return !(*this == rhs);
    }

    bool __device__ __host__ __inline__ is_valid() const
    {
        return m_handle != INVALID64;
    }

    uint64_t __device__ __host__ __inline__ unique_id() const
    {
        return m_handle;
    }

    std::pair<uint32_t, uint16_t> __device__ __host__ __inline__ unpack() const
    {
        return detail::unpack(m_handle);
    }

   private:
    uint64_t         m_handle;
    //const PatchInfo* m_patch_info;
};

/**
 * @brief print vertex unique_id to stream
 */
inline std::ostream& operator<<(std::ostream& os, VertexHandle v_handle)
{
    return (os << 'v' << v_handle.unique_id());
}

/**
 * @brief edges identifier
 */
struct EdgeHandle
{
    using LocalT = LocalEdgeT;


    __device__ __host__ EdgeHandle()
        : m_handle(INVALID64)//, m_patch_info(nullptr)
    {
    }

    /*__device__ __host__ EdgeHandle(const PatchInfo* patch_info,
                                   LocalEdgeT       edge_local_id)
        : m_handle(
              detail::unique_id(edge_local_id.id, patch_info->patch_id)),
          m_patch_info(patch_info)
    {
    }*/

    __device__ __host__ EdgeHandle(uint32_t patch_id, LocalEdgeT edge_local_id)
        : m_handle(detail::unique_id(edge_local_id.id, patch_id))
    {
    }

    bool __device__ __host__ __inline__ operator==(const EdgeHandle& rhs) const
    {
        return m_handle == rhs.m_handle;
    }

    bool __device__ __host__ __inline__ operator!=(const EdgeHandle& rhs) const
    {
        return !(*this == rhs);
    }

    bool __device__ __host__ __inline__ is_valid() const
    {
        return m_handle != INVALID64;
    }

    uint64_t __device__ __host__ __inline__ unique_id() const
    {
        return m_handle;
    }

    std::pair<uint32_t, uint16_t> __device__ __host__ __inline__ unpack() const
    {
        return detail::unpack(m_handle);
    }

   private:
    uint64_t         m_handle;
    //const PatchInfo* m_patch_info;
};

/**
 * @brief print edge unique_id to stream
 */
inline std::ostream& operator<<(std::ostream& os, EdgeHandle e_handle)
{
    return (os << 'e' << e_handle.unique_id());
}

/**
 * @brief faces identifier
 */
struct FaceHandle
{
    using LocalT = LocalFaceT;

    __device__ __host__ FaceHandle()
        : m_handle(INVALID64)//, m_patch_info(nullptr)
    {
    }

    /*__device__ __host__ FaceHandle(const PatchInfo* patch_info,
                                   LocalFaceT       face_local_id)
        : m_handle(
              detail::unique_id(face_local_id.id, patch_info->patch_id)),
          m_patch_info(patch_info)
    {
    }*/

    __device__ __host__ FaceHandle(uint32_t patch_id, LocalFaceT face_local_id)
        : m_handle(detail::unique_id(face_local_id.id, patch_id))
    {
    }

    bool __device__ __host__ __inline__ operator==(const FaceHandle& rhs) const
    {
        return m_handle == rhs.m_handle;
    }

    bool __device__ __host__ __inline__ operator!=(const FaceHandle& rhs) const
    {
        return !(*this == rhs);
    }

    bool __device__ __host__ __inline__ is_valid() const
    {
        return m_handle != INVALID64;
    }

    uint64_t __device__ __host__ __inline__ unique_id() const
    {
        return m_handle;
    }

    std::pair<uint32_t, uint16_t> __device__ __host__ __inline__ unpack() const
    {
        return detail::unpack(m_handle);
    }

   private:
    uint64_t         m_handle;
    //const PatchInfo* m_patch_info;
};

/**
 * @brief print face unique_id to stream
 */
inline std::ostream& operator<<(std::ostream& os, FaceHandle f_handle)
{
    return (os << 'f' << f_handle.unique_id());
}
}  // namespace rxmesh