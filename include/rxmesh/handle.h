#pragma once
#include <stdint.h>
#include <string>
#include "rxmesh/local.h"
#include "rxmesh/patch_info.h"
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


    __device__ __host__ VertexHandle() : m_handle(INVALID64)
    {
    }

    __device__ __host__ VertexHandle(uint32_t     patch_id,
                                     LocalVertexT vertex_local_id)
        : m_handle(detail::unique_id(vertex_local_id.id, patch_id))
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

    std::pair<uint32_t, uint16_t> __device__
        __host__ __inline__ unpack(const PatchInfo* patch_info_base) const
    {
        auto ret = unpack();

        if (!is_valid()) {
            return ret;
        }

        const uint16_t num_owned_v =
            patch_info_base[ret.first].num_owned_vertices;
        if (ret.second >= num_owned_v) {
            const uint32_t p = ret.first;
            ret.first =
                patch_info_base[p].not_owned_patch_v[ret.second - num_owned_v];
            ret.second =
                patch_info_base[p].not_owned_id_v[ret.second - num_owned_v].id;
        }

        return ret;
    }

   private:
    uint64_t m_handle;
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


    __device__ __host__ EdgeHandle() : m_handle(INVALID64)
    {
    }

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

    std::pair<uint32_t, uint16_t> __device__
        __host__ __inline__ unpack(const PatchInfo* patch_info_base) const
    {
        auto ret = unpack();

        if (!is_valid()) {
            return ret;
        }

        const uint16_t num_owned_e = patch_info_base[ret.first].num_owned_edges;
        if (ret.second >= num_owned_e) {
            const uint32_t p = ret.first;
            ret.first =
                patch_info_base[p].not_owned_patch_e[ret.second - num_owned_e];
            ret.second =
                patch_info_base[p].not_owned_id_e[ret.second - num_owned_e].id;
        }
        return ret;
    }

   private:
    uint64_t m_handle;
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

    __device__ __host__ FaceHandle() : m_handle(INVALID64)
    {
    }

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

    std::pair<uint32_t, uint16_t> __device__
        __host__ __inline__ unpack(const PatchInfo* patch_info_base) const
    {
        auto ret = unpack();

        if (!is_valid()) {
            return ret;
        }

        const uint16_t num_owned_f = patch_info_base[ret.first].num_owned_faces;
        if (ret.second >= num_owned_f) {
            const uint32_t p = ret.first;
            ret.first =
                patch_info_base[p].not_owned_patch_f[ret.second - num_owned_f];
            ret.second =
                patch_info_base[p].not_owned_id_f[ret.second - num_owned_f].id;
        }

        return ret;
    }

   private:
    uint64_t m_handle;
};

/**
 * @brief print face unique_id to stream
 */
inline std::ostream& operator<<(std::ostream& os, FaceHandle f_handle)
{
    return (os << 'f' << f_handle.unique_id());
}
}  // namespace rxmesh