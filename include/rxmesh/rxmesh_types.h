#pragma once
#include <stdint.h>
#include <string>
#include "rxmesh/util/macros.h"

namespace rxmesh {

/**
 * @brief ELEMENT represents the three types of mesh elements
 */
enum class ELEMENT
{
    VERTEX = 0,
    EDGE   = 1,
    FACE   = 2
};

/**
 * @brief Various query operations supported in RXMesh
 */
enum class Op
{
    VV = 0,
    VE = 1,
    VF = 2,
    FV = 3,
    FE = 4,
    FF = 5,
    EV = 6,
    EE = 7,
    EF = 8,
};

/**
 * @brief Convert an operation to string
 * @param op a query operation
 * @return name of the query operation as a string
 */
inline std::string op_to_string(const Op& op)
{
    switch (op) {
        case Op::VV:
            return "VV";
        case Op::VE:
            return "VE";
        case Op::VF:
            return "VF";
        case Op::FV:
            return "FV";
        case Op::FE:
            return "FE";
        case Op::FF:
            return "FF";
        case Op::EV:
            return "EV";
        case Op::EF:
            return "EF";
        case Op::EE:
            return "EE";
        default: {
            RXMESH_ERROR("op_to_string() unknown input operation");
            return "";
        }
    }
}

/**
 * @brief Given a query operations, what is the input and output of the
 * operation
 * @param op a query operation
 * @param source_ele the input of op
 * @param output_ele the output of op
 * @return
 */
void __device__ __host__ __inline__ io_elements(const Op& op,
                                                ELEMENT&  source_ele,
                                                ELEMENT&  output_ele)
{
    if (op == Op::VV || op == Op::VE || op == Op::VF) {
        source_ele = ELEMENT::VERTEX;
    } else if (op == Op::EV || op == Op::EE || op == Op::EF) {
        source_ele = ELEMENT::EDGE;
    } else if (op == Op::FV || op == Op::FE || op == Op::FF) {
        source_ele = ELEMENT::FACE;
    }
    if (op == Op::VV || op == Op::EV || op == Op::FV) {
        output_ele = ELEMENT::VERTEX;
    } else if (op == Op::VE || op == Op::EE || op == Op::FE) {
        output_ele = ELEMENT::EDGE;
    } else if (op == Op::VF || op == Op::EF || op == Op::FF) {
        output_ele = ELEMENT::FACE;
    }
}

/**
 * @brief Local vertex type (wrapped around uint16_t)
 */
struct LocalVertexT
{
    __device__ __host__ LocalVertexT() : id(INVALID16)
    {
    }
    __device__ __host__ LocalVertexT(uint16_t id) : id(id)
    {
    }
    uint16_t id;
};

/**
 * @brief Local edge type (wrapped around uint16_t)
 */
struct LocalEdgeT
{
    __device__ __host__ LocalEdgeT() : id(INVALID16)
    {
    }
    __device__ __host__ LocalEdgeT(uint16_t id) : id(id)
    {
    }
    uint16_t id;
};

/**
 * @brief Local face type (wrapped around uint16_t)
 */
struct LocalFaceT
{
    __device__ __host__ LocalFaceT() : id(INVALID16)
    {
    }
    __device__ __host__ LocalFaceT(uint16_t id) : id(id)
    {
    }
    uint16_t id;
};

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
}  // namespace detail

/**
 * @brief vertices identifier
 */
struct VertexHandle
{
    using LocalT = LocalVertexT;

    template <typename T>
    friend class RXMeshVertexAttribute;
    friend class RXMeshStatic;
    friend class PatchInfo;

    __device__ __host__ VertexHandle() : m_patch_id(INVALID32), m_v({INVALID16})
    {
    }

    __device__ __host__ VertexHandle(uint32_t     patch_id,
                                     LocalVertexT vertex_local_id)
        : m_patch_id(patch_id), m_v(vertex_local_id)
    {
    }

    bool __device__ __host__ __inline__ operator==(
        const VertexHandle& rhs) const
    {
        return m_v.id == rhs.m_v.id && m_patch_id == rhs.m_patch_id;
    }

    bool __device__ __host__ __inline__ operator!=(
        const VertexHandle& rhs) const
    {
        return !(*this == rhs);
    }

    bool __device__ __host__ __inline__ is_valid() const
    {
        return m_patch_id != INVALID32 && m_v.id != INVALID16;
    }

    uint64_t __device__ __host__ __inline__ unique_id()
    {
        return detail::unique_id(m_v.id, m_patch_id);
    }

   private:
    uint32_t     m_patch_id;
    LocalVertexT m_v;
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

    template <typename T>
    friend class RXMeshEdgeAttribute;
    friend class RXMeshStatic;
    friend class PatchInfo;

    __device__ __host__ EdgeHandle() : m_patch_id(INVALID32), m_e({INVALID16})
    {
    }

    __device__ __host__ EdgeHandle(uint32_t patch_id, LocalEdgeT edge_local_id)
        : m_patch_id(patch_id), m_e(edge_local_id)
    {
    }

    bool __device__ __host__ __inline__ operator==(const EdgeHandle& rhs) const
    {
        return m_e.id == rhs.m_e.id && m_patch_id == rhs.m_patch_id;
    }

    bool __device__ __host__ __inline__ operator!=(const EdgeHandle& rhs) const
    {
        return !(*this == rhs);
    }

    bool __device__ __host__ __inline__ is_valid() const
    {
        return m_patch_id != INVALID32 && m_e.id != INVALID16;
    }

    uint64_t __device__ __host__ __inline__ unique_id()
    {
        return detail::unique_id(m_e.id, m_patch_id);
    }

   private:
    uint32_t   m_patch_id;
    LocalEdgeT m_e;
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

    template <typename T>
    friend class RXMeshFaceAttribute;
    friend class RXMeshStatic;
    friend class PatchInfo;

    __device__ __host__ FaceHandle() : m_patch_id(INVALID32), m_f({INVALID16})
    {
    }

    __device__ __host__ FaceHandle(uint32_t patch_id, LocalFaceT face_local_id)
        : m_patch_id(patch_id), m_f(face_local_id)
    {
    }

    bool __device__ __host__ __inline__ operator==(const FaceHandle& rhs) const
    {
        return m_f.id == rhs.m_f.id && m_patch_id == rhs.m_patch_id;
    }

    bool __device__ __host__ __inline__ operator!=(const FaceHandle& rhs) const
    {
        return !(*this == rhs);
    }

    bool __device__ __host__ __inline__ is_valid() const
    {
        return m_patch_id != INVALID32 && m_f.id != INVALID16;
    }

    uint64_t __device__ __host__ __inline__ unique_id() const
    {
        return detail::unique_id(m_f.id, m_patch_id);
    }

   private:
    uint32_t   m_patch_id;
    LocalFaceT m_f;
};

/**
 * @brief print face unique_id to stream
 */
inline std::ostream& operator<<(std::ostream& os, FaceHandle f_handle)
{
    return (os << 'f' << f_handle.unique_id());
}
}  // namespace rxmesh