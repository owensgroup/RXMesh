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
    LocalVertexT() : id(INVALID16)
    {
    }
    LocalVertexT(uint16_t id) : id(id)
    {
    }
    uint16_t id;
};

/**
 * @brief Local edge type (wrapped around uint16_t)
 */
struct LocalEdgeT
{
    LocalEdgeT() : id(INVALID16)
    {
    }
    LocalEdgeT(uint16_t id) : id(id)
    {
    }
    uint16_t id;
};

/**
 * @brief Local face type (wrapped around uint16_t)
 */
struct LocalFaceT
{
    LocalFaceT() : id(INVALID16)
    {
    }
    LocalFaceT(uint16_t id) : id(id)
    {
    }
    uint16_t id;
};


/**
 * @brief Unique identifier for vertices
 */
struct VertexHandle
{
    VertexHandle() : m_patch_id(INVALID32), m_v({INVALID16})
    {
    }
    VertexHandle(uint32_t patch_id, LocalVertexT vertex_local_id)
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

   private:
    uint32_t     m_patch_id;
    LocalVertexT m_v;
};

/**
 * @brief Unique identifier for edges
 */
struct EdgeHandle
{
    EdgeHandle() : m_patch_id(INVALID32), m_e({INVALID16})
    {
    }
    EdgeHandle(uint32_t patch_id, LocalEdgeT edge_local_id)
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

   private:
    uint32_t   m_patch_id;
    LocalEdgeT m_e;
};

/**
 * @brief Unique identifier for faces
 */
struct FaceHandle
{
    FaceHandle() : m_patch_id(INVALID32), m_f({INVALID16})
    {
    }
    FaceHandle(uint32_t patch_id, LocalFaceT face_local_id)
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

   private:
    uint32_t   m_patch_id;
    LocalFaceT m_f;
};

}  // namespace rxmesh