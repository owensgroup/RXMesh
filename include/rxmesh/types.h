#pragma once
#include <stdint.h>
#include <string>
#include "rxmesh/util/macros.h"

#include <glm/fwd.hpp>

namespace rxmesh {

template <typename T>
using vec1 = glm::vec<1, T, glm::defaultp>;

template <typename T>
using vec2 = glm::vec<2, T, glm::defaultp>;

template <typename T>
using vec3 = glm::vec<3, T, glm::defaultp>;

template <typename T>
using mat2x2 = glm::mat<2, 2, T, glm::defaultp>;

template <typename T>
using mat3x2 = glm::mat<3, 2, T, glm::defaultp>;

template <typename T>
using mat2x3 = glm::mat<2, 3, T, glm::defaultp>;

template <typename T>
using mat3x3 = glm::mat<3, 3, T, glm::defaultp>;

template <typename T>
using mat4x4 = glm::mat<4, 4, T, glm::defaultp>;


/**
 * @brief Flags for where data resides. Used with Attributes
 */
using locationT = uint32_t;
enum : locationT
{
    LOCATION_NONE = 0x00,
    HOST          = 0x01,
    DEVICE        = 0x02,
    LOCATION_ALL  = 0x0F,
};

/**
 * @brief convert locationT to string
 */
static std::string location_to_string(const locationT location)
{
    switch (location) {
        case LOCATION_NONE:
            return "NONE";
        case HOST:
            return "HOST";
        case DEVICE:
            return "DEVICE";
        case LOCATION_ALL:
            return "ALL";
        default: {
            RXMESH_ERROR("to_string() unknown location");
            return "";
        }
    }
}

/**
 * @brief Memory layout
 */
using layoutT = uint32_t;
enum : layoutT
{
    AoS = 0x00,
    SoA = 0x01,
};
/**
 * @brief convert locationT to string
 */
static std::string layout_to_string(const layoutT layout)
{
    switch (layout) {
        case AoS:
            return "AoS";
        case SoA:
            return "SoA";
        default: {
            RXMESH_ERROR("to_string() unknown layout");
            return "";
        }
    }
}

/**
 * @brief Various query operations supported in RXMeshStatic
 */
enum class Op
{
    V         = 0,
    E         = 1,
    F         = 2,
    VV        = 3,
    VE        = 4,
    VF        = 5,
    FV        = 6,
    FE        = 7,
    FF        = 8,
    EV        = 9,
    EE        = 10,
    EF        = 11,
    EVDiamond = 12,
};

/**
 * @brief defines how we create a cavity. The first element is the source and
 * the second is the target e.g., VE means removing a vertex and edges connected
 * to this vertex, FV means removing a face and vertices incident to this face.
 * Some operations are topologically redundant e.g., since removing a vertex
 * removes all its incident edges and faces, V and VE are equivalent.
 */
enum class CavityOp
{
    V  = 0,
    E  = 1,
    F  = 2,
    VV = 3,
    VE = V,  // same as removing a vertex
    VF = V,  // invalid since it leaves wire-frames edges so we fall back to V
    FV = 4,
    FE = 5,
    FF = FE,  // invalid since it leaves wire-frames edges so we fall back to FE
    EV = 6,
    EE = 7,
    EF = E,  // same as removing an edge
};

/**
 * @brief Convert an operation to string
 * @param op a query operation
 * @return name of the query operation as a string
 */
static std::string op_to_string(const Op& op)
{
    switch (op) {
        case Op::V:
            return "V";
        case Op::E:
            return "E";
        case Op::F:
            return "F";
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
        case Op::EVDiamond:
            return "EVDiamond";
        default: {
            RXMESH_ERROR("to_string() unknown input operation");
            return "";
        }
    }
}
}  // namespace rxmesh