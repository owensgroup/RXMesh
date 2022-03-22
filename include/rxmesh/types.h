#pragma once
#include <stdint.h>
#include <string>
#include "rxmesh/util/macros.h"

namespace rxmesh {

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
 * @brief ELEMENT represents the three types of mesh elements
 */
enum class ELEMENT
{
    VERTEX = 0,
    EDGE   = 1,
    FACE   = 2
};

/**
 * @brief Various query operations supported in RXMeshStatic
 */
enum class Op
{
    V  = 0,
    E  = 1,
    F  = 2,
    VV = 3,
    VE = 4,
    VF = 5,
    FV = 6,
    FE = 7,
    FF = 8,
    EV = 9,
    EE = 10,
    EF = 11,
};


/**
 * @brief different dynaimc operators supported in RXMeshDynamic
 */
enum class DynOp
{
    EdgeFlip     = 0,
    DeleteVertex = 1,
    DeleteEdge   = 2,
    DeleteFace   = 3,
    EdgeCollapse = 4,
};
/**
 * @brief Convert an operation to string
 * @param op a query operation
 * @return name of the query operation as a string
 */
static std::string op_to_string(const Op& op)
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
            RXMESH_ERROR("to_string() unknown input operation");
            return "";
        }
    }
}
}  // namespace rxmesh