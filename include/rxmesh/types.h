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
}  // namespace rxmesh