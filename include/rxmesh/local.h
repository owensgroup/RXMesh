#pragma once
#include <stdint.h>
#include <string>
#include "rxmesh/util/macros.h"

namespace rxmesh {

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

}  // namespace rxmesh