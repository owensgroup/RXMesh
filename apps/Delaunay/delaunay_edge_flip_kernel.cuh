#pragma once
#include <stdint.h>
#include "rxmesh/context.h"
#include "rxmesh/attribute.h"


template <typename T, uint32_t blockThreads>
__global__ static void delaunay_edge_flip(const rxmesh::Context      context,
                                          rxmesh::VertexAttribute<T> coords)
{
}