#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename VAttrT, typename VAttrBCT>
void flag_bc(RXMeshStatic&    rx,
             VAttrBCT&        is_bc,
             VAttrT&          x,
             const glm::vec3& bb_lower,
             const glm::vec3& bb_upper)
{
    using T = typename VAttrT::Type;

    rx.for_each_vertex(
        DEVICE,
        [bb_upper, bb_lower, is_bc, x] __device__(const VertexHandle& vh) {
            if (x(vh, 0) < std::numeric_limits<T>::min()) {
                is_bc(vh) = 1;
            }
        });
}

template <typename T, typename VAttrT>
void apply_init_stretch(RXMeshStatic& rx, VAttrT& x, const T initial_stretch)
{
    rx.for_each_vertex(DEVICE,
                       [initial_stretch, x] __device__(const VertexHandle& vh) {
                           x(vh, 1) = x(vh, 1) * initial_stretch;
                       });
}