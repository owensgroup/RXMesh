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

template <typename T, typename VAttrT, typename VAttrB>
void update_dbc(RXMeshStatic& rx,
                const VAttrB& is_dbc,
                const VAttrT& x,
                const vec3<T> v_dbc_vel,
                const vec3<T> v_dbc_limit,
                const T       h,
                VAttrT&       dbc_target)
{
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        if (is_dbc(vh)) {

            const vec3<T> xi = x.to_glm<3>(vh);

            if (glm::dot(v_dbc_limit - xi, v_dbc_vel) > 0) {
                const vec3<T> t = xi + h * v_dbc_vel;
                dbc_target.from_glm(vh, t);
            } else {
                dbc_target.from_glm(vh, xi);
            }
        }
    });
}

template <typename VAttrT, typename VAttrB, typename VAttrI, typename T>
void check_dbc_satisfied(RXMeshStatic& rx,
                         VAttrI&       is_dbc_satisfied,
                         const VAttrT& x,
                         const VAttrB& is_dbc,
                         const VAttrT& dbc_target,
                         const T       h,
                         const T       tol)
{
    is_dbc_satisfied.reset(0, DEVICE);

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        if (is_dbc(vh)) {

            const vec3<T> xi = x.to_glm<3>(vh);

            const vec3<T> x_target = dbc_target.to_glm<3>(vh);

            if (glm::distance(xi, x_target) / h < tol) {
                is_dbc_satisfied(vh) = true;
            }
        }
    });
}