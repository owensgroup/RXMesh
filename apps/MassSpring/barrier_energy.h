#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT,
          typename VAttrT,
          typename T = typename VAttrT::Type>
void barrier_energy(ProblemT& problem,
                    VAttrT&   x,
                    T         contact_area,
                    const T   h,
                    const T   y_ground,
                    const T   dhat  = T(0.01),
                    const T   kappa = T(1e5))
{

    const T h_sq = h * h;

    problem.template add_term<Op::V, true>(
        [x, contact_area, h_sq, y_ground, dhat, kappa] __device__(
            const auto& vh, auto& obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            Eigen::Vector3<ActiveT> xx = iter_val<ActiveT, 3>(vh, obj);

            ActiveT E(T(0));

            ActiveT d = xx[1] - y_ground;

            if (d < dhat) {
                const ActiveT s = d / dhat;

                E = h_sq * contact_area * dhat * T(0.5) * kappa * (s - T(1)) *
                    log(s);
            }

            return E;
        });
}

template <typename ProblemT,
          typename VAttrT,
          typename T = typename VAttrT::Type>
void barrier_energy(ProblemT&               problem,
                    VAttrT&                 x,
                    T                       contact_area,
                    const T                 h,
                    const T                 sphere_radius_sq,
                    const Eigen::Vector3<T> sphere_center,
                    const T                 dhat  = T(0.01),
                    const T                 kappa = T(1e5))
{
    const T h_sq = h * h;

    const T r = std::sqrt(sphere_radius_sq);

    problem.template add_term<Op::V, true>([=] __device__(const auto& vh,
                                                          auto& obj) mutable {
        using ActiveT  = ACTIVE_TYPE(vh);
        using PassiveT = PassiveType<ActiveT>;

        Eigen::Vector3<ActiveT> xx = iter_val<ActiveT, 3>(vh, obj);

        Eigen::Vector3<ActiveT> c;
        c << ActiveT(sphere_center.x()), ActiveT(sphere_center.y()),
            ActiveT(sphere_center.z());

        Eigen::Vector3<ActiveT> diff = xx - c;
        ActiveT                 d2   = diff.squaredNorm();


        const T rp = r + dhat;
        if (d2 < ActiveT(rp * rp)) {

            ActiveT d   = sqrt(d2);
            ActiveT gap = d - ActiveT(r);  // signed distance to sphere surface

            if (gap < ActiveT(dhat)) {
                ActiveT s = gap / ActiveT(dhat);

                if (s <= ActiveT(0)) {
                    return ActiveT(std::numeric_limits<PassiveT>::max());
                }

                ActiveT E =
                    ActiveT(h_sq * contact_area * dhat * T(0.5) * kappa) *
                    (s - ActiveT(1)) * log(s);

                return E;
            }
        }

        return ActiveT(0);
    });
}


template <typename VAttrT,
          typename DenseMatT,
          typename T = typename VAttrT::Type>
T init_step_size(RXMeshStatic&    rx,
                 const DenseMatT& search_dir,
                 DenseMatT&       alpha,
                 const VAttrT&    x,
                 const T          y_ground)
{
    alpha.reset(T(1), DEVICE);

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        if (search_dir(vh, 1) < 0) {
            alpha(vh) = std::min(
                alpha(vh), T(0.9) * (y_ground - x(vh, 1)) / search_dir(vh, 1));
        }
    });

    // we want the min here but since the min value is greater than 1 (y_ground
    // is less than 0, and search_dir is also less than zero)
    return alpha.abs_min();
}


template <typename VAttrT,
          typename DenseMatT,
          typename T = typename VAttrT::Type>
T init_step_size(RXMeshStatic&           rx,
                 const DenseMatT&        search_dir,
                 DenseMatT&              alpha,
                 const VAttrT&           x,
                 const T                 sphere_radius_sq,
                 const Eigen::Vector3<T> sphere_center)
{
    alpha.reset(T(1), DEVICE);

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        Eigen::Vector3<T> p = x.to_eigen<3>(vh);
        Eigen::Vector3<T> d;
        d[0] = search_dir(vh, 0);
        d[1] = search_dir(vh, 1);
        d[2] = search_dir(vh, 2);

        const T dd = d.dot(d);
        if (dd <= T(0)) {
            return;
        }

        const Eigen::Vector3<T> m = p - sphere_center;

        // Solve ||m + a d||^2 = r^2
        // => (d·d) a^2 + 2 (m·d) a + (m·m - r^2) = 0
        const T b    = T(2) * m.dot(d);
        const T c    = m.dot(m) - sphere_radius_sq;
        const T disc = b * b - T(4) * dd * c;


        if (disc <= T(0)) {
            return;
        }

        const T sqrt_disc = sqrt(disc);

        const T a0 = (-b - sqrt_disc) / (T(2) * dd);
        const T a1 = (-b + sqrt_disc) / (T(2) * dd);


        T a_hit = T(1);
        if (a0 > T(0)) {
            a_hit = a0;
        } else if (a1 > T(0)) {
            a_hit = a1;
        }

        else {
            return;
        }

        alpha(vh) = min(alpha(vh), T(0.9) * a_hit);
    });

    return alpha.abs_min();
}