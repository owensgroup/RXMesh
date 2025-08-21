#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT, typename VAttrT, typename T>
void gravity_energy(ProblemT& problem, VAttrT& x, T h, T mass)
{
    const Eigen::Vector3<T> g(0.0, -9.81, 0.0);

    const T neg_mass_times_h_sq = -mass * h * h;

    problem.template add_term<Op::V, true>(
        [x, neg_mass_times_h_sq, g] __device__(const auto& vh,
                                               auto&       obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            Eigen::Vector3<ActiveT> x_tilda = iter_val<ActiveT, 3>(vh, obj);

            ActiveT E = neg_mass_times_h_sq * x_tilda.dot(g);

            return E;
        });
}