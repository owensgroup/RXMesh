#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT, typename EAttrT, typename T>
void mass_spring_energy(ProblemT& problem, EAttrT& rest_l, T h, T k)
{
    T half_k_times_h_sq = T(0.5) * k * h * h;
    problem.template add_term<Op::EV, true>(
        [rest_l, half_k_times_h_sq] __device__(
            const auto& eh, const auto& iter, auto& opt_var) mutable {
            assert(iter[0].is_valid() && iter[1].is_valid());

            assert(iter.size() == 2);

            using ActiveT = ACTIVE_TYPE(eh);

            const Eigen::Vector3<ActiveT> a =
                opt_var.template active<3>(eh, iter, 0);
            const Eigen::Vector3<ActiveT> b =
                opt_var.template active<3>(eh, iter, 1);

            const T r = rest_l(eh);

            const Eigen::Vector3<ActiveT> diff = a - b;

            const ActiveT ratio = diff.squaredNorm() / r;

            const ActiveT s = (ratio - T(1.0));

            const ActiveT E = half_k_times_h_sq * r * s * s;

            return E;
        });
}