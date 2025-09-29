#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT, typename VAttrT, typename T>
void inertial_energy(ProblemT& problem, VAttrT& x, T mass)
{
    T half_mass = T(0.5) * mass;
    problem.template add_term<Op::V, true>(
        [x, half_mass] __device__(const auto& vh, auto& obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            Eigen::Vector3<ActiveT> x_tilda = iter_val<ActiveT, 3>(vh, obj);

            Eigen::Vector3<T> xx = x.to_eigen<3>(vh);

            Eigen::Vector3<ActiveT> l = xx - x_tilda;

            ActiveT E = half_mass * l.squaredNorm();

            return E;
        });
}