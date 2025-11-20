#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT, typename VAttrT, typename VAttrI, typename T>
void inertial_energy(ProblemT&     problem,
                     const VAttrT& x,
                     const VAttrT& x_tilde,
                     const VAttrI& is_dbc,
                     const T       mass)
{
    T half_mass = T(0.5) * mass;
    problem.template add_term<Op::V, true>(
        [=] __device__(const auto& vh, auto& obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            ActiveT E;

            if (is_dbc(vh) == 0) {

                Eigen::Vector3<ActiveT> xx = iter_val<ActiveT, 3>(vh, x);

                Eigen::Vector3<T> xx_tilde = x_tilde.to_eigen<3>(vh);

                Eigen::Vector3<ActiveT> l = xx - xx_tilde;

                E = half_mass * l.squaredNorm();
            }

            return E;
        });
}