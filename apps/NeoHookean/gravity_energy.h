#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT, typename T>
void gravity_energy(ProblemT&     problem,                    
                    const T       h,
                    const T       mass)
{
    const Eigen::Vector3<T> g(0.0, -9.81, 0.0);

    const T neg_mass_times_h_sq = -mass * h * h;

    problem.template add_term<Op::V, true>([=] __device__(const auto& vh,
                                                          auto& obj) mutable {
        using ActiveT = ACTIVE_TYPE(vh);

        ActiveT E(T(0));

        Eigen::Vector3<ActiveT> xi = iter_val<ActiveT, 3>(vh, obj);

        E = neg_mass_times_h_sq * xi.dot(g);            
        return E;
    });
}