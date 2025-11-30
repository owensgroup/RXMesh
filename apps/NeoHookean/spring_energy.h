#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT,
          typename VAttrT,
          typename VAttrI,
          typename MatT,
          typename T>
void spring_energy(ProblemT&     problem,                   
                   const VAttrT& dbc_target,
                   const VAttrI& is_dbc,
                   const T       mass,
                   const MatT&   dbc_stiff)
{
    // obj here is x

    problem.template add_term<Op::V, true>([=] __device__(const auto& vh,
                                                          auto& obj) mutable {
        using ActiveT = ACTIVE_TYPE(vh);

        ActiveT E(T(0));

        if (is_dbc(vh) == 1) {
            const Eigen::Vector3<ActiveT> xi = iter_val<ActiveT, 3>(vh, obj);

            const Eigen::Vector3<T> x_target =
                dbc_target.template to_eigen<3>(vh);

            Eigen::Vector3<ActiveT> diff = xi - x_target;

            E = T(0.5) * dbc_stiff(0) * mass * diff.dot(diff);
        }

        return E;
    });
}