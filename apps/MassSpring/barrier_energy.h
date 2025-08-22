#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT,
          typename VAttrT,
          typename T = typename VAttrT::Type>
void barrier_energy(ProblemT& problem,
                    VAttrT&   x,
                    VAttrT&   contact_area,
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

            ActiveT E;

            ActiveT d = xx[1] - y_ground;

            if (d < dhat) {
                const ActiveT s = d / dhat;

                E = h_sq * kappa * contact_area(vh) * dhat / 2 * (s - 1) *
                    log(s);

                // if constexpr (std::is_base_of_v<ActiveT, Scalar<T, 3, true>>)
                // {
                //     // E.print();
                // }
            }

            return E;
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

    rx.for_each_vertex(DEVICE,
                       [search_dir, x, alpha, y_ground] __device__(
                           const VertexHandle& vh) mutable {
                           if (search_dir(vh, 1) < 0) {
                               alpha(vh) =
                                   std::min(alpha(vh),
                                            T(0.9) * (y_ground - x(vh, 1)) /
                                                search_dir(vh, 1));
                           }
                       });

    // we want the min here but since the min value is greater than 1 (y_ground
    // is less than 0, and search_dir is also less than zero)
    return alpha.abs_min();
}