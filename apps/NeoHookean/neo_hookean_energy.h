#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT,
          typename VAttrT,
          typename VAttrF,
          typename VAttrFM,
          typename T>
void neo_hookean_energy(ProblemT&      problem,
                        const VAttrT&  x,
                        const VAttrF&  volume,
                        const VAttrFM& inv_b,
                        const T        mu_lame,
                        const T        h,  // time step
                        const T        lam)
{

    constexpr int dim = 3;

    problem.template add_term<Op::FV, true>(
        [=] __device__(const auto& fh, const auto& iter, auto& obj) mutable {
            assert(iter.size() == 3);

            using ActiveT = ACTIVE_TYPE(fh);

            Eigen::Vector3<ActiveT> x0 = iter_val<ActiveT, 3>(fh, iter, obj, 0);
            Eigen::Vector3<ActiveT> x1 = iter_val<ActiveT, 3>(fh, iter, obj, 1);
            Eigen::Vector3<ActiveT> x2 = iter_val<ActiveT, 3>(fh, iter, obj, 2);


            Eigen::Matrix<ActiveT, 3, 2> f = col_mat(x1 - x0, x2 - x0);

            Eigen::Matrix<T, 2, 3> ib = inv_b(fh);

            // F is the deformation gradient
            Eigen::Matrix<ActiveT, 3, 3> F = f * ib;

            ActiveT J = F.determinant();

            ActiveT lnJ = log(J);

            Eigen::Matrix<ActiveT, 3, 3> FtF = (F.transpose() * F);


            // psi is energy density function
            ActiveT psi = T(0.5) * mu_lame * (FtF.trace() - dim) -
                          mu_lame * lnJ + T(0.5) * lam * lnJ * lnJ;

            ActiveT E = volume(fh) * psi * h * h;


            return E;
        });
}