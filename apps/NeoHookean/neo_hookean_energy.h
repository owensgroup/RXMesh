#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT,
          typename VAttrI,
          typename VAttrF,
          typename VAttrFM,
          typename T>
void neo_hookean_energy(ProblemT&      problem,
                        const VAttrI&  is_dbc,
                        const VAttrF&  volume,
                        const VAttrFM& inv_b,
                        const T        mu_lame,
                        const T        h,  // time step
                        const T        lam)
{

    constexpr int dim = 3;

    const T h_sq = h * h;

    problem.template add_term<Op::FV, true>(
        [=] __device__(const auto& fh, const auto& iter, auto& obj) mutable {
            assert(iter.size() == 3);

            using ActiveT = ACTIVE_TYPE(fh);

            if (is_dbc(iter[0]) || is_dbc(iter[1]) || is_dbc(iter[2])) {
                return ActiveT();
            }

            Eigen::Vector3<ActiveT> x0 = iter_val<ActiveT, 3>(fh, iter, obj, 0);
            Eigen::Vector3<ActiveT> x1 = iter_val<ActiveT, 3>(fh, iter, obj, 1);
            Eigen::Vector3<ActiveT> x2 = iter_val<ActiveT, 3>(fh, iter, obj, 2);

            Eigen::Vector3<ActiveT> e0 = x2 - x0;
            Eigen::Vector3<ActiveT> e1 = x1 - x0;

            Eigen::Vector3<ActiveT> n = e0.cross(e1);
            n.normalize();

            Eigen::Matrix<ActiveT, 3, 3> f = col_mat(e0, e1, n);

            Eigen::Matrix<T, 3, 3> ib = inv_b(fh);

            // F is the deformation gradient
            Eigen::Matrix<ActiveT, 3, 3> F = f * ib;

            ActiveT J = F.determinant();

            ActiveT lnJ = log(J);

            Eigen::Matrix<ActiveT, 3, 3> FtF = (F.transpose() * F);


            // psi is energy density function
            ActiveT psi = T(0.5) * mu_lame * (FtF.trace() - dim) -
                          mu_lame * lnJ + T(0.5) * lam * lnJ * lnJ;

            ActiveT E = volume(fh) * psi * h_sq;


            return E;
        });
}

template <typename DenseMatT,
          typename VAttrT,
          typename T = typename DenseMatT::Type>
T neo_hookean_step_size(RXMeshStatic&    rx,
                        const VAttrT&    x,
                        const DenseMatT& dir,
                        DenseMatT&       alpha)
{

    alpha.reset(T(1), DEVICE);

    constexpr uint32_t blockThreads = 256;

    rx.run_query_kernel<Op::FV, blockThreads>(
        [=] __device__(const FaceHandle&     fh,
                       const VertexIterator& iter) mutable {
            const Eigen::Vector3<T> x0 = x.to_eigen<3>(iter[0]);
            const Eigen::Vector3<T> x1 = x.to_eigen<3>(iter[1]);
            const Eigen::Vector3<T> x2 = x.to_eigen<3>(iter[2]);

            const Eigen::Vector3<T> p0(
                dir(iter[0], 0), dir(iter[0], 1), dir(iter[0], 2));
            const Eigen::Vector3<T> p1(
                dir(iter[1], 0), dir(iter[1], 1), dir(iter[1], 2));
            const Eigen::Vector3<T> p2(
                dir(iter[2], 0), dir(iter[2], 1), dir(iter[2], 2));

            const Eigen::Vector3<T> x21 = x1 - x0;
            const Eigen::Vector3<T> x31 = x2 - x0;

            const Eigen::Vector3<T> p21 = p1 - p0;
            const Eigen::Vector3<T> p31 = p2 - p0;


            auto smallest_positive_real_root_quad =
                [=](T a, T b, T c, T tol = static_cast<T>(1e-6)) {
                    // return negative value if no positive real root is found
                    T t = 0;
                    if (std::abs(a) <= tol) {
                        if (std::abs(b) <= tol) {  // f(x) = c > 0 for all x
                            t = -1;
                        } else {
                            t = -c / b;
                        }
                    } else {
                        T desc = b * b - 4 * a * c;
                        if (desc > 0) {
                            t = (-b - std::sqrt(desc)) / (2 * a);
                            if (t < 0) {
                                t = (-b + std::sqrt(desc)) / (2 * a);
                            }
                        } else {  // desc<0 ==> imag, f(x) > 0 for all x > 0
                            t = -1;
                        }
                    }
                    return t;
                };

            T detT = x21.cross(x31).norm();

            T a = p21.cross(p31).norm() / detT;

            T b = (x21.cross(p31).norm() + p21.cross(x31).norm()) / detT;

            // solve for alpha that first brings the new volume to 0.1x the old
            // volume for slackness
            T c = static_cast<T>(0.9);

            T critical_alpha = smallest_positive_real_root_quad(a, b, c);

            if (critical_alpha > 0) {
                alpha(fh) = std::min(alpha(fh), critical_alpha);
            }
        });

    return alpha.abs_min();
}