#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT, typename VAttrT, typename DenseMatT, typename T>
void friction_energy(ProblemT&        problem,                     
                     const VAttrT&    x_n,
                     const DenseMatT& p,
                     const T&         alpha,
                     const VAttrT&    mu_lambda,
                     const T          h,
                     const vec3<T>&   ground_n)
{
    // TODO alpha should change for different runs (e.g., line search)
    const Eigen::Vector3<T> n(ground_n[0], ground_n[1], ground_n[2]);

    const Eigen::Matrix3<T> tangent =
        Eigen::Matrix3<T>::Identity() - n * n.transpose();

    const T h_sq = h * h;

    problem.template add_term<Op::V, true>(
        [=] __device__(const auto& vh, auto& obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            auto f0 = [&](ActiveT vbarnorm, T epsv, T hhat) {
                if (vbarnorm >= epsv) {
                    return vbarnorm * hhat;
                } else {

                    ActiveT vbarnormhhat = vbarnorm * hhat;

                    T epsvhhat = epsv * hhat;

                    return vbarnormhhat * vbarnormhhat *
                               (-vbarnormhhat / T(3.0) + epsvhhat) /
                               (epsvhhat * epsvhhat) +
                           epsvhhat / T(3.0);
                }
            };

            ActiveT E(T(0));

            T ml = mu_lambda(vh);
            if (ml > 0) {
                // this part is kinda annoying. The user should be able to send
                //  v = (x - xn) / h as a one vertex attribute but since
                // we need to lift x to active variable while xn is not, the
                // user sends each term and then do the subtraction and division
                // by hand, otherwise the derivative won't be computed correctly
                // (?)

                const Eigen::Vector3<ActiveT> xi = iter_val<ActiveT, 3>(vh, obj);

                const Eigen::Vector3<T> xi_n = x_n.template to_eigen<3>(vh);

                const Eigen::Vector3<T> pi(p(vh, 0), p(vh, 1), p(vh, 2));

                const Eigen::Vector3<ActiveT> vi = (xi + alpha * pi - xi_n) / h;

                const Eigen::Vector3<ActiveT> vbar = tangent.transpose() * vi;

                constexpr T epsv = 1e-3;

                E = ml * h_sq * f0(vbar.norm(), epsv, h);
            }

            return E;
        });
}

template <typename VAttrT, typename T = typename VAttrT::Type>
void compute_mu_lambda(RXMeshStatic&  rx,
                       const T        fricition_coef,
                       const T        dhat,
                       const T        kappa,
                       const vec3<T>& ground_n,
                       const vec3<T>& ground_o,
                       const VAttrT&  x,
                       const VAttrT&  contact_area,
                       VAttrT&        mu_lambda)
{
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        const vec3<T> xi = x.to_glm<3>(vh);

        T d = glm::dot(ground_n, (xi - ground_o));

        if (d < dhat) {
            T s = d / dhat;

            mu_lambda(vh) = fricition_coef * -contact_area(vh) * dhat *
                            (kappa / 2 * (log(s) / dhat + (s - 1) / d));
        }
    });
}