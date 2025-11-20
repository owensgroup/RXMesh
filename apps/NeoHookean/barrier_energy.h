#pragma once

#include "rxmesh/diff/candidate_pairs.h"
#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename ProblemT,
          typename VAttrT,
          typename VAttrI,
          typename T = typename VAttrT::Type>
void floor_barrier_energy(ProblemT&      problem,
                          VAttrT&        contact_area,
                          const VAttrT&  x,
                          const T        h,  // time_step
                          const VAttrI&  is_dbc,
                          const vec3<T>& ground_n,
                          const vec3<T>& ground_o,
                          const T        dhat,
                          const T        kappa)
{

    const T h_sq = h * h;

    const Eigen::Vector3<T> o(ground_o[0], ground_o[1], ground_o[2]);
    const Eigen::Vector3<T> n(ground_n[0], ground_n[1], ground_n[2]);

    const Eigen::Vector3<T> normal(0.0, -1.0, 0.0);

    problem.template add_term<Op::V, true>(
        [=] __device__(const auto& vh, auto& obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            const Eigen::Vector3<ActiveT> xi = iter_val<ActiveT, 3>(vh, x);

            ActiveT E;


            if (!is_dbc(vh)) {
                // floor
                ActiveT d = (xi - o).dot(n);
                if (d < dhat) {
                    ActiveT s = d / dhat;

                    if (s <= T(0)) {
                        using PassiveT = PassiveType<ActiveT>;
                        return ActiveT(std::numeric_limits<PassiveT>::max());
                    }

                    E = h_sq * contact_area(vh) * dhat * T(0.5) * kappa *
                        (s - 1) * log(s);
                }
            }

            return E;
        });
}

template <typename VAttrT, typename VAttrB, typename T = typename VAttrT::Type>
void add_contact(RXMeshStatic&      rx,
                 CandidatePairsVV&  contact_pairs,
                 const VertexHandle dbc_vertex,
                 const VAttrB&      is_dbc,
                 const VAttrT&      x,
                 const T            dhat)
{
    contact_pairs.reset();

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        if (vh == dbc_vertex || is_dbc(vh)) {
            return;
        }
        const Eigen::Vector3<T> x_dbc = x.template to_eigen<3>(dbc_vertex);
        const Eigen::Vector3<T> xi    = x.template to_eigen<3>(vh);
        const Eigen::Vector3<T> normal(0.0, -1.0, 0.0);

        T d = (xi - x_dbc).dot(normal);

        if (d < dhat) {
            contact_pairs.insert(vh, dbc_vertex);        
        }
    });
}

template <typename ProblemT,
          typename VAttrT,
          typename T = typename VAttrT::Type>
void ceiling_barrier_energy(ProblemT&      problem,
                            VAttrT&        contact_area,
                            const VAttrT&  x,
                            const T        h,  // time_step
                            const vec3<T>& ground_n,
                            const vec3<T>& ground_o,
                            const T        dhat,
                            const T        kappa)
{
    const T h_sq = h * h;

    const Eigen::Vector3<T> normal(0.0, -1.0, 0.0);

    problem.template add_term<true>([=] __device__(const auto& id,
                                                   const auto& iter,
                                                   const auto& obj) mutable {
        using ActiveT = ACTIVE_TYPE(id);

        const VertexHandle c0 = iter[0];

        const VertexHandle c1 = iter[1];

        //???
        const Eigen::Vector3<ActiveT> xi = iter_val<ActiveT, 3>(id, iter, x, 0);

        const Eigen::Vector3<T> x_dbc = x.template to_eigen<3>(c1);


        // ceiling
        ActiveT d = (xi - x_dbc).dot(normal);

        assert(d < dhat);

        // if (d < dhat) {
        ActiveT s = d / dhat;

        ActiveT E =
            h_sq * contact_area(c0) * dhat * T(0.5) * kappa * (s - 1) * log(s);


        return E;
    });
}

template <typename VAttrT,
          typename VAttrI,
          typename DenseMatT,
          typename T = typename VAttrT::Type>
T barrier_step_size(RXMeshStatic&      rx,
                    const DenseMatT&   search_dir,
                    DenseMatT&         alpha,
                    const VertexHandle dbc_vertex,
                    const VAttrT&      x,
                    const VAttrI&      is_dbc,
                    const vec3<T>&     ground_n,
                    const vec3<T>&     ground_o)
{
    alpha.reset(T(1), DEVICE);

    const vec3<T> n(0.0, -1.0, 0.0);

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        const vec3<T> p_dbc(search_dir(dbc_vertex, 0),
                            search_dir(dbc_vertex, 1),
                            search_dir(dbc_vertex, 2));

        const vec3<T> x_dbc = x.to_glm<3>(dbc_vertex);


        const vec3<T> pi(
            search_dir(vh, 0), search_dir(vh, 1), search_dir(vh, 2));

        const vec3<T> xi = x.to_glm<3>(vh);

        // floor
        T p_n = glm::dot(pi, ground_n);
        if (p_n < 0) {
            alpha(vh) = std::min(
                alpha(vh), T(0.9) * glm::dot(ground_n, (xi - ground_o)) / -p_n);
        }

        // ceiling
        //TODO this should be generalized 
        if (!is_dbc(vh)) {
            p_n = glm::dot(n, (pi - p_dbc));
            if (p_n < 0) {
                alpha(vh) = std::min(alpha(vh),
                                     T(0.9) * glm::dot(n, (xi - x_dbc)) / -p_n);
            }
        }
    });

    // we want the min here but since the min value is greater than 1 (y_ground
    // is less than 0, and search_dir is also less than zero)
    return alpha.abs_min();
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