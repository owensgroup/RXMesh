#pragma once

#include "rxmesh/diff/diff_scalar_problem.h"

#include "rxmesh/diff/armijo_condition.h"

namespace rxmesh {


template <typename T, int VariableDim, typename ObjHandleT>
struct LBFGSSolver
{

    using DiffProblemT = DiffScalarProblem<T, VariableDim, ObjHandleT, false>;
    using DenseMatT    = typename DiffProblemT::DenseMatT;

    DiffProblemT&                             problem;
    int                                       m;  // history size
    int                                       k;  // iteration count
    std::vector<DenseMatT>                    s_list;
    std::vector<DenseMatT>                    y_list;
    std::vector<T>                            rho_list;
    DenseMatT                                 dir, q, r;
    std::shared_ptr<Attribute<T, ObjHandleT>> temp_objective;

    LBFGSSolver(DiffProblemT& p, int history_size)
        : problem(p),
          m(history_size),
          k(0),
          dir(DenseMatT(p.rx, p.grad.rows(), p.grad.cols())),
          q(DenseMatT(p.rx, p.grad.rows(), p.grad.cols())),
          r(DenseMatT(p.rx, p.grad.rows(), p.grad.cols())),
          temp_objective(
              p.rx.add_attribute_like("temp_objective", *p.objective))
    {

        dir.reset(0, LOCATION_ALL);
        q.reset(0, LOCATION_ALL);
        r.reset(0, LOCATION_ALL);

        s_list.resize(m);
        y_list.resize(m);
        rho_list.resize(m, T(0));

        for (int i = 0; i < m; ++i) {
            s_list[i] = DenseMatT(p.rx, p.grad.rows(), p.grad.cols());
            y_list[i] = DenseMatT(p.rx, p.grad.rows(), p.grad.cols());
            s_list[i].reset(0, LOCATION_ALL);
            y_list[i].reset(0, LOCATION_ALL);
        }
    }

    inline void solve(cudaStream_t stream = NULL)
    {
        compute_direction(stream);
        line_search(stream);
    }

    inline void compute_direction(cudaStream_t stream = NULL)
    {
        q.copy_from(problem.grad, DEVICE, DEVICE, stream);  // q = grad
        q.multiply(T(-1.f), stream);                        // q = -grad


        std::vector<T> alpha(m, T(0));

        for (int i = 1; i <= std::min(k, m); ++i) {
            int idx    = (k - i) % m;
            alpha[idx] = rho_list[idx] * s_list[idx].dot(q, false, stream);
            q.axpy(y_list[idx], -alpha[idx], stream);  // q = -alpha * y + q
        }

        // Initial H0 = identity
        r.copy_from(q, DEVICE, DEVICE, stream);

        for (int i = 1; i <= std::min(k, m); ++i) {
            int idx  = (k - std::min(k, m) + i - 1) % m;
            T   beta = rho_list[idx] * y_list[idx].dot(r, false, stream);
            r.axpy(s_list[idx], alpha[idx] - beta, stream);
        }

        dir.copy_from(r, DEVICE, DEVICE, stream);
    }

    inline void update_history(cudaStream_t stream = NULL)
    {
        int idx = k % m;

        // s_k = x_{k+1} - x_k
        // s_list[idx].copy(*problem.objective);??
        // s_list[idx].axpy(-1.0, *temp_objective, stream);??

        // y_k = grad_{k+1} - grad_k
        y_list[idx].copy_from(problem.grad, DEVICE, DEVICE, stream);

        // problem.eval_gradient(temp_objective.get(), stream);??

        y_list[idx].axpy(problem.grad, T(-1.0), stream);

        T sy = s_list[idx].dot(y_list[idx], false, stream);
        if (sy > 1e-10) {
            rho_list[idx] = T(1) / sy;
        } else {
            rho_list[idx] = 0;
        }
    }

    inline void line_search(const T      s_max        = 1.0,
                            const T      shrink       = 0.8,
                            const int    max_iters    = 64,
                            const T      armijo_const = 1e-4,
                            cudaStream_t stream       = NULL)
    {

        const T current_f = problem.get_current_loss(stream);
        T       s         = s_max;
        bool    update    = false;

        for (int i = 0; i < max_iters; ++i) {
            problem.rx.template for_each<ObjHandleT>(
                DEVICE,
                [s,
                 dir   = dir,
                 t_obj = *temp_objective,
                 obj   = *problem.objective] __device__(const ObjHandleT& h) {
                    for (int j = 0; j < t_obj.get_num_attributes(); ++j) {
                        t_obj(h, j) = obj(h, j) + s * dir(h, j);
                    }
                });

            problem.eval_terms_passive(temp_objective.get(), stream);
            T f_new = problem.get_current_loss(stream);

            if (armijo_condition(
                    current_f, f_new, s, dir, problem.grad, armijo_const)) {
                update = true;
                break;
            }

            s *= shrink;
        }

        if (update) {
            update_history(stream);
            problem.rx.template for_each<ObjHandleT>(
                DEVICE,
                [t_obj = *temp_objective,
                 obj   = *problem.objective] __device__(const ObjHandleT& h) {
                    for (int j = 0; j < t_obj.get_num_attributes(); ++j) {
                        obj(h, j) = t_obj(h, j);
                    }
                });

            problem.eval_gradient(problem.objective.get(), stream);
            ++k;
        }
    }
};

}  // namespace rxmesh
