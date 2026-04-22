#pragma once

#include "rxmesh/diff/diff_scalar_problem.h"

#include "rxmesh/diff/armijo_condition.h"

namespace rxmesh {


template <typename T, int VariableDim, typename OptVarHandleT>
struct LBFGSSolver
{

    using DiffProblemT =
        DiffScalarProblem<T, VariableDim, OptVarHandleT, false>;
    using DenseMatT = typename DiffProblemT::DenseMatT;

    DiffProblemT&                                problem;
    int                                          m;  // history size
    int                                          k;  // iteration count
    std::vector<DenseMatT>                       s_list;
    std::vector<DenseMatT>                       y_list;
    std::vector<T>                               rho_list;
    DenseMatT                                    dir, q, r;
    std::shared_ptr<Attribute<T, OptVarHandleT>> temp_opt_var;

    LBFGSSolver(DiffProblemT& p, int history_size)
        : problem(p),
          m(history_size),
          k(0),
          dir(DenseMatT(p.rx, p.grad.rows(), p.grad.cols(), LOCATION_ALL)),
          q(DenseMatT(p.rx, p.grad.rows(), p.grad.cols(), LOCATION_ALL)),
          r(DenseMatT(p.rx, p.grad.rows(), p.grad.cols(), LOCATION_ALL)),
          temp_opt_var(p.rx.add_attribute_like("temp_opt_var", *p.opt_var))
    {

        dir.reset(0, LOCATION_ALL);
        q.reset(0, LOCATION_ALL);
        r.reset(0, LOCATION_ALL);

        s_list.resize(m);
        y_list.resize(m);
        rho_list.resize(m, T(0));

        for (int i = 0; i < m; ++i) {
            s_list[i] =
                DenseMatT(p.rx, p.grad.rows(), p.grad.cols(), LOCATION_ALL);
            y_list[i] =
                DenseMatT(p.rx, p.grad.rows(), p.grad.cols(), LOCATION_ALL);
            s_list[i].reset(0, LOCATION_ALL);
            y_list[i].reset(0, LOCATION_ALL);
        }
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

        if (k == 0) {
            // Initial H0 = identity
            r.copy_from(q, DEVICE, DEVICE, stream);
        } else {
            // scaled identity matrix
            int last = (k - 1) % m;

            T sy    = (rho_list[last] > 0) ? (1.0 / rho_list[last]) : T(1.0);
            T yy    = y_list[last].dot(y_list[last], false, stream);
            T gamma = (yy > 1e-10) ? sy / yy : 1.0;

            r.copy_from(q, DEVICE, DEVICE, stream);
            r.multiply(gamma, stream);  // r = gamma * q
        }

        for (int i = 1; i <= std::min(k, m); ++i) {
            int idx = (k - std::min(k, m) + i - 1) % m;
            // beta = rho*y^T.r
            T beta = rho_list[idx] * y_list[idx].dot(r, false, stream);
            // r = (alpha-beta) * s + r
            r.axpy(s_list[idx], alpha[idx] - beta, stream);
        }

        dir.copy_from(r, DEVICE, DEVICE, stream);
        // dir.multiply(T(-1.f), stream);
    }

    inline void update_history(cudaStream_t stream = NULL)
    {
        // update history is called after temp_opt_var (x_{k+1}) is being
        // updated in line_search and before updating problem.opt_var (x_k)

        int idx = k % m;

        // s_k = x_{k+1} - x_k
        // s_k = temp_opt_var - problem.opt_var
        s_list[idx].reset(0, DEVICE);
        problem.rx.template for_each<OptVarHandleT>(
            DEVICE,
            [s    = s_list[idx],
             temp = *temp_opt_var,
             prev =
                 *problem.opt_var] __device__(const OptVarHandleT& h) mutable {
                for (int j = 0; j < s.cols(); ++j) {
                    s(h, j) = temp(h, j) - prev(h, j);
                }
            });

        // y_k = grad_k
        y_list[idx].copy_from(problem.grad, DEVICE, DEVICE, stream);

        // y_k = -grad_k
        y_list[idx].multiply(T(-1.f), stream);

        // update grad_{k+1}
        problem.eval_terms_grad_only(temp_opt_var.get(), stream);

        // y_k = grad_{k+1} - grad_k
        y_list[idx].axpy(problem.grad, T(1.0), stream);

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
        assert(dir.rows() == problem.grad.rows());
        assert(dir.cols() == problem.grad.cols());
        assert(problem.opt_var->rows() == problem.grad.rows());
        assert(problem.opt_var->cols() == problem.grad.cols());
        assert(problem.opt_var->rows() == temp_opt_var->rows());
        assert(problem.opt_var->cols() == temp_opt_var->cols());
        assert(s_max > 0.0);

        T s = s_max;

        bool update = false;

        const T current_f = problem.get_current_loss(stream);


        for (int i = 0; i < max_iters; ++i) {
            problem.rx.template for_each<OptVarHandleT>(
                DEVICE,
                [s,
                 dir       = dir,
                 t_opt_var = *temp_opt_var,
                 opt_var =
                     *problem.opt_var] __device__(const OptVarHandleT& h) {
                    for (int j = 0; j < t_opt_var.get_num_attributes(); ++j) {
                        t_opt_var(h, j) = opt_var(h, j) + s * dir(h, j);
                    }
                });

            problem.eval_terms_passive(temp_opt_var.get(), stream);
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
            problem.rx.template for_each<OptVarHandleT>(
                DEVICE,
                [t_opt_var = *temp_opt_var,
                 opt_var =
                     *problem.opt_var] __device__(const OptVarHandleT& h) {
                    for (int j = 0; j < t_opt_var.get_num_attributes(); ++j) {
                        opt_var(h, j) = t_opt_var(h, j);
                    }
                });
            ++k;
        }
    }
};

}  // namespace rxmesh
