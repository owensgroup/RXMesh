#pragma once

#include "rxmesh/diff/diff_scalar_problem.h"

#include "rxmesh/diff/armijo_condition.h"

namespace rxmesh {

template <typename T, int VariableDim, typename ObjHandleT>
struct NetwtonSolver
{
    DiffScalarProblem<T, VariableDim, ObjHandleT>& problem;
    DenseMatrix<T, Eigen::RowMajor>                dir;
    std::shared_ptr<Attribute<T, ObjHandleT>>      temp_objective;
    Solver                                         solver;


    /**
     * @brief
     */
    NetwtonSolver(DiffScalarProblem<T, VariableDim, ObjHandleT>& p, Solver s)
        : problem(p),
          dir(DenseMatrix<T, Eigen::RowMajor>(p.rx,
                                              p.grad.rows(),
                                              p.grad.cols())),
          temp_objective(
              p.rx.add_attribute_like("temp_objective", *p.objective)),
          solver(s)
    {
        // TODO
        // hess.pre_solve(rx, Solver::CHOL);

        dir.reset(0, LOCATION_ALL);
    }

    /**
     * @brief
     */
    inline void solve(cudaStream_t stream = NULL)
    {
        newton_direction(stream);
        line_search(stream);
    }

    /**
     * @brief
     */
    inline void newton_direction(cudaStream_t stream = NULL)
    {
        // TODO we should refactor (or at least analyze_pattern) once
        problem.grad.multiply(T(-1.f), stream);

        if (solver == Solver::CHOL || solver == Solver::QR) {
            problem.hess.solve(
                problem.grad.data(), dir.data(), solver, PermuteMethod::NSTDIS);
        } else if (solver == Solver::LU) {
            problem.grad.move(DEVICE, HOST);
            problem.hess.move(DEVICE, HOST);
            problem.hess.solve(problem.grad.data(HOST),
                               dir.data(HOST),
                               solver,
                               PermuteMethod::NSTDIS);
            dir.move(HOST, DEVICE);
        }
    }


    /**
     * @brief
     */
    inline void line_search(const T      s_max        = 1.0,
                            const T      shrink       = 0.8,
                            const int    max_iters    = 64,
                            const T      armijo_const = 1e-4,
                            cudaStream_t stream       = NULL)
    {
        // we are going to keep trying to update temp_objective until we reach
        // solution we are satisfied with, then we will copy it to sol. If no
        // good solution found, then sol will not be updated.

        assert(dir.rows() == problem.grad.rows());
        assert(dir.cols() == problem.grad.cols());
        assert(problem.objective->rows() == problem.grad.rows());
        assert(problem.objective->cols() == problem.grad.cols());
        assert(problem.objective->rows() == temp_objective->rows());
        assert(problem.objective->cols() == temp_objective->cols());

        assert(s_max > 0.0);

        const bool try_one = s_max > 1.0;

        T s = s_max;

        bool update = false;

        const T current_f = problem.get_current_loss(stream);

        for (int i = 0; i < max_iters; ++i) {

            // update solution
            problem.rx.for_each<ObjHandleT>(
                DEVICE,
                [s     = s,
                 dir   = dir,
                 t_obj = *temp_objective,
                 obj   = *problem.objective] __device__(const ObjHandleT& h) {
                    for (int j = 0; j < t_obj.get_num_attributes(); ++j) {
                        t_obj(h, j) = obj(h, j) + s * dir(h, j);
                    }
                });


            // eval new obj func
            problem.eval_terms_passive(temp_objective.get(), stream);

            // get the new value of the objective function
            T f_new = problem.get_current_loss(stream);

            if (armijo_condition(
                    current_f, f_new, s, dir, problem.grad, armijo_const)) {
                update = true;
                break;
            }

            if (try_one && s > 1.0 && s * shrink < 1.0) {
                s = 1.0;
            } else {
                s *= shrink;
            }
        }

        if (update) {
            problem.rx.for_each<ObjHandleT>(
                DEVICE,
                [t_obj = *temp_objective,
                 obj   = *problem.objective] __device__(const ObjHandleT& h) {
                    for (int j = 0; j < t_obj.get_num_attributes(); ++j) {
                        obj(h, j) = t_obj(h, j);
                    }
                });
        }
    }
};

}  // namespace rxmesh