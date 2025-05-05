#pragma once

#include "rxmesh/diff/diff_scalar_problem.h"

#include "rxmesh/diff/armijo_condition.h"

#include "rxmesh/matrix/cg_mat_free_solver.h"
#include "rxmesh/matrix/cg_solver.h"
#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/matrix/lu_solver.h"
#include "rxmesh/matrix/pcg_solver.h"
#include "rxmesh/matrix/qr_solver.h"

namespace rxmesh {

template <typename T, int VariableDim, typename ObjHandleT, typename SolverT>
struct NetwtonSolver
{

    using DiffProblemT = DiffScalarProblem<T, VariableDim, ObjHandleT, true>;
    using HessMatT     = typename DiffProblemT::HessMatT;
    using DenseMatT    = typename DiffProblemT::DenseMatT;

    DiffProblemT&                             problem;
    DenseMatT                                 dir;
    std::shared_ptr<Attribute<T, ObjHandleT>> temp_objective;
    SolverT*                                  solver;

    float solve_time;

    /**
     * @brief Newton solver
     */
    NetwtonSolver(DiffProblemT& p, SolverT* s)
        : problem(p),
          dir(DenseMatT(p.rx, p.grad.rows(), p.grad.cols())),
          temp_objective(
              p.rx.add_attribute_like("temp_objective", *p.objective)),
          solver(s),
          solve_time(0)
    {
        dir.reset(0, LOCATION_ALL);

        if constexpr (std::is_base_of_v<CGMatFreeSolver<T, DenseMatT::OrderT>,
                                        SolverT>) {

            solver->m_mat_vec =
                [&](const DenseMatT& in, DenseMatT& out, cudaStream_t stream) {
                    problem.eval_matvec(in, out, stream);
                };
        }
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
     * @brief solve to get Newton direction
     */
    inline void newton_direction(cudaStream_t stream = NULL)
    {
        problem.grad.multiply(T(-1.f), stream);


        // LU
        if constexpr (std::is_base_of_v<LUSolver<HessMatT, DenseMatT::OrderT>,
                                        SolverT>) {
            problem.grad.move(DEVICE, HOST, stream);
            problem.hess.move(DEVICE, HOST, stream);

            CPUTimer timer;
            timer.start();

            solver->pre_solve(problem.rx);
            solver->solve(problem.grad, dir);
            timer.stop();

            solve_time += timer.elapsed_millis();


            dir.move(HOST, DEVICE);
        }

        // Cholesky or QR
        if constexpr (std::is_base_of_v<
                          CholeskySolver<HessMatT, DenseMatT::OrderT>,
                          SolverT> ||
                      std::is_base_of_v<QRSolver<HessMatT, DenseMatT::OrderT>,
                                        SolverT>) {

            GPUTimer timer;
            timer.start();

            // solver->solve_hl_api(problem.grad, dir);

            solver->pre_solve(problem.rx);
            solver->solve(problem.grad, dir);

            timer.stop();
            solve_time += timer.elapsed_millis();
        }

        //  Iterative Solver (CG/PCG) either matrix-based or matrix-free
        if constexpr (std::is_base_of_v<CGSolver<T, DenseMatT::OrderT>,
                                        SolverT>) {

            int r = problem.grad.rows();
            int c = problem.grad.cols();

            GPUTimer timer;
            timer.start();

            problem.grad.reshape(r * c, 1);
            dir.reshape(r * c, 1);

            solver->pre_solve(problem.grad, dir);
            solver->solve(problem.grad, dir);

            timer.stop();
            solve_time += timer.elapsed_millis();

            // RXMESH_INFO(
            //     "Init residual = {}, final residual {}, #Iter taken= {}",
            //     solver->start_residual(),
            //     solver->final_residual(),
            //     solver->iter_taken());

            problem.grad.reshape(r, c);
            dir.reshape(r, c);
        }
    }


    /**
     * @brief line search
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
            problem.rx.template for_each<ObjHandleT>(
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
            problem.rx.template for_each<ObjHandleT>(
                DEVICE,
                [t_obj = *temp_objective,
                 obj   = *problem.objective] __device__(const ObjHandleT& h) {
                    for (int j = 0; j < t_obj.get_num_attributes(); ++j) {
                        obj(h, j) = t_obj(h, j);
                    }
                });
        }
    }


    /**
     * @brief apply boundary condition on the system by doing the following to
     *  the elements corresponding to the boundary
     * 1) zeroing out off-diagonal elements in the Hessian
     * 2) setting the diagonal elements to 1
     * 3) zeroing out the gradient
     * @param bc is an attribute storing 1/true for boundary condition and
     * 0/false otherwise.
     */
    template <typename bcT>
    inline void apply_bc(Attribute<bcT, ObjHandleT>& bc)
    {
        auto g   = problem.grad;
        auto H   = problem.hess;
        int  dim = VariableDim;

        auto& ctx = problem.rx.get_context();

        problem.rx.template for_each<ObjHandleT>(
            DEVICE,
            [bc, g, H, dim, ctx] __device__(const ObjHandleT& h) mutable {
                if (int(bc(h)) == 1) {

                    int h_row_id = ctx.linear_id(h) * dim;

                    // for each row that belongs to this handle
                    for (int i = 0; i < dim; ++i) {
                        // TODO the stride here is 1 which might differ
                        // if we change the stride in the sparse matrix
                        int row_id = h_row_id + i;

                        int start = H.row_ptr()[row_id];
                        int stop  = H.row_ptr()[row_id + 1];

                        for (int j = start; j < stop; ++j) {
                            int col_id = H.col_idx()[j];

                            if (row_id == col_id) {
                                H.get_val_at(j) = T(1);
                            } else {
                                H.get_val_at(j) = T(0);

                                // maintain symmetry
                                // this may create race condition if col_id
                                // is also a boundary condition but here all
                                // threads write 0 so it should be fine
                                H(col_id, row_id) = T(0);
                            }
                        }
                    }


                    for (int i = 0; i < dim; ++i) {
                        g(h, i) = 0;
                    }
                }
            });
    }
};

}  // namespace rxmesh