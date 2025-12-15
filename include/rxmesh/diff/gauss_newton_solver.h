#pragma once

#include "rxmesh/diff/diff_vector_problem.h"

#include "rxmesh/matrix/cg_solver.h"
#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/matrix/cudss_cholesky_solver.h"
#include "rxmesh/matrix/pcg_solver.h"


namespace rxmesh {

template <typename T, int VariableDim, typename ObjHandleT>
struct GaussNetwtonSolver
{

    using DiffProblemT = DiffVectorProblem<T, VariableDim, ObjHandleT>;
    using SpMatT       = SparseMatrix<T>;
    using DenseMatT    = typename DiffProblemT::DenseMatT;
    using SolverT      = cuDSSCholeskySolver<SpMatT, DenseMatT::OrderT>;

    DiffProblemT& problem;
    DenseMatT     dir;
    SolverT       solver;
    SpMatT        JtJ;

    bool is_prep_solver_called;

    float solve_time;

    /**
     * @brief Newton solver
     */
    GaussNetwtonSolver(DiffProblemT& p)
        : problem(p), solve_time(0), is_prep_solver_called(false)
    {
        dir.reset(0, LOCATION_ALL);
    }

    /**
     * @brief should be called only after prep_eval in the given problem
     * (DiffVectorProblem)
     */
    void prep_solver()
    {
        is_prep_solver_called = true;

        // allocate the direction
        dir = DenseMatT(problem.rx, problem.jac->cols(), 1, LOCATION_ALL);

        // create JtJ
        create_JtJ();

        // create an instance of SolverT (solver)
        solver = SolverT(&JtJ);
        solver.pre_solve(problem.rx, *problem.grad, dir);
    }

    /**
     * @brief solve to get a new direction. prep_solver should be called before
     * calling compute direction to get accurate timing
     */
    inline void compute_direction(cudaStream_t stream = NULL)
    {
        if (!is_prep_solver_called) {
            prep_solver();
        }

        // TODO compute JtJ
        create_JtJ();

        // solve for new direction

        GPUTimer timer;
        timer.start();

        solver.pre_solve(problem.rx, dir, *problem.grad);
        solver.solve(*problem.grad, dir, stream);


        timer.stop();
        solve_time += timer.elapsed_millis();
    }


   private:
    void create_JtJ()
    {
       
    }
};

}  // namespace rxmesh