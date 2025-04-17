#pragma once
#include "rxmesh/matrix/iterative_solver.h"

#include "rxmesh/attribute.h"
#include "rxmesh/matrix/gmg/gmg.h"
#include "rxmesh/matrix/gmg/v_cycle.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/reduce_handle.h"

namespace rxmesh {

/**
 * @brief Geometric Multi Gird Solver
 */
template <typename T>
struct GMGSolver : public IterativeSolver<T, DenseMatrix<T>>
{
    using Type = T;

    GMGSolver(RXMeshStatic&     rx,
              SparseMatrix<T>& A,
              int               max_iter,
              int               num_pre_relax  = 2,
              int               num_post_relax = 2,
              CoarseSolver      coarse_solver  = CoarseSolver::Jacobi,
              T                 abs_tol        = 1e-6,
              T                 rel_tol        = 1e-6)
        : IterativeSolver<T, DenseMatrix<T>>(max_iter, abs_tol, rel_tol),
          m_rx(&rx),
          m_A(&A),
          m_coarse_solver(coarse_solver),
          m_num_pre_relax(num_pre_relax),
          m_num_post_relax(num_post_relax)
    {
    }

    virtual void pre_solve(DenseMatrix<T>&       X,
                           const DenseMatrix<T>& B,
                           cudaStream_t          stream = NULL) override
    {
        m_gmg = GMG<T>(*m_rx);

        m_v_cycle = VCycle<T>(m_gmg,
                              *m_rx,
                              *m_A,
                              B,
                              m_coarse_solver,
                              m_num_pre_relax,
                              m_num_post_relax);

        m_start_residual = m_v_cycle.m_r[0].norm2();
    }

    virtual void solve(DenseMatrix<T>&       X,
                       const DenseMatrix<T>& B,
                       cudaStream_t          stream = NULL) override
    {
        m_iter_taken = 0;
        while (m_iter_taken < m_max_iter) {
            m_v_cycle.cycle(0, m_gmg, *m_A, B, X);
            T current_res;
            current_res = m_v_cycle.m_r[0].norm2();


            if (is_converged(m_start_residual, current_res)) {
                m_final_residual = current_res;
                return;
            }

            m_iter_taken++;
        }
    }

    virtual std::string name() override
    {
        return std::string("GMG");
    }


    virtual ~GMGSolver()
    {
    }

   protected:
    RXMeshStatic*     m_rx;
    SparseMatrix<T>* m_A;
    GMG<T>            m_gmg;
    VCycle<T>         m_v_cycle;
    CoarseSolver      m_coarse_solver;
    int               m_num_pre_relax;
    int               m_num_post_relax;
};

}  // namespace rxmesh