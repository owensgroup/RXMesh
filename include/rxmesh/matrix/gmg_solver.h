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

    GMGSolver(RXMeshStatic&    rx,
              SparseMatrix<T>& A,
              int              max_iter,
              int              num_levels     = 0,
              int              num_pre_relax  = 2,
              int              num_post_relax = 2,
              CoarseSolver     coarse_solver  = CoarseSolver::Jacobi,
              T                abs_tol        = 1e-6,
              T                rel_tol        = 1e-6,
              int threshold=1000)
        : IterativeSolver<T, DenseMatrix<T>>(max_iter, abs_tol, rel_tol),
          m_rx(&rx),
          m_A(&A),
          m_coarse_solver(coarse_solver),
          m_num_pre_relax(num_pre_relax),
          m_num_post_relax(num_post_relax),
          m_num_levels(num_levels),
          m_threshold(threshold)
    {
    }

    virtual void pre_solve(const DenseMatrix<T>& B,
                           DenseMatrix<T>&       X,
                           cudaStream_t          stream = NULL) override
    {

        // if (m_num_levels > 1)
        m_gmg = GMG<T>(*m_rx, m_num_levels,m_threshold);
        if (m_num_levels == 1) 
        {

            exit(1);

        }
        else 
        {
            m_v_cycle = VCycle<T>(m_gmg,
                                  *m_rx,
                                  *m_A,
                                  B,
                                  m_coarse_solver,
                                  m_num_pre_relax,
                                  m_num_post_relax);
            constexpr int numCols = 3;
            assert(numCols == B.cols());
            m_v_cycle.template calc_residual<numCols>(
                m_v_cycle.m_a[0].a, X, B, m_v_cycle.m_r[0]);
            m_start_residual = m_v_cycle.m_r[0].norm2();
        }
    }

    virtual void solve(const DenseMatrix<T>& B,
                       DenseMatrix<T>&       X,
                       cudaStream_t          stream = NULL) override
    {
        T current_res;

        if (m_num_levels==1) {
            RXMESH_TRACE("GMG:Direct solve used");
            //m_coarse_solver.template solve<numCols>(m_A, B, X, 1000);
            return;
        }
        else {
            m_iter_taken = 0;
            while (m_iter_taken < m_max_iter) {
                m_v_cycle.cycle(0, m_gmg, *m_A, B, X,*m_rx);
                current_res = m_v_cycle.m_r[0].norm2();
                // std::cout << "\ncurrent residual: " << current_res;

                if (is_converged(m_start_residual, current_res)) {
                    m_final_residual = current_res;
                    /*std::cout << "\nconverged! at " << m_final_residual
                              << " from residual of " << m_start_residual;*/
                    RXMESH_TRACE("GMG: #number of iterations to solve: {}",
                                 m_iter_taken);
                    RXMESH_TRACE("GMG: final residual: {}", m_final_residual);


                    return;
                }

                m_iter_taken++;
            }
            RXMESH_TRACE(
                "GMG: Solver did not reach convergence criteria. Residual: {}",
                current_res);
            RXMESH_TRACE("GMG: #number of iterations to solve: {}",
                         m_iter_taken);
        }
    }

    virtual std::string name() override
    {
        return std::string("GMG");
    }

    void render_hierarchy()
    {
        m_gmg.render_hierarchy();
    }

    virtual ~GMGSolver()
    {
    }

   protected:
    RXMeshStatic*    m_rx;
    SparseMatrix<T>* m_A;
    GMG<T>           m_gmg;
    VCycle<T>        m_v_cycle;
    CoarseSolver     m_coarse_solver;
    int              m_num_pre_relax;
    int              m_num_post_relax;
    int              m_num_levels;
    int              m_threshold;
};

}  // namespace rxmesh