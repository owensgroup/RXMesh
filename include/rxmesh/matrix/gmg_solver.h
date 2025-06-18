#pragma once
#include "rxmesh/matrix/iterative_solver.h"

#include "rxmesh/attribute.h"
#include "rxmesh/matrix/gmg/gmg.h"
#include "rxmesh/matrix/gmg/v_cycle.h"
#include "rxmesh/matrix/gmg/v_cycle_better_ptap.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/reduce_handle.h"

namespace rxmesh {

/**
 * @brief Geometric Multi Grid Solver
 */
template <typename T>
struct GMGSolver : public IterativeSolver<T, DenseMatrix<T>>
{
    using Type = T;
    float gmg_memory_alloc_time;
    float v_cycle_memory_alloc_time;

    GMGSolver(RXMeshStatic&    rx,
              SparseMatrix<T>& A,
              int              max_iter,
              int              num_levels     = 0,
              int              num_pre_relax  = 2,
              int              num_post_relax = 2,
              CoarseSolver     coarse_solver  = CoarseSolver::Jacobi,
              T                abs_tol        = 1e-6,
              T                rel_tol        = 1e-6,
              int              threshold      = 1000)
        : IterativeSolver<T, DenseMatrix<T>>(max_iter, abs_tol, rel_tol),
          m_rx(&rx),
          m_A(&A),
          m_coarse_solver(coarse_solver),
          m_num_pre_relax(num_pre_relax),
          m_num_post_relax(num_post_relax),
          m_num_levels(num_levels),
          m_threshold(threshold),
          AX(DenseMatrix<T>(A.rows(), 1, DEVICE)),
          R(DenseMatrix<T>(A.rows(), 1, DEVICE))
    {
    }

    virtual void pre_solve(const DenseMatrix<T>& B,
                           DenseMatrix<T>&       X,
                           cudaStream_t          stream = NULL) override
    {

        // if (m_num_levels > 1)

        CPUTimer timer;
        GPUTimer gtimer;

        timer.start();
        gtimer.start();
        m_gmg = GMG<T>(*m_rx, m_num_levels, m_threshold);
        timer.stop();
        gtimer.stop();
        RXMESH_INFO("full gmg operator construction took {} (ms), {} (ms)",
                    timer.elapsed_millis(),
                    gtimer.elapsed_millis());
        gmg_memory_alloc_time = m_gmg.memory_alloc_time;
        m_num_levels = m_gmg.m_num_levels;
        if (m_num_levels == 1) {

            exit(1);

        } else {

            timer.start();
            gtimer.start();
            m_v_cycle = VCycle<T>(m_gmg,
                                  *m_rx,
                                  *m_A,
                                  B,
                                  m_coarse_solver,
                                  m_num_pre_relax,
                                  m_num_post_relax);



            v_cycle_memory_alloc_time = m_v_cycle.memory_alloc_time;

            timer.stop();
            gtimer.stop();
            RXMESH_INFO("v cycle prep took {} (ms), {} (ms)",
                        timer.elapsed_millis(),
                        gtimer.elapsed_millis());

            constexpr int numCols = 3;
            assert(numCols == B.cols());
            m_v_cycle.template calc_residual<numCols>(
                m_v_cycle.m_a[0].a, X, B, m_v_cycle.m_r[0]);
            m_start_residual = m_v_cycle.m_r[0].norm2();
        }
    }


    template <typename T>
    bool is_converged_special_gpu(rxmesh::SparseMatrix<T>& A,
                                  rxmesh::DenseMatrix<T>&  X,
                                  rxmesh::DenseMatrix<T>&  B)
    {
        // using namespace rxmesh;
        using IndexT = typename DenseMatrix<T>::IndexT;

        // Move everything to device
        /*A.move(HOST, DEVICE);
        X.move(HOST, DEVICE);
        B.move(HOST, DEVICE);*/

        const IndexT num_rhs = B.cols();
        const IndexT n       = A.rows();

        T max_residual = 0.0;
        for (IndexT i = 0; i < num_rhs; ++i) {
            A.multiply(X.col_data(i, DEVICE), AX.col_data(0, DEVICE));
            R.copy_from(AX, DEVICE, DEVICE);


            R.axpy(B.col(i), T(-1));  // R = AX - B_i
            T r_norm = R.norm2();
            T b_norm = B.col(i).norm2();  // You'll need to add this utility

            T residual   = r_norm / (b_norm + 1e-20);
            max_residual = std::max(max_residual, residual);
        }

        bool abs_ok = max_residual < this->m_abs_tol;
        bool rel_ok = max_residual < this->m_rel_tol;

        if (abs_ok || rel_ok) {
            RXMESH_TRACE("GMG: convergence reached with residual ON GPU: {}",
                         max_residual);
            m_final_residual = max_residual;
        }

        return abs_ok || rel_ok;
    }

    virtual void solve(const DenseMatrix<T>& B,
                       DenseMatrix<T>&       X,
                       cudaStream_t          stream = NULL) override
    {

        float    time = 0;
        CPUTimer timer;
        GPUTimer gtimer;
        T        current_res;

        if (m_num_levels == 1) {
            RXMESH_TRACE("GMG:Direct solve used");
            // m_coarse_solver.template solve<numCols>(m_A, B, X, 1000);
            return;
        } else {
            m_iter_taken = 0;
            while (m_iter_taken < m_max_iter) {
                // m_v_cycle.cycle(0, m_gmg, *m_A, B, X,*m_rx);
                m_v_cycle.cycle(0, m_gmg, *m_A, m_v_cycle.B, X, *m_rx);
                current_res = m_v_cycle.m_r[0].norm2();
                /*RXMESH_TRACE(
                    "GMG: current residual: "
                    "{}",
                    current_res);*/

                timer.start();
                gtimer.start();

                if (
                    //is_converged(m_start_residual, current_res) ||
                    //is_converged_special(*m_A, X, m_v_cycle.B) ||
                    is_converged_special_gpu(*m_A, X, m_v_cycle.B)) {
                   

                    m_final_residual = current_res;
                    /*std::cout << "\nconverged! at " << m_final_residual
                              << " from residual of " << m_start_residual;*/
                    RXMESH_TRACE("GMG: #number of iterations to solve: {}",
                                 m_iter_taken);
                    // RXMESH_TRACE("GMG: final residual: {}",
                    // m_final_residual);
                    timer.stop();
                    gtimer.stop();
                    time += std::max(timer.elapsed_millis(),
                                     gtimer.elapsed_millis());
                    RXMESH_TRACE("GMG: #time taken to test for convergence: {}",
                                 time);
                    return;
                }
                timer.stop();
                gtimer.stop();
                time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());

                m_iter_taken++;
            }
            RXMESH_TRACE(
                "GMG: Solver did not reach convergence criteria. Residual: {}",
                current_res);
            RXMESH_TRACE("GMG: #number of iterations to solve: {}",
                         m_iter_taken);
            RXMESH_TRACE("GMG: #time taken to test for convergence: {}", time);
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

    int get_num_levels()
    {
        return m_num_levels;
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

    DenseMatrix<T> AX;
    DenseMatrix<T> R;
};

}  // namespace rxmesh