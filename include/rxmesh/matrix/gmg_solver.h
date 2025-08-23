#pragma once
#include "rxmesh/matrix/iterative_solver.h"

#include "rxmesh/attribute.h"
#include "rxmesh/matrix/gmg/gmg.h"
#include "rxmesh/matrix/gmg/v_cycle.h"
#include "rxmesh/matrix/gmg/v_cycle_pruned.h"
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

    GMGSolver(RXMeshStatic&    rx,
              SparseMatrix<T>& A,
              int              max_iter,
              int              num_levels     = 0,
              int              num_pre_relax  = 2,
              int              num_post_relax = 2,
              CoarseSolver     coarse_solver  = CoarseSolver::Jacobi,
              Sampling         sampling       = Sampling::Rand,
              T                abs_tol        = 1e-6,
              T                rel_tol        = 1e-6,
              int              threshold      = 1000,
              bool             pruned_ptap    = false,
              bool             verify_ptap    = false)
        : IterativeSolver<T, DenseMatrix<T>>(max_iter, abs_tol, rel_tol),
          m_rx(&rx),
          m_A(&A),
          m_coarse_solver(coarse_solver),
          m_sampling(sampling),
          m_num_pre_relax(num_pre_relax),
          m_num_post_relax(num_post_relax),
          m_num_levels(num_levels),
          m_threshold(threshold),
          m_pruned_ptap(pruned_ptap),
          AX(DenseMatrix<T>(A.rows(), 1, DEVICE)),
          R(DenseMatrix<T>(A.rows(), 1, DEVICE)),
          m_verify_ptap(verify_ptap)
    {
        if (m_coarse_solver == CoarseSolver::None) {
            RXMESH_ERROR("GMGSolver::GMGSolver() invalid coarse solver {}",
                         CoarseSolver::None);
        }
    }

    virtual void pre_solve(const DenseMatrix<T>& B,
                           DenseMatrix<T>&       X,
                           cudaStream_t          stream = NULL) override
    {
        CPUTimer timer;
        GPUTimer gtimer;

        // Construct GMG operator
        timer.start();
        gtimer.start();
        m_gmg =
            GMG<T>(*m_rx, m_num_levels, m_threshold, m_sampling, m_pruned_ptap);
        timer.stop();
        gtimer.stop();

        RXMESH_INFO(
            "GMGSolver::pre_solve() GMG construction took {} (ms), {} (ms)",
            timer.elapsed_millis(),
            gtimer.elapsed_millis());

        m_num_levels = m_gmg.m_num_levels;


        timer.start();
        gtimer.start();

        // Construct V-cycle
        if (!m_pruned_ptap) {
            m_v_cycle = std::make_unique<VCycle<T>>(m_gmg,
                                                    *m_rx,
                                                    *m_A,
                                                    B.cols(),
                                                    m_coarse_solver,
                                                    m_num_pre_relax,
                                                    m_num_post_relax);


        } else {
            m_v_cycle = std::make_unique<VCyclePruned<T>>(m_gmg,
                                                          *m_rx,
                                                          *m_A,
                                                          B.cols(),
                                                          m_coarse_solver,
                                                          m_num_pre_relax,
                                                          m_num_post_relax);
        }
        m_v_cycle->coarser_systems(m_gmg, *m_rx, *m_A);


        timer.stop();
        gtimer.stop();

        RXMESH_INFO(
            "GMGSolver::pre_solve(): V-cycle construction took {} (ms), {} "
            "(ms)",
            timer.elapsed_millis(),
            gtimer.elapsed_millis());

        if (m_verify_ptap && m_pruned_ptap) {
            m_v_cycle->verify_coarse_system(m_gmg, *m_A);
        }

        constexpr int numCols = 3;
        assert(numCols == B.cols());

        m_v_cycle->template calc_residual<numCols>(
            m_v_cycle->m_a[0].a, X, B, m_v_cycle->m_r[0]);

        this->m_start_residual = m_v_cycle->m_r[0].norm2();
    }

    void render_laplacian()
    {

        auto make_connections_vector = [&](std::vector<int>& connector,
                                           int               l,
                                           int               vertex = 0) {
            constexpr uint32_t blockThreads = 256;
            uint32_t           blocks_new =
                DIVIDE_UP(m_v_cycle->m_a[l - 1].a.rows(), blockThreads);

            int* d_c;
            CUDA_ERROR(
                cudaMalloc(&d_c, m_v_cycle->m_a[l - 1].a.rows() * sizeof(int)));
            auto a = m_v_cycle->m_a[l - 1].a;
            for_each_item<<<blocks_new, blockThreads>>>(
                m_v_cycle->m_a[l - 1].a.rows(),
                [a, vertex, d_c] __device__(int i) mutable {
                    if (i != vertex) {
                        d_c[i] = 0;
                        return;
                    }
                    for (int q = a.row_ptr()[i]; q < a.row_ptr()[i + 1]; ++q) {
                        int a_col  = a.col_idx()[q];
                        d_c[a_col] = 1;
                    }
                });
            int* h_c = new int[m_v_cycle->m_a[l - 1].a.rows() + 1];

            CUDA_ERROR(cudaMemcpy(h_c,
                                  d_c,
                                  sizeof(int) * m_v_cycle->m_a[l - 1].a.rows(),
                                  cudaMemcpyDeviceToHost));

            for (int i = 0; i < m_v_cycle->m_a[l - 1].a.rows(); i++) {
                connector[i] = h_c[i];
            }
            connector[vertex] = 1;
        };

        std::vector<std::vector<int>> connections(m_num_levels);
        for (int l = 1; l < m_num_levels; l++) {

            connections[l - 1] =
                std ::vector<int>(m_v_cycle->m_a[l - 1].a.rows());

            make_connections_vector(connections[l - 1], l);


            polyscope::getSurfaceMesh("Level" + std::to_string(l))
                ->addVertexScalarQuantity("connections", connections[l - 1]);
        }
    }


    bool is_converged(SparseMatrix<T>& A, DenseMatrix<T>& X, DenseMatrix<T>& B)
    {
        using IndexT = typename DenseMatrix<T>::IndexT;

        const IndexT num_rhs = B.cols();
        const IndexT n       = A.rows();

        T max_residual = 0.0;
        for (IndexT i = 0; i < num_rhs; ++i) {
            A.multiply(X.col_data(i, DEVICE), AX.col_data(0, DEVICE));
            R.copy_from(AX, DEVICE, DEVICE);

            auto col_i = B.col(i);
            R.axpy(col_i, T(-1));  // R = AX - B_i

            // Compute norm of R and B
            T r_norm = R.norm2();
            T b_norm = B.col(i).norm2();

            T residual   = r_norm / (b_norm + 1e-20);
            max_residual = std::max(max_residual, residual);
        }

        bool abs_ok = max_residual < this->m_abs_tol;
        bool rel_ok = max_residual < this->m_rel_tol;


        if (abs_ok || rel_ok) {
            this->m_final_residual = max_residual;
        }

        return abs_ok || rel_ok;
    }

    virtual void solve(DenseMatrix<T>& B,
                       DenseMatrix<T>& X,
                       cudaStream_t    stream = NULL) override
    {

        float time = 0;

        T current_res;

        this->m_iter_taken = 0;
        while (this->m_iter_taken < this->m_max_iter) {
            m_v_cycle->cycle(0, m_gmg, *m_A, B, X, *m_rx);
            // current_res = m_v_cycle.m_r[0].norm2();

            if (is_converged(*m_A, X, B)) {
                return;
            }
            this->m_iter_taken++;
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


    float gmg_memory_alloc_time() const
    {
        return m_gmg.memory_alloc_time;
    }

    float v_cycle_memory_alloc_time() const
    {
        return m_v_cycle->memory_alloc_time;
    }

    virtual ~GMGSolver()
    {
    }

   protected:
    RXMeshStatic*              m_rx;
    SparseMatrix<T>*           m_A;
    GMG<T>                     m_gmg;
    std::unique_ptr<VCycle<T>> m_v_cycle;
    CoarseSolver               m_coarse_solver;
    Sampling                   m_sampling;
    int                        m_num_pre_relax;
    int                        m_num_post_relax;
    int                        m_num_levels;
    int                        m_threshold;
    bool                       m_pruned_ptap;
    DenseMatrix<T>             AX;
    DenseMatrix<T>             R;
    bool                       m_verify_ptap;
};

}  // namespace rxmesh