#pragma once

#include <optional>

#include "rxmesh/diff/diff_vector_problem.h"

#include "rxmesh/matrix/cg_solver.h"
#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/matrix/cudss_cholesky_solver.h"
#include "rxmesh/matrix/pcg_solver.h"


namespace rxmesh {

template <typename T, int VariableDim, typename ObjHandleT, typename SolverT>
struct GaussNetwtonSolver
{

    using DiffProblemT = DiffVectorProblem<T, VariableDim, ObjHandleT>;
    using SpMatT       = SparseMatrix<T>;
    using DenseMatT    = typename DiffProblemT::DenseMatT;
    using IndexT       = typename SpMatT::IndexT;

    DiffProblemT&          problem;
    DenseMatT              dir;
    std::optional<SolverT> solver;
    SpMatT                 JtJ;
    SpMatT                 jac_trans;

    bool is_prep_solver_called;

    float solve_time;

    /**
     * @brief Newton solver
     */
    GaussNetwtonSolver(DiffProblemT& p)
        : problem(p),
          solve_time(0),
          is_prep_solver_called(false),
          m_JtJ_nnz_expected(-1)
    {
        dir.reset(0, LOCATION_ALL);
    }

    /**
     * @brief should be called only after prep_eval in the given problem
     * (DiffVectorProblem)
     */
    void prep_solver(int cg_max_iter = 1000,
                     T   cg_abs_tol  = 1e-3,
                     T   cg_rel_tol  = 0.0)
    {
        is_prep_solver_called = true;

        // allocate the direction
        dir = DenseMatT(problem.rx, problem.jac->cols(), 1, LOCATION_ALL);

        // create JtJ
        create_JtJ();

        // create an instance of SolverT (solver)

#ifdef USE_CUDSS
        // cuDSS Cholesky
        if constexpr (std::is_base_of_v<
                          cuDSSCholeskySolver<SpMatT, DenseMatT::OrderT>,
                          SolverT>) {
            solver.emplace(&JtJ);
            solver->pre_solve(problem.rx, *problem.grad, dir);
        }
#endif

        // cuSolver Cholesky
        if constexpr (std::is_base_of_v<
                          CholeskySolver<SpMatT, DenseMatT::OrderT>,
                          SolverT>) {
            solver.emplace(&JtJ);
            solver->pre_solve(problem.rx);
        }

        // CG
        if constexpr (std::is_base_of_v<CGSolver<T, DenseMatT::OrderT>,
                                        SolverT>) {
            solver.emplace(JtJ, 1, cg_max_iter, cg_abs_tol, cg_rel_tol);
            solver->pre_solve(*problem.grad, dir);
        }
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

        // compute JtJ
        compute_JtJ();

        // solve for new direction

        GPUTimer timer;
        timer.start();

#ifdef USE_CUDSS
        // cuDSS Cholesky
        if constexpr (std::is_base_of_v<
                          cuDSSCholeskySolver<SpMatT, DenseMatT::OrderT>,
                          SolverT>) {
            solver->pre_solve(problem.rx, dir, *problem.grad);
            solver->solve(*problem.grad, dir, stream);
        }
#endif

        // cuSolver Cholesky
        if constexpr (std::is_base_of_v<
                          CholeskySolver<SpMatT, DenseMatT::OrderT>,
                          SolverT>) {
            solver->pre_solve(problem.rx);
            solver->solve(*problem.grad, dir, stream);
        }

        // CG
        if constexpr (std::is_base_of_v<CGSolver<T, DenseMatT::OrderT>,
                                        SolverT>) {
            solver->pre_solve(*problem.grad, dir, stream);
            solver->solve(*problem.grad, dir, stream);
        }

        timer.stop();
        solve_time += timer.elapsed_millis();
    }

    /**
     * @brief release all memory and resources
     */
    __host__ void release()
    {
        // SpGEMM reuse resources
        if (m_spgemmDesc) {
            CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(m_spgemmDesc));
            m_spgemmDesc = nullptr;
        }

        GPU_FREE(m_dBuffer1);
        GPU_FREE(m_dBuffer2);
        GPU_FREE(m_dBuffer3);
        GPU_FREE(m_dBuffer4);
        GPU_FREE(m_dBuffer5);

        m_bufferSize1      = 0;
        m_bufferSize2      = 0;
        m_bufferSize3      = 0;
        m_bufferSize4      = 0;
        m_bufferSize5      = 0;
        m_spgemm_ready     = false;
        m_JtJ_nnz_expected = -1;

        JtJ.release();
        jac_trans.release();

        dir.release();

        is_prep_solver_called = false;
        solve_time            = 0.0f;
    }

   private:
    void create_JtJ()
    {
        // Build J^T once (allocates CSR for jac_trans and internal transpose
        // buffer in J)
        jac_trans = problem.jac->transpose();

        // Dimension: J is (m x n), JtJ is (n x n)
        const IndexT n = problem.jac->cols();

        // If we already built SpGEMM reuse once, do not rebuild.
        if (m_spgemm_ready) {
            return;
        }


        // Create a JtJ placeholder with:
        // - correct (n x n)
        // - nnz = 0
        // - device row_ptr allocated
        // - col/val nullptr for now
        JtJ                   = SpMatT();
        JtJ.m_context         = problem.rx.get_context();
        JtJ.m_block_shape     = BlockShape(1, 1);
        JtJ.m_num_rows        = n;
        JtJ.m_num_cols        = n;
        JtJ.m_nnz             = 0;
        JtJ.m_is_user_managed = false;
        JtJ.m_op              = Op::INVALID;

        // device
        CUDA_ERROR(
            cudaMalloc((void**)&JtJ.m_d_row_ptr, (n + 1) * sizeof(IndexT)));
        CUDA_ERROR(cudaMemset(JtJ.m_d_row_ptr, 0, (n + 1) * sizeof(IndexT)));

        // host
        JtJ.m_h_row_ptr = (IndexT*)malloc((n + 1) * sizeof(IndexT));

        // We do not allocate d_col_idx/d_val yet (unknown nnz)
        JtJ.m_d_col_idx = nullptr;
        JtJ.m_d_val     = nullptr;

        // Mark device allocation so release() logic behaves
        JtJ.m_allocated = (JtJ.m_allocated | DEVICE);
        JtJ.m_allocated = (JtJ.m_allocated | HOST);

        // Create cusparse handle + placeholder CSR descriptor for JtJ
        JtJ.init_cusparse(JtJ);

        // Create SpGEMM reuse descriptor
        CUSPARSE_ERROR(cusparseSpGEMM_createDescr(&m_spgemmDesc));


        // We use jac_trans's handle. any handle is fine, but keep it
        // consistent.
        cusparseHandle_t handle = jac_trans.m_cusparse_handle;

        const T alpha = T(1);
        const T beta  = T(0);

        const cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        const cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

        // 1) workEstimation
        CUSPARSE_ERROR(
            cusparseSpGEMMreuse_workEstimation(handle,
                                               opA,
                                               opB,
                                               jac_trans.m_spdescr,
                                               problem.jac->m_spdescr,
                                               JtJ.m_spdescr,
                                               CUSPARSE_SPGEMM_DEFAULT,
                                               m_spgemmDesc,
                                               &m_bufferSize1,
                                               nullptr));
        CUDA_ERROR(cudaMalloc(&m_dBuffer1, m_bufferSize1));

        CUSPARSE_ERROR(
            cusparseSpGEMMreuse_workEstimation(handle,
                                               opA,
                                               opB,
                                               jac_trans.m_spdescr,
                                               problem.jac->m_spdescr,
                                               JtJ.m_spdescr,
                                               CUSPARSE_SPGEMM_DEFAULT,
                                               m_spgemmDesc,
                                               &m_bufferSize1,
                                               m_dBuffer1));

        // 2) nnz (symbolic) stage: computes row_ptr and nnz(C)
        CUSPARSE_ERROR(cusparseSpGEMMreuse_nnz(handle,
                                               opA,
                                               opB,
                                               jac_trans.m_spdescr,
                                               problem.jac->m_spdescr,
                                               JtJ.m_spdescr,
                                               CUSPARSE_SPGEMM_DEFAULT,
                                               m_spgemmDesc,
                                               &m_bufferSize2,
                                               nullptr,
                                               &m_bufferSize3,
                                               nullptr,
                                               &m_bufferSize4,
                                               nullptr));

        CUDA_ERROR(cudaMalloc(&m_dBuffer2, m_bufferSize2));
        CUDA_ERROR(cudaMalloc(&m_dBuffer3, m_bufferSize3));
        CUDA_ERROR(cudaMalloc(&m_dBuffer4, m_bufferSize4));

        CUSPARSE_ERROR(cusparseSpGEMMreuse_nnz(handle,
                                               opA,
                                               opB,
                                               jac_trans.m_spdescr,
                                               problem.jac->m_spdescr,
                                               JtJ.m_spdescr,
                                               CUSPARSE_SPGEMM_DEFAULT,
                                               m_spgemmDesc,
                                               &m_bufferSize2,
                                               m_dBuffer2,
                                               &m_bufferSize3,
                                               m_dBuffer3,
                                               &m_bufferSize4,
                                               m_dBuffer4));

        // free some early buffers
        GPU_FREE(m_dBuffer1);
        GPU_FREE(m_dBuffer2);


        // Query nnz(C)
        int64_t C_num_rows1 = 0, C_num_cols1 = 0, C_nnz1 = 0;
        CUSPARSE_ERROR(cusparseSpMatGetSize(
            JtJ.m_spdescr, &C_num_rows1, &C_num_cols1, &C_nnz1));


        m_JtJ_nnz_expected = C_nnz1;

        // Allocate C col/val
        JtJ.m_nnz = static_cast<IndexT>(C_nnz1);

        // device
        CUDA_ERROR(
            cudaMalloc((void**)&JtJ.m_d_col_idx, JtJ.m_nnz * sizeof(IndexT)));
        CUDA_ERROR(cudaMalloc((void**)&JtJ.m_d_val, JtJ.m_nnz * sizeof(T)));
        CUDA_ERROR(cudaMemset(JtJ.m_d_val, 0, JtJ.m_nnz * sizeof(T)));

        // host
        JtJ.m_h_col_idx = (IndexT*)malloc(JtJ.m_nnz * sizeof(IndexT));
        JtJ.m_h_val     = (T*)malloc(JtJ.m_nnz * sizeof(T));

        // Update matC descriptor pointers to the real storage
        CUSPARSE_ERROR(cusparseCsrSetPointers(
            JtJ.m_spdescr, JtJ.m_d_row_ptr, JtJ.m_d_col_idx, JtJ.m_d_val));


        // Initialize cuDSS matrix now that CSR storage is complete
        JtJ.init_cudss(JtJ);


        // 3) copy stage (finalize reuse setup)
        cusparseHandle_t handle2 = jac_trans.m_cusparse_handle;

        CUSPARSE_ERROR(cusparseSpGEMMreuse_copy(handle2,
                                                opA,
                                                opB,
                                                jac_trans.m_spdescr,
                                                problem.jac->m_spdescr,
                                                JtJ.m_spdescr,
                                                CUSPARSE_SPGEMM_DEFAULT,
                                                m_spgemmDesc,
                                                &m_bufferSize5,
                                                nullptr));
        CUDA_ERROR(cudaMalloc(&m_dBuffer5, m_bufferSize5));

        CUSPARSE_ERROR(cusparseSpGEMMreuse_copy(handle2,
                                                opA,
                                                opB,
                                                jac_trans.m_spdescr,
                                                problem.jac->m_spdescr,
                                                JtJ.m_spdescr,
                                                CUSPARSE_SPGEMM_DEFAULT,
                                                m_spgemmDesc,
                                                &m_bufferSize5,
                                                m_dBuffer5));
        GPU_FREE(m_dBuffer3);


        m_spgemm_ready = true;

        // populate host
        CUDA_ERROR(cudaMemcpy(JtJ.m_h_row_ptr,
                              JtJ.m_d_row_ptr,
                              (n + 1) * sizeof(IndexT),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(JtJ.m_h_val,
                              JtJ.m_d_val,
                              JtJ.m_nnz * sizeof(T),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(JtJ.m_h_col_idx,
                              JtJ.m_d_col_idx,
                              JtJ.m_nnz * sizeof(IndexT),
                              cudaMemcpyDeviceToHost));
    }

    void compute_JtJ()
    {
        // Update jac_trans numerically from current J
        // (uses cached transpose buffer in SparseMatrix::transpose(ret))
        problem.jac->transpose(jac_trans);

        if (!m_spgemm_ready) {
            // Fallback (should not happen if prep_solver calls create_JtJ)
            create_JtJ();
        }

        // Sanity check nnz(JtJ) didn't change
        if (false) {
            int64_t r = 0, c = 0, nnz = 0;
            CUSPARSE_ERROR(cusparseSpMatGetSize(JtJ.m_spdescr, &r, &c, &nnz));
            if (m_JtJ_nnz_expected >= 0 && nnz != m_JtJ_nnz_expected) {
                RXMESH_ERROR(
                    "GaussNetwtonSolver::compute_JtJ() JtJ nnz changed ({} -> "
                    "{}). Rebuild required.",
                    m_JtJ_nnz_expected,
                    nnz);
                return;
            }
        }

        const T alpha = T(1);
        const T beta  = T(0);

        const cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        const cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

        // Compute: JtJ = alpha * (J^T * J) + beta * JtJ
        CUSPARSE_ERROR(cusparseSpGEMMreuse_compute(jac_trans.m_cusparse_handle,
                                                   opA,
                                                   opB,
                                                   &alpha,
                                                   jac_trans.m_spdescr,
                                                   problem.jac->m_spdescr,
                                                   &beta,
                                                   JtJ.m_spdescr,
                                                   cuda_type<T>(),
                                                   CUSPARSE_SPGEMM_DEFAULT,
                                                   m_spgemmDesc));
    }

    // SpGEMM reuse state (persist across iterations)
    bool                  m_spgemm_ready = false;
    cusparseSpGEMMDescr_t m_spgemmDesc   = nullptr;
    int                   m_JtJ_nnz_expected;

    void*  m_dBuffer1    = nullptr;  // workEstimation
    void*  m_dBuffer2    = nullptr;  // nnz
    void*  m_dBuffer3    = nullptr;  // nnz
    void*  m_dBuffer4    = nullptr;  // nnz
    void*  m_dBuffer5    = nullptr;  // copy
    size_t m_bufferSize1 = 0;
    size_t m_bufferSize2 = 0;
    size_t m_bufferSize3 = 0;
    size_t m_bufferSize4 = 0;
    size_t m_bufferSize5 = 0;
};

}  // namespace rxmesh