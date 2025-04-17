#pragma once

#include "rxmesh/matrix/gmg/gmg.h"

#include "rxmesh/matrix/gmg/jacobi_solver.h"

namespace rxmesh {

enum class CoarseSolver
{
    Jacobi      = 0,
    GaussSeidel = 1,
    CG          = 2,
};

/**
 * @brief the coarse A
 */
template <typename T>
struct CoarseA
{

    CoarseA() {};
    SparseMatrix<T> a;

    T*   d_val;
    T*   h_val;
    int* d_row_ptr;
    int* h_row_ptr;
    int* d_col_idx;
    int* h_col_idx;
};

template <typename T>
struct VCycle
{
    VCycle(const VCycle&)            = delete;
    VCycle()                         = default;
    VCycle(VCycle&&)                 = default;
    VCycle& operator=(const VCycle&) = default;
    VCycle& operator=(VCycle&&)      = default;
    ~VCycle()                        = default;

    int m_num_pre_relax;
    int m_num_post_relax;

    // For index, the fine mesh is always indexed with 0
    // The first coarse level index is 1
    //
    // Here we store the rhs and x for the levels.
    // rhs and x of the fine are user inputs and we should not copy them

    std::vector<DenseMatrix<T>> m_rhs;  // levels
    std::vector<DenseMatrix<T>> m_x;    // levels
    std::vector<DenseMatrix<T>> m_r;    // fine + levels


    std::vector<CoarseA<T>> m_a;  // levels

    // TODO abstract away the solver type
    std::vector<JacobiSolver<T>> m_smoother;  // fine + levels

    JacobiSolver<T> m_coarse_solver;


    VCycle(GMG<T>&               gmg,
           RXMeshStatic&         rx,
           SparseMatrix<T>&      A,
           const DenseMatrix<T>& rhs,
           CoarseSolver          coarse_solver  = CoarseSolver::Jacobi,
           int                   num_pre_relax  = 2,
           int                   num_post_relax = 2)
        : m_num_pre_relax(num_pre_relax), m_num_post_relax(num_post_relax)
    {
        // allocate memory for coarsened LHS and RHS
        m_smoother.emplace_back(gmg.m_num_samples[0], gmg.m_num_samples[0]);

        m_r.emplace_back(rx, gmg.m_num_samples[0], rhs.cols());
        m_rhs.emplace_back(rx, gmg.m_num_samples[0], rhs.cols());
        for (int l = 1; l < gmg.m_num_levels; ++l) {
            m_rhs.emplace_back(rx, gmg.m_num_samples[l], rhs.cols());
            m_x.emplace_back(rx, gmg.m_num_samples[l], rhs.cols());

            m_r.emplace_back(rx, gmg.m_num_samples[l], rhs.cols());
            gmg.m_prolong_op[l - 1].alloc_multiply_buffer(m_r.back(),
                                                          m_rhs[l - 1]);

            if (l < gmg.m_num_levels - 1) {
                m_smoother.emplace_back(gmg.m_num_samples[l],
                                        gmg.m_num_samples[l]);
            } else {
                // coarsest level
                m_coarse_solver =
                    JacobiSolver<T>(gmg.m_num_samples[l], gmg.m_num_samples[l]);
            }
        }
        m_a.resize(gmg.m_num_levels - 1);
        // construct m_a for all levels
        pt_A_p(gmg.m_prolong_op[0], A, m_a[0]);
        for (int l = 1; l < gmg.m_num_levels - 1; ++l) {
            pt_A_p(gmg.m_prolong_op[l], m_a[l - 1].a, m_a[l]);
        }
    }


    void pt_A_p(SparseMatrixConstantNNZRow<T, 3>& P,
                SparseMatrix<T>&                  A,
                CoarseA<T>&                       C)
    {
        // S = transpose(P) * A
        // C = S * A


        cusparseSpGEMMDescr_t spgemmDesc;
        CUSPARSE_ERROR(cusparseSpGEMM_createDescr(&spgemmDesc));

        SparseMatrix<T> Pt = P.transpose();

        cusparseSpMatDescr_t S_spmat;
        cusparseSpMatDescr_t C_spmat;

        int     s_rows = P.cols();
        int     s_cols = A.cols();
        int64_t s_nnz  = 0;

        int     c_rows = P.cols();
        int     c_cols = P.cols();
        int64_t c_nnz  = 0;

        int*   s_rowPtr;
        int*   s_colIdx;
        float* s_values;

        // Create an empty descriptor for C and S
        CUSPARSE_ERROR(cusparseCreateCsr(&S_spmat,
                                         s_rows,
                                         s_cols,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_32F));

        CUSPARSE_ERROR(cusparseCreateCsr(&C_spmat,
                                         c_rows,
                                         c_cols,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_32F));


        // S = transpose(P) *A
        sparse_gemm(A.m_cusparse_handle,
                    Pt.m_spdescr,
                    A.m_spdescr,
                    s_rows,
                    s_cols,
                    s_nnz,
                    s_rowPtr,
                    s_colIdx,
                    s_values,
                    S_spmat,
                    spgemmDesc);


        // C = S * P
        sparse_gemm(A.m_cusparse_handle,
                    S_spmat,
                    P.m_spdescr,
                    c_rows,
                    c_cols,
                    c_nnz,
                    C.d_row_ptr,
                    C.d_col_idx,
                    C.d_val,
                    C_spmat,
                    spgemmDesc);

        // Alloc C on the host
        C.h_row_ptr = static_cast<int*>(malloc((c_rows + 1) * sizeof(int)));
        C.h_col_idx = static_cast<int*>(malloc(c_nnz * sizeof(int)));
        C.h_val     = static_cast<T*>(malloc(c_nnz * sizeof(T)));

        // move C to the host
        CUDA_ERROR(cudaMemcpy(C.h_row_ptr,
                              C.d_row_ptr,
                              (c_rows + 1) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        CUDA_ERROR(cudaMemcpy(C.h_col_idx,
                              C.d_col_idx,
                              c_nnz * sizeof(int),
                              cudaMemcpyDeviceToHost));

        CUDA_ERROR(cudaMemcpy(
            C.h_val, C.d_val, c_nnz * sizeof(T), cudaMemcpyDeviceToHost));

        // Create C SparseMatrix
        C.a = SparseMatrix<T>(c_rows,
                              c_cols,
                              c_nnz,
                              C.d_row_ptr,
                              C.d_col_idx,
                              C.d_val,
                              C.h_row_ptr,
                              C.h_col_idx,
                              C.h_val);

        // clean up
        Pt.release();
        GPU_FREE(s_rowPtr);
        GPU_FREE(s_colIdx);
        GPU_FREE(s_values);

        CUSPARSE_ERROR(cusparseDestroySpMat(S_spmat));
        CUSPARSE_ERROR(cusparseDestroySpMat(C_spmat));
        CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(spgemmDesc));
    }

    /**
     * @brief run the solver.
     */
    void solve(GMG<T>&          gmg,
               SparseMatrix<T>& A,
               DenseMatrix<T>&  rhs,
               DenseMatrix<T>&  result,
               int              num_iter)
    {

        for (int i = 0; i < num_iter; ++i) {
            cycle(0, gmg, A, rhs, result);
        }
    }

    /**
     * @brief implement one step/cycle of the V cycle (bootstrap the recursive
     * call)
     */
    void cycle(int                   level,
               GMG<T>&               gmg,
               SparseMatrix<T>&      A,
               const DenseMatrix<T>& f,  // rhs
               DenseMatrix<T>&       v)        // x
    {
        constexpr int numCols = 3;
        assert(numCols == f.cols());

        if (level < gmg.m_num_levels - 1) {
            // pre-smoothing
            m_smoother[level].template solve<numCols>(A, f, v, m_num_pre_relax);

            // calc residual
            calc_residual<numCols>(A, v, f, m_r[level]);

            //// restrict residual
            gmg.m_prolong_op[level].multiply(
                m_r[level], m_rhs[level + 1], true);
            //// recurse
            cycle(level + 1, gmg, m_a[level].a, m_rhs[level + 1], m_x[level]);

            // prolong
            // x = x + P*u
            gmg.m_prolong_op[level].multiply(
                m_x[level], v, false, false, T(1.0), T(1.0));

            //// post-smoothing
            m_smoother[level].template solve<numCols>(
                A, f, v, m_num_post_relax);

        } else {
            // the coarsest level
            // m_coarse_solver.template solve<numCols>(A, f, v,
            // m_num_post_relax);
            m_coarse_solver.template solve<numCols>(A, f, v, 5);
        }
    }


    /**
     * @brief compute r = f - A.v
     */
    template <int numCols>
    void calc_residual(const SparseMatrix<T>& A,
                       const DenseMatrix<T>&  v,
                       const DenseMatrix<T>&  f,
                       DenseMatrix<T>&        r)
    {
        assert(numCols == f.cols());

        constexpr uint32_t blockThreads = 256;

        uint32_t blocks = DIVIDE_UP(A.rows(), blockThreads);


        for_each_item<<<blocks, blockThreads>>>(
            A.rows(), [=] __device__(int row) mutable {
                T av[numCols];

                for (int c = 0; c < numCols; ++c) {
                    av[c] = 0;
                }

                int start = A.row_ptr()[row];
                int stop  = A.row_ptr()[row + 1];

                for (int j = start; j < stop; ++j) {

                    const int col = A.col_idx()[j];

                    const T val = A.get_val_at(j);

                    for (int c = 0; c < numCols; ++c) {
                        av[c] += val * v(col, c);
                    }
                }

                for (int c = 0; c < numCols; ++c) {
                    r(row, c) = f(row, c) - av[c];
                }
            });
    }

   private:
    void sparse_gemm(const cusparseHandle_t&     handle,
                     const cusparseSpMatDescr_t& A,
                     const cusparseSpMatDescr_t& B,
                     const int                   c_rows,
                     const int                   c_cols,
                     int64_t&                    c_nnz,
                     int*&                       c_rowPtr,
                     int*&                       c_colIdx,
                     float*&                     c_values,
                     const cusparseSpMatDescr_t& C_spmat,
                     cusparseSpGEMMDescr_t&      spgemmDesc)
    {
        // C = op(A)*B


        // Ask bufferSize1 bytes for external memory
        size_t bufferSize1 = 0;
        void*  dBuffer1    = nullptr;

        // Allocate workspace buffer for SpGEMM
        float alpha = 1.0f, beta = 0.0f;


        CUSPARSE_ERROR(
            cusparseSpGEMM_workEstimation(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          A,
                                          B,
                                          &beta,
                                          C_spmat,
                                          CUDA_R_32F,
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc,
                                          &bufferSize1,
                                          nullptr));

        CUDA_ERROR(cudaMalloc(&dBuffer1, bufferSize1));

        // Execute work estimation
        // inspect the matrices op(A) and B to understand the memory
        // requirement for the next step
        CUSPARSE_ERROR(
            cusparseSpGEMM_workEstimation(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          A,
                                          B,
                                          &beta,
                                          C_spmat,
                                          CUDA_R_32F,
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc,
                                          &bufferSize1,
                                          dBuffer1));

        // ask bufferSize2 bytes for external memory
        size_t bufferSize2 = 0;
        void*  dBuffer2    = nullptr;
        CUSPARSE_ERROR(cusparseSpGEMM_compute(handle,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha,
                                              A,
                                              B,
                                              &beta,
                                              C_spmat,
                                              CUDA_R_32F,
                                              CUSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc,
                                              &bufferSize2,
                                              nullptr));
        CUDA_ERROR(cudaMalloc(&dBuffer2, bufferSize2));

        // compute the intermediate product of A * B
        CUSPARSE_ERROR(cusparseSpGEMM_compute(handle,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha,
                                              A,
                                              B,
                                              &beta,
                                              C_spmat,
                                              CUDA_R_32F,
                                              CUSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc,
                                              &bufferSize2,
                                              dBuffer2));

        // get matrix C non-zero entries C_nnz1
        int64_t cr, cc;
        CUSPARSE_ERROR(cusparseSpMatGetSize(C_spmat, &cr, &cc, &c_nnz));
        assert(c_rows == cr);
        assert(c_cols == cc);


        // allocate matrix C
        CUDA_ERROR(cudaMalloc(&c_rowPtr, (c_rows + 1) * sizeof(int)));
        CUDA_ERROR(cudaMalloc(&c_colIdx, c_nnz * sizeof(int)));
        CUDA_ERROR(cudaMalloc(&c_values, c_nnz * sizeof(T)));

        // update S_spmat with the new pointers
        CUSPARSE_ERROR(
            cusparseCsrSetPointers(C_spmat, c_rowPtr, c_colIdx, c_values));

        // copy the final products to the matrix C
        CUSPARSE_ERROR(cusparseSpGEMM_copy(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           A,
                                           B,
                                           &beta,
                                           C_spmat,
                                           CUDA_R_32F,
                                           CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc));

        GPU_FREE(dBuffer1);
        GPU_FREE(dBuffer2);
    }
};
}  // namespace rxmesh