#pragma once

#include "gmg.h"

#include "jacobi_solver.h"

namespace rxmesh {

enum class CoarseSolver
{
    Jacobi      = 0,
    GaussSeidel = 1,
    CG          = 2,
};

template <typename T>
struct VCycle
{
    VCycle(const VCycle&) = delete;

    int m_num_pre_relax;
    int m_num_post_relax;
    T   m_omega;

    // For index, the fine mesh is always indexed with 0
    // The first coarse level index is 1
    //
    // Here we store the rhs and x for the levels.
    // rhs and x of the fine are user inputs and we should not copy them

    std::vector<DenseMatrix<T>>  m_rhs;  // levels
    std::vector<DenseMatrix<T>>  m_x;    // levels
    std::vector<SparseMatrix<T>> m_a;    // levels
    std::vector<DenseMatrix<T>>  m_r;    // levels

    // TODO abstract away the solver type
    std::vector<JacobiSolver<T>> m_coarse_solver;  // fine + levels


    VCycle(GMG<T>&          gmg,
           RXMeshStatic&    rx,
           SparseMatrix<T>& A,
           DenseMatrix<T>&  rhs,
           CoarseSolver     coarse_solver  = CoarseSolver::Jacobi,
           int              num_cycles     = 2,
           int              num_pre_relax  = 2,
           int              num_post_relax = 2,
           T                omega          = 0.5)
        : m_num_pre_relax(num_pre_relax),
          m_num_post_relax(num_post_relax),
          m_omega(omega)
    {
        // allocate memory for coarsened LHS and RHS
        m_coarse_solver.emplace_back(gmg.m_num_samples[0],
                                     gmg.m_num_samples[0]);

        m_r.emplace_back(rx, gmg.m_num_samples[0], rhs.cols());

        for (int l = 1; l < gmg.m_num_levels; ++l) {
            m_rhs.emplace_back(rx, gmg.m_num_samples[l], rhs.cols());
            m_x.emplace_back(rx, gmg.m_num_samples[l], rhs.cols());
            m_coarse_solver.emplace_back(A.rows(), A.cols());
            m_coarse_solver.emplace_back(gmg.m_num_samples[l],
                                         gmg.m_num_samples[l]);

            m_r.emplace_back(rx, gmg.m_num_samples[l], rhs.cols());
        }


        // TODO construct m_a for all levels
    }

#if 0
    void pt_A_p(SparseMatrixConstantNNZRow<T, 3>& p,
                SparseMatrix<T>&                  A,
                SparseMatrix<T>&                  out)
    {

        // Create an empty descriptor for matC
        CUSPARSE_ERROR(cusparseCreateCsr(&out.m_spdescr,
                                         p.rows(),
                                         B_cols,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_32F));

        // Allocate workspace buffer for SpGEMM
        float                 alpha = 1.0f, beta = 0.0f;
        cusparseSpGEMMDescr_t spgemmDesc;
        CUSPARSE_ERROR(cusparseSpGEMM_createDescr(&spgemmDesc));

        // phase 1: work estimation
        size_t bufferSize1 = 0;
        void*  dBuffer1    = nullptr;

        // MAKE THIS DO THE TRANSPOSE, DONT TRANSPOSE EXPLICITLY
        auto operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
        if (transpose == 1)
            operation = CUSPARSE_OPERATION_TRANSPOSE;

        CUSPARSE_ERROR(
            cusparseSpGEMM_workEstimation(handle,
                                          operation,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          A.m_spdescr,
                                          B.m_spdescr,
                                          &beta,
                                          matC,
                                          cuda_type<T>(),
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc,
                                          &bufferSize1,
                                          nullptr));

        CUDA_ERROR(cudaMalloc(&dBuffer1, bufferSize1));

        // Execute work estimation
        CUSPARSE_ERROR(
            cusparseSpGEMM_workEstimation(handle,
                                          operation,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          A.m_spdescr,
                                          B.m_spdescr,
                                          &beta,
                                          matC,
                                          cuda_type<T>(),
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc,
                                          &bufferSize1,
                                          dBuffer1));

        // Phase 2: Compute non-zero pattern of C
        size_t bufferSize2 = 0;
        void*  dBuffer2    = nullptr;
        CUSPARSE_ERROR(cusparseSpGEMM_compute(handle,
                                              operation,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha,
                                              A.m_spdescr,
                                              B.m_spdescr,
                                              &beta,
                                              matC,
                                              cuda_type<T>(),
                                              CUSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc,
                                              &bufferSize2,
                                              nullptr));
        CUDA_ERROR(cudaMalloc(&dBuffer2, bufferSize2));

        // Execute non-zero pattern computation
        CUSPARSE_ERROR(cusparseSpGEMM_compute(handle,
                                              operation,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha,
                                              A.m_spdescr,
                                              B.m_spdescr,
                                              &beta,
                                              matC,
                                              cuda_type<T>(),
                                              CUSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc,
                                              &bufferSize2,
                                              dBuffer2));

        // Get the size of matrix C
        int64_t C_rows, C_cols, nnzC;
        CUSPARSE_ERROR(cusparseSpMatGetSize(matC, &C_rows, &C_cols, &nnzC));


        // Allocate memory for matrix C
        int*   d_C_rowPtr;
        int*   d_C_colIdx;
        float* d_C_values;
        CUDA_ERROR(cudaMalloc(&d_C_rowPtr, (A.rows() + 1) * sizeof(int)));
        CUDA_ERROR(cudaMalloc(&d_C_colIdx, nnzC * sizeof(int)));
        CUDA_ERROR(cudaMalloc(&d_C_values, nnzC * sizeof(T)));

        // Set pointers for matrix C
        CUSPARSE_ERROR(
            cusparseCsrSetPointers(matC, d_C_rowPtr, d_C_colIdx, d_C_values));

        // Phase 3: Compute actual values
        CUSPARSE_ERROR(cusparseSpGEMM_copy(handle,
                                           operation,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           A.m_spdescr,
                                           B.m_spdescr,
                                           &beta,
                                           matC,
                                           cuda_type<T>(),
                                           CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc));


        // First, copy the computed data to host to filter zeros
        int*   h_C_rowPtr = new int[A_rows + 1];
        int*   h_C_colIdx = new int[nnzC];
        float* h_C_values = new float[nnzC];

        CUDA_ERROR(cudaMemcpy(h_C_rowPtr,
                              d_C_rowPtr,
                              (A_rows + 1) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(h_C_colIdx,
                              d_C_colIdx,
                              nnzC * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(h_C_values,
                              d_C_values,
                              nnzC * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Count actual non-zeros and create filtered arrays
        const float        ZERO_THRESHOLD = 1e-6f;
        std::vector<int>   filtered_rowPtr(A_rows + 1, 0);
        std::vector<int>   filtered_colIdx;
        std::vector<float> filtered_values;
        filtered_colIdx.reserve(nnzC);
        filtered_values.reserve(nnzC);

        // Process first row pointer
        filtered_rowPtr[0] = 0;

        // Filter out zeros and build new CSR structure
        int actual_nnz = 0;
        for (int i = 0; i < A_rows; i++) {
            int row_start = h_C_rowPtr[i];
            int row_end   = h_C_rowPtr[i + 1];

            for (int j = row_start; j < row_end; j++) {
                if (std::abs(h_C_values[j]) > ZERO_THRESHOLD ||
                    h_C_values[j] != 0.0f) {
                    filtered_colIdx.push_back(h_C_colIdx[j]);
                    filtered_values.push_back(h_C_values[j]);
                    actual_nnz++;
                }
            }
            filtered_rowPtr[i + 1] = actual_nnz;
        }

        // Create new CSR object
        CSR result;
        result.num_rows = A_rows;

        // Allocate new memory with correct sizes
        result.non_zeros = actual_nnz;
        cudaMallocManaged(&result.row_ptr, (A_rows + 1) * sizeof(int));
        cudaMallocManaged(&result.value_ptr, actual_nnz * sizeof(int));
        cudaMallocManaged(&result.data_ptr, actual_nnz * sizeof(float));

        // Copy filtered data to result CSR
        cudaMemcpy(result.row_ptr,
                   filtered_rowPtr.data(),
                   (A_rows + 1) * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(result.value_ptr,
                   filtered_colIdx.data(),
                   actual_nnz * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(result.data_ptr,
                   filtered_values.data(),
                   actual_nnz * sizeof(float),
                   cudaMemcpyHostToDevice);


        // Cleanup host memory
        delete[] h_C_rowPtr;
        delete[] h_C_colIdx;
        delete[] h_C_values;

        // Cleanup device memory
        CUSPARSE_ERROR(cusparseDestroySpMat(matA));
        CUSPARSE_ERROR(cusparseDestroySpMat(matB));
        CUSPARSE_ERROR(cusparseDestroySpMat(matC));
        CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(spgemmDesc));
        CUSPARSE_ERROR(cusparseDestroy(handle));

        CUDA_ERROR(cudaFree(dBuffer1));
        CUDA_ERROR(cudaFree(dBuffer2));
        CUDA_ERROR(cudaFree(d_C_rowPtr));
        CUDA_ERROR(cudaFree(d_C_colIdx));
        CUDA_ERROR(cudaFree(d_C_values));


        return result;
    }
#endif
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

    void cycle(int              level,
               GMG<T>&          gmg,
               SparseMatrix<T>& A,
               DenseMatrix<T>&  f,  // rhs
               DenseMatrix<T>&  v)   // x
    {
        constexpr int numCols = 3;

        // pre-smoothing
        m_coarse_solver[level].template solve<numCols>(
            A, f, v, m_num_pre_relax);


        // calc residual
        calc_residual<numCols>(A, v, f, m_r[level]);

        // restrict residual TODO


        // recurse TODO
        if (level == gmg.m_num_levels - 1) {
            // the coarsest level
        } else {
            // solve();
        }


        // prolong TODO


        // post-smoothing
        m_coarse_solver[level].template solve<numCols>(
            A, f, v, m_num_post_relax);
    }


    /**
     * @brief compute r = f - A.v
     * @param A
     * @param v
     * @param f
     * @param r
     */
    template <int numCol>
    void calc_residual(const SparseMatrix<T>& A,
                       const DenseMatrix<T>&  v,
                       const DenseMatrix<T>&  f,
                       DenseMatrix<T>&        r)
    {
        constexpr uint32_t blockThreads = 256;

        uint32_t blocks = DIVIDE_UP(A.rows(), blockThreads);


        for_each_item<<<blocks, blockThreads>>>(
            A.rows(), [=] __device__(int row) mutable {
                T av[numCol];

                for (int c = 0; c < numCol; ++c) {
                    av[c] = 0;
                }

                int start = A.row_ptr()[row];
                int stop  = A.row_ptr()[row + 1];

                for (int j = start; j < stop; ++j) {

                    const int col = A.col_idx()[j];

                    const T val = A.get_val_at(j);

                    for (int c = 0; c < numCol; ++c) {
                        av[c] += val * v(col, c);
                    }
                }

                for (int c = 0; c < numCol; ++c) {
                    r(row, c) = f(row, c) - av[c];
                }
            });
    }
};
}  // namespace rxmesh