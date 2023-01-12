#pragma once

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "dense_matrix.cuh"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "sparse_matrix.cuh"

#include "cusolverSp.h"
#include "cusparse.h"

namespace rxmesh {

/**
 * @brief The enum class for choosing different solver types
 */
enum class Solver
{
    CHOL = 0,
    LU   = 1,
    QR   = 2
};

/**
 * @brief The enum class for choosing different reorder types
 * NONE for No Reordering Applied, SYMRCM for Symmetric Reverse Cuthill-McKee
 * permutation, SYMAMD for Symmetric Approximate Minimum Degree Algorithm based
 * on Quotient Graph, NSTDIS for Nested Dissection
 */
enum class Reorder
{
    NONE   = 0,
    SYMRCM = 1,
    SYMAMD = 2,
    NSTDIS = 3
};

static int reorder_to_int(const Reorder& reorder)
{
    switch (reorder) {
        case Reorder::NONE:
            return 0;
        case Reorder::SYMRCM:
            return 1;
        case Reorder::SYMAMD:
            return 2;
        case Reorder::NSTDIS:
            return 3;
        default: {
            RXMESH_ERROR("reorder_to_int() unknown input reorder");
            return 0;
        }
    }
}

/**
 * @brief for transpose the dense matrix using cublas
 */
template <typename T>
void denmat_transpose(rxmesh::DenseMatrix<T> den_mat)
{
    T* d_rt_arr;
    cudaMalloc(&d_rt_arr, den_mat.bytes());

    float const alpha(1.0);
    float const beta(0.0);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgeam(handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                den_mat.m_row_size,
                den_mat.m_col_size,
                &alpha,               // 0
                den_mat.data(),       // A_arr
                den_mat.m_col_size,   // ld_a
                &beta,                // 0
                den_mat.data(),       // B_arr
                den_mat.m_row_size,   // ld_b
                d_rt_arr,             // rt_arr
                den_mat.m_row_size);  // ld_cm
    cublasDestroy(handle);

    den_mat.m_d_val = d_rt_arr;
    // TODO cont;
}

/**
 * @brief solve the Ax=b for x where x and b are all array
 */
template <typename T>
void spmat_linear_solve(rxmesh::SparseMatrix<T> A_mat,
                        T*                      B_arr,
                        T*                      X_arr,
                        rxmesh::Solver          solver,
                        rxmesh::Reorder         reorder 
                        //cudaStream_t       stream = null
                        )
{
    cusolverSpHandle_t handle         = NULL;
    cusparseHandle_t   cusparseHandle = NULL;
    cudaStream_t       stream         = NULL;
    cusparseMatDescr_t descrA         = NULL;

    cusolverSpCreate(&handle);
    cusparseCreate(&cusparseHandle);

    cudaStreamCreate(&stream);
    cusolverSpSetStream(handle, stream);
    cusparseSetStream(cusparseHandle, stream);

    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    cusparse_linear_solver_wrapper(solver,
                                   reorder,
                                   handle,
                                   cusparseHandle,
                                   descrA,
                                   A_mat.m_row_size,
                                   A_mat.m_col_size,
                                   A_mat.m_nnz,
                                   A_mat.m_d_row_ptr,
                                   A_mat.m_d_col_idx,
                                   A_mat.m_d_val,
                                   B_arr,
                                   X_arr);

    cusolverSpDestroy(handle);
    cusparseDestroy(cusparseHandle);
    cudaStreamDestroy(stream);
    cusparseDestroyMatDescr(descrA);
}

/**
 * @brief solve the AX=B for X where X and B are all dense matrix and we would
 * solve it in a column wise manner
 */
template <typename T>
void spmat_linear_solve(rxmesh::SparseMatrix<T> A_mat,
                        rxmesh::DenseMatrix<T>  B_mat,
                        rxmesh::DenseMatrix<T>  X_mat,
                        rxmesh::Solver          solver,
                        rxmesh::Reorder         reorder)
{
    cusolverSpHandle_t handle         = NULL;
    cusparseHandle_t   cusparseHandle = NULL;
    cudaStream_t       stream         = NULL;
    cusparseMatDescr_t descrA         = NULL;

    cusolverSpCreate(&handle);
    cusparseCreate(&cusparseHandle);

    cudaStreamCreate(&stream);
    cusolverSpSetStream(handle, stream);
    cusparseSetStream(cusparseHandle, stream);

    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    for (int i = 0; i < B_mat.m_col_size; ++i) {
        cusparse_linear_solver_wrapper(solver,
                                       reorder,
                                       handle,
                                       cusparseHandle,
                                       descrA,
                                       A_mat.m_row_size,
                                       A_mat.m_col_size,
                                       A_mat.m_nnz,
                                       A_mat.m_d_row_ptr,
                                       A_mat.m_d_col_idx,
                                       A_mat.m_d_val,
                                       B_mat.ld_data(i),
                                       X_mat.ld_data(i));
    }

    cusolverSpDestroy(handle);
    cusparseDestroy(cusparseHandle);
    cudaStreamDestroy(stream);
    cusparseDestroyMatDescr(descrA);
}

/**
 * @brief wrap up the cusolver api for solving linear systems. This is a lower
 * level api
 */
template <typename T>
void cusparse_linear_solver_wrapper(const rxmesh::Solver  solver,
                                    const rxmesh::Reorder reorder,
                                    cusolverSpHandle_t    handle,
                                    cusparseHandle_t      cusparseHandle,
                                    cusparseMatDescr_t    descrA,
                                    int                   rowsA,
                                    int                   colsA,
                                    int                   nnzA,
                                    int*                  d_csrRowPtrA,
                                    int*                  d_csrColIndA,
                                    T*                    d_csrValA,
                                    T*                    d_b,
                                    T*                    d_x)
{
    if constexpr ((!std::is_same_v<T, float>)&&(!std::is_same_v<T, double>)) {
        RXMESH_ERROR(
            "Unsupported type for cusparse: {}"
            "Only float and double are supported",
            typeid(T).name());
    }

    double tol         = 1.e-12;
    int    singularity = 0; /* -1 if A is invertible under tol. */

    /* solve B*z = Q*b */
    if (solver == Solver::CHOL) {
        if constexpr (std::is_same_v<T, float>) {
            cusolverSpScsrlsvchol(handle,
                                  rowsA,
                                  nnzA,
                                  descrA,
                                  d_csrValA,
                                  d_csrRowPtrA,
                                  d_csrColIndA,
                                  d_b,
                                  tol,
                                  reorder_to_int(reorder),
                                  d_x,
                                  &singularity);
        }

        if constexpr (std::is_same_v<T, double>) {
            cusolverSpDcsrlsvchol(handle,
                                  rowsA,
                                  nnzA,
                                  descrA,
                                  d_csrValA,
                                  d_csrRowPtrA,
                                  d_csrColIndA,
                                  d_b,
                                  tol,
                                  reorder_to_int(reorder),
                                  d_x,
                                  &singularity);
        }

    } else if (solver == Solver::QR) {
        if constexpr (std::is_same_v<T, float>) {
            cusolverSpScsrlsvqr(handle,
                                rowsA,
                                nnzA,
                                descrA,
                                d_csrValA,
                                d_csrRowPtrA,
                                d_csrColIndA,
                                d_b,
                                tol,
                                reorder_to_int(reorder),
                                d_x,
                                &singularity);
        }

        if constexpr (std::is_same_v<T, double>) {
            cusolverSpDcsrlsvqr(handle,
                                rowsA,
                                nnzA,
                                descrA,
                                d_csrValA,
                                d_csrRowPtrA,
                                d_csrColIndA,
                                d_b,
                                tol,
                                reorder_to_int(reorder),
                                d_x,
                                &singularity);
        }
    } else {
        RXMESH_ERROR(
            "Only Solver::CHOL and Solver::QR is supported, use CUDA 12.x for "
            "Solver::LU");
    }
    cudaDeviceSynchronize();

    if (0 <= singularity) {
        RXMESH_WARN("WARNING: the matrix is singular at row {} under tol ({})",
                    singularity,
                    tol);
    }
}

/**
 * @brief wrap up the cusparse api for sparse matrix dense matrix
 * multiplication.
 */
template <typename T>
void spmat_denmat_mul(rxmesh::SparseMatrix<T> A_mat,
                      rxmesh::DenseMatrix<T>  B_mat,
                      rxmesh::DenseMatrix<T>  C_mat)
{
    float alpha = 1.0f;
    float beta  = 0.0f;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;

    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA,
                      A_mat.m_row_size,
                      A_mat.m_col_size,
                      A_mat.m_nnz,
                      A_mat.m_d_row_ptr,
                      A_mat.m_d_col_idx,
                      A_mat.m_d_val,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_32F);
    // Create dense matrix B
    cusparseCreateDnMat(&matB,
                        B_mat.m_row_size,
                        B_mat.m_col_size,
                        B_mat.lead_dim(), // lead_dim < row_size
                        B_mat.data(),
                        CUDA_R_32F,
                        CUSPARSE_ORDER_COL);
    // Create dense matrix C
    cusparseCreateDnMat(&matC,
                        C_mat.m_row_size,
                        C_mat.m_col_size,
                        C_mat.lead_dim(),
                        C_mat.data(),
                        CUDA_R_32F,
                        CUSPARSE_ORDER_COL);
    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            matA,
                            matB,
                            &beta,
                            matC,
                            CUDA_R_32F,
                            CUSPARSE_SPMM_ALG_DEFAULT,
                            &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute SpMM
    cusparseSpMM(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha,
                 matA,
                 matB,
                 &beta,
                 matC,
                 CUDA_R_32F,
                 CUSPARSE_SPMM_ALG_DEFAULT,
                 dBuffer);

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
}


/**
 * @brief wrap up the cusparse api for sparse matrix array
 * multiplication.
 */
// only works for float
template <typename T>
void spmat_arr_mul(rxmesh::SparseMatrix<T> sp_mat, T* in_arr, T* rt_arr)
{
    const float minus_one = -1.0f;
    const float one       = 1.0f;
    const float zero      = 0.0f;

    cusparseHandle_t     handle     = NULL;
    void*                buffer     = NULL;
    size_t               bufferSize = 0;
    cusparseSpMatDescr_t sp_mat_des = NULL;
    cusparseDnVecDescr_t vecx       = NULL;
    cusparseDnVecDescr_t vecy       = NULL;

    cusparseCreate(&handle);
    cusparseCreateCsr(&sp_mat_des,
                      sp_mat.m_row_size,
                      sp_mat.m_col_size,
                      sp_mat.m_nnz,
                      sp_mat.m_d_row_ptr,
                      sp_mat.m_d_col_idx,
                      sp_mat.m_d_val,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_32F);
    cusparseCreateDnVec(&vecx, sp_mat.m_col_size, in_arr, CUDA_R_32F);
    cusparseCreateDnVec(&vecy, sp_mat.m_row_size, rt_arr, CUDA_R_32F);

    cusparseSpMV_bufferSize(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &one,
                            sp_mat_des,
                            vecx,
                            &zero,
                            vecy,
                            CUDA_R_32F,
                            CUSPARSE_SPMV_ALG_DEFAULT,
                            &bufferSize);
    cudaMalloc(&buffer, bufferSize);

    cusparseSpMV(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &one,
                 sp_mat_des,
                 vecx,
                 &zero,
                 vecy,
                 CUDA_R_32F,
                 CUSPARSE_SPMV_ALG_DEFAULT,
                 buffer);

    cusparseDestroySpMat(sp_mat_des);
    cusparseDestroyDnVec(vecx);
    cusparseDestroyDnVec(vecy);
    cusparseDestroy(handle);
}

/**
 * @brief do the sparse matrix dense matrix multiplication using sparse matrix
 * array multiplication in a column wise way
 */
template <typename T>
void spmat_denmat_mul_cw(rxmesh::SparseMatrix<T> A_mat,
                         rxmesh::DenseMatrix<T>  B_mat,
                         rxmesh::DenseMatrix<T>  C_mat)
{
    for (int i = 0; i < B_mat.m_col_size; ++i) {
        spmat_arr_mul(A_mat, B_mat.ld_data(i), C_mat.ld_data(i));
    }
}

}  // namespace rxmesh