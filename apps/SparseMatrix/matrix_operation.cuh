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

template <typename E>
constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept
{
    return static_cast<typename std::underlying_type<E>::type>(e);
}

__global__ void print_device(float* arr, int size)
{
    for (int i = 0; i < size; ++i) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

namespace rxmesh {
enum Solver
{
    CHOL = 0,
    LU   = 1,
    QR   = 2
};

enum Reorder
{
    NONE   = 0,
    SYMRCM = 1,
    SYMAMD = 2,
    NSTDIS = 3
};

template <typename T>
void spmat_linear_solve(rxmesh::SparseMatInfo<T> A_mat,
                        T*                       B_arr,
                        T*                       X_arr,
                        rxmesh::Solver           solver,
                        rxmesh::Reorder          reorder)
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
}

template <typename T>
void spmat_linear_solve(rxmesh::SparseMatInfo<T> A_mat,
                        rxmesh::DenseMatInfo<T>  B_mat,
                        rxmesh::DenseMatInfo<T>  X_mat,
                        rxmesh::Solver           solver,
                        rxmesh::Reorder          reorder)
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
                                       B_mat.col_data(i),
                                       X_mat.col_data(i));
    }
}

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
            printf("\nHit\n \n");

            print_device<<<1, 1>>>(d_x, (int)rowsA);
            cudaDeviceSynchronize();
            cusolverSpScsrlsvchol(handle,
                                  rowsA,
                                  nnzA,
                                  descrA,
                                  d_csrValA,
                                  d_csrRowPtrA,
                                  d_csrColIndA,
                                  d_b,
                                  tol,
                                  reorder,
                                  d_x,
                                  &singularity);

            print_device<<<1, 1>>>(d_x, (int)rowsA);
            cudaDeviceSynchronize();

            printf("\n End \n \n");
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
                                  reorder,
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
                                reorder,
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
                                reorder,
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


template <typename T>
void spmat_denmat_mul(rxmesh::SparseMatInfo<T> A_mat,
                      rxmesh::DenseMatInfo<T>  B_mat,
                      rxmesh::DenseMatInfo<T>  C_mat)
{
    float alpha = 1.0f;
    float beta  = 0.0f;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;

    printf("%d, %d, %d\n", B_mat.m_row_size, B_mat.m_col_size, B_mat.m_ld);

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
                        B_mat.m_ld,
                        B_mat.data(),
                        CUDA_R_32F,
                        CUSPARSE_ORDER_COL);
    // Create dense matrix C
    cusparseCreateDnMat(&matC,
                        C_mat.m_row_size,
                        C_mat.m_col_size,
                        C_mat.m_ld,
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

// only works for float
template <typename T>
void spmat_arr_mul(rxmesh::SparseMatInfo<T> sp_mat, T* in_arr, T* rt_arr)
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

template <typename T>
void spmat_denmat_mul_test(rxmesh::SparseMatInfo<T> A_mat,
                           rxmesh::DenseMatInfo<T>  B_mat,
                           rxmesh::DenseMatInfo<T>  C_mat)
{
    for (int i = 0; i < B_mat.m_col_size; ++i) {
        spmat_arr_mul(A_mat, B_mat.col_data(i), C_mat.col_data(i));
    }
}

}  // namespace rxmesh