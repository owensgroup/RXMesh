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
                                       B_mat.data(i),
                                       X_mat.data(i));
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
                                    T*               d_csrValA,
                                    T*               d_b,
                                    T*               d_x)
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
                                  reorder,
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

}  // namespace rxmesh