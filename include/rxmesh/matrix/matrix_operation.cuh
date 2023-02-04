#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/rxmesh_static.h"

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
 * @brief solve the Ax=b for x where x and b are all array
 */
template <typename T>
void spmat_linear_solve(rxmesh::SparseMatrix<T> A_mat,
                        T*                      B_arr,
                        T*                      X_arr,
                        rxmesh::Solver          solver,
                        rxmesh::Reorder         reorder)
{
    A_mat.create_cusolver_sphandle();
    cusparse_linear_solver_wrapper(solver,
                                   reorder,
                                   A_mat.m_cusolver_sphandles,
                                   A_mat.m_descr,
                                   A_mat.m_row_size,
                                   A_mat.m_col_size,
                                   A_mat.m_nnz,
                                   A_mat.m_d_row_ptr,
                                   A_mat.m_d_col_idx,
                                   A_mat.m_d_val,
                                   B_arr,
                                   X_arr);
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
    // A_mat.create_cusolver_sphandle();
    for (int i = 0; i < B_mat.m_col_size; ++i) {
        cusparse_linear_solver_wrapper(solver,
                                       reorder,
                                       A_mat.m_cusolver_sphandle,
                                       A_mat.m_descr,
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

/**
 * @brief wrap up the cusolver api for solving linear systems. This is a lower
 * level api
 */
template <typename T>
void cusparse_linear_solver_wrapper(const rxmesh::Solver  solver,
                                    const rxmesh::Reorder reorder,
                                    cusolverSpHandle_t    handle,
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
            CUSOLVER_ERROR(cusolverSpScsrlsvchol(handle,
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
                                                 &singularity));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrlsvchol(handle,
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
                                                 &singularity));
        }

    } else if (solver == Solver::QR) {
        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrlsvqr(handle,
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
                                               &singularity));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrlsvqr(handle,
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
                                               &singularity));
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
 * @brief do the sparse matrix dense matrix multiplication using sparse matrix
 * array multiplication in a column wise way
 */
template <typename T>
void spmat_denmat_mul_cw(rxmesh::SparseMatrix<T> A_mat,
                         rxmesh::DenseMatrix<T>  B_mat,
                         rxmesh::DenseMatrix<T>  C_mat)
{
    for (int i = 0; i < B_mat.m_col_size; ++i) {
        spmat_arr_mul(A_mat, B_mat.col_data(i), C_mat.col_data(i));
    }
}

}  // namespace rxmesh