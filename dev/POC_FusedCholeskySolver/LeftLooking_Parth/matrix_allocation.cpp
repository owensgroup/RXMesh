//
// Created by behrooz on 25/10/22.
//

#include "LeftLooking_Parth.h"

namespace PARTH {

void ParthSolver::setMatrix(const int* Ap, ///<[in] pointer array
    const int* Ai, ///<[in] index array
    const double* Ax, /// [in] value array
    const int n, /// [in] Number of rows/columns
    const int nnz /// [in] Number of nonzeros
)
{
    assert(n != 0);
    assert(nnz != 0);
    assert(Ap != nullptr);
    assert(Ai != nullptr);
    assert(Ax != nullptr);
    A.conservativeResize(n, n);
    A.reserve(nnz);
    std::copy(Ap, Ap + n + 1, A.outerIndexPtr());
    std::copy(Ai, Ai + nnz, A.innerIndexPtr());
    std::copy(Ax, Ax + nnz, A.valuePtr());
    this->A_nnz = nnz;
    this->A_n = n;
    this->Ap = A.outerIndexPtr();
    this->Ai = A.innerIndexPtr();
    this->Ax = A.valuePtr();

    if (!chol_A) {
        chol_A = cholmod_allocate_sparse(A_n, A_n,
            A_nnz, true, true, -1,
            CHOLMOD_REAL, &A_cm);
        chol_Ap_tmp = chol_A->p;
        chol_Ai_tmp = chol_A->i;
        chol_Ax_tmp = chol_A->x;
    }

    chol_A->p = this->Ap;
    chol_A->i = this->Ai;
    chol_A->x = this->Ax;

    this->hessian_perm.resize(A_n);
    this->mat_allocated = true;
}

void ParthSolver::setMatrix(const Eigen::SparseMatrix<double>& A)
{
    setMatrix(A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), A.rows(), A.nonZeros());
}

void ParthSolver::setMatrixPointers(int* Ap, ///<[in] pointer array
    int* Ai, ///<[in] index array
    double* Ax, /// [in] value array
    int n, /// [in] Number of rows/columns
    int nnz /// [in] Number of nonzeros
)
{
    assert(n != 0);
    assert(nnz != 0);
    assert(Ap != nullptr);
    assert(Ai != nullptr);
    assert(Ax != nullptr);

    this->A_nnz = nnz;
    this->A_n = n;
    this->Ap = Ap;
    this->Ai = Ai;
    this->Ax = Ax;

    if (!chol_A) {
        chol_A = cholmod_allocate_sparse(A_n, A_n,
            A_nnz, true, true, -1,
            CHOLMOD_REAL, &A_cm);
        chol_Ap_tmp = chol_A->p;
        chol_Ai_tmp = chol_A->i;
        chol_Ax_tmp = chol_A->x;
    }

    chol_A->p = this->Ap;
    chol_A->i = this->Ai;
    chol_A->x = this->Ax;

    this->hessian_perm.resize(A_n);

    this->mat_allocated = false;
}

void ParthSolver::setMatrixPointers(Eigen::SparseMatrix<double>& A)
{
    setMatrixPointers(A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), A.rows(), A.nonZeros());
}

void ParthSolver::setMeshPointers(int n, int* Mp, int* Mi)
{
    assert(Mp != nullptr);
    assert(Mi != nullptr);
    assert(n != 0);
    this->Mp = Mp;
    this->Mi = Mi;
    this->M_n = n;
    this->mesh_init = true;
}

}