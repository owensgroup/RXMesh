//
// Created by behrooz on 25/10/22.
//
#include "parth_solver.h"


namespace PARTH_SOLVER {

ParthSolverAPI::ParthSolverAPI(void)
{
    ///@brief Init options to a good default value
    ///@bierf Hessian related variables
    Ap    = nullptr;
    Ai    = nullptr;
    Ax    = nullptr;
    A_nnz = 0;
    A_n   = 0;
    cholmod_start(&A_cm);
    chol_A = nullptr;
    chol_Z = nullptr;
    chol_Y = nullptr;
    chol_E = nullptr;
    chol_R = nullptr;

    chol_L = nullptr;

    chol_b      = nullptr;
    chol_sol    = nullptr;
    chol_Ap_tmp = nullptr;
    chol_Ai_tmp = nullptr;
    chol_Ax_tmp = nullptr;
    chol_b_tmp  = nullptr;
}

ParthSolverAPI::~ParthSolverAPI()
{
    this->parth_solver_clean_memory();
}

void ParthSolverAPI::parth_solver_clean_memory()
{
    if (chol_A) {
        chol_A->p = chol_Ap_tmp;
        chol_A->i = chol_Ai_tmp;
        chol_A->x = chol_Ax_tmp;
        cholmod_free_sparse(&chol_A, &A_cm);
    }

    if (chol_b) {
        chol_b->x = chol_b_tmp;
        cholmod_free_dense(&chol_b, &A_cm);
    }

    if (chol_sol) {
        cholmod_free_dense(&chol_sol, &A_cm);
    }

    if (chol_Z) {
        cholmod_free_dense(&chol_Z, &A_cm);
    }

    if (chol_Y) {
        cholmod_free_dense(&chol_Y, &A_cm);
    }

    if (chol_E) {
        cholmod_free_dense(&chol_E, &A_cm);
    }

    if (chol_R) {
        cholmod_free_dense(&chol_R, &A_cm);
    }

    cholmod_finish(&A_cm);
}

void ParthSolverAPI::setMatrix(int* p, int* i, double* x, int A_N, int NNZ)
{
    assert(p[A_N] == NNZ);
    this->A_n   = A_N;
    this->A_nnz = NNZ;

    this->parth_solver_clean_memory();

    if (!chol_A) {
        chol_A = cholmod_allocate_sparse(
            A_n, A_n, A_nnz, true, true, -1, CHOLMOD_REAL, &A_cm);
        this->chol_Ap_tmp = chol_A->p;
        this->chol_Ax_tmp = chol_A->x;
        this->chol_Ai_tmp = chol_A->i;
        // -1: upper right part will be ignored during computation
    }

    chol_A->p = p;
    chol_A->i = i;
    chol_A->x = x;

    //Prepare the mesh for parth
    removeDiagonal(A_N, p, i, Gp, Gi);
    int G_N = Gp.size() - 1;
    parth.setMesh(G_N, Gp.data(), Gi.data());
}


void ParthSolverAPI::removeDiagonal(int N,
        int* Ap, int* Ai, std::vector<int>& Gp, std::vector<int>& Gi)
{
    int dim = 1;
    std::vector<std::tuple<int, int>> coefficients;
    for (int c = 0; c < N; c += dim) {
        assert((Ap[c + 1] - Ap[c]) % dim == 0);
        for (int r_ptr = Ap[c]; r_ptr < Ap[c + 1]; r_ptr += dim) {
            int r = Ai[r_ptr];
            int mesh_c = c / dim;
            int mesh_r = r / dim;
            if (mesh_c != mesh_r) {
                coefficients.emplace_back(mesh_c, mesh_r);
                coefficients.emplace_back(mesh_r, mesh_c);
            }
        }
    }

    //Remove duplicates
    std::sort(coefficients.begin(), coefficients.end());
    coefficients.erase(std::unique(coefficients.begin(), coefficients.end()), coefficients.end());
    Gp.resize(N + 1);
    for (int i = 0; i < coefficients.size(); i++) {
        Gp[std::get<0>(coefficients[i]) + 1]++;
    }
    for (int i = 1; i < Gp.size(); i++) {
        Gp[i] += Gp[i - 1];
    }

    Gi.resize(Gp.back());
    std::vector<int> Mp_vec_cnt(Gp.size(), 0);
    for (int i = 0; i < coefficients.size(); i++) {
        int row = std::get<0>(coefficients[i]);
        int col = std::get<1>(coefficients[i]);
        Gi[Gp[row] + Mp_vec_cnt[row]] = col;
        Mp_vec_cnt[row]++;
    }


#ifndef NDEBUG
    //Make sure that each row is sorted
    for(int r = 0; r < Gp.size() - 1; r++) {
        for (int i = Gp[r]; i < Gp[r + 1] - 1; i++) {
            assert(Gi[i] < Gi[i + 1]);
        }
    }
#endif
}
}  // namespace PARTH
