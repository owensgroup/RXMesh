//
// Created by behrooz on 28/09/22.
//


#pragma once

#include <Eigen/SparseCore>

#include "parth/parth.h"
#include "cholmod.h"
#include "cholmod_core.h"

#include <string>
#include <vector>

namespace PARTH {

class ParthSolverAPI
{
public:
    ///@brief Solver configs
    int num_cores = 10;
    int num_sockets = 2;
    bool verbose = true;
    std::vector<bool> symbolic_region_catch;
    ParthAPI parth;
    std::vector<int> perm_vector;

    ///@brief Basic Storages for A
    Eigen::SparseMatrix<double> A;
    int *Ap, *Ai;
    double *Ax;
    int A_nnz, A_n;
    ///@bierf Hessian related variables
    cholmod_common A_cm;
    cholmod_sparse *chol_A;

    cholmod_factor *chol_L;
    cholmod_dense *chol_b, *chol_sol;
    cholmod_dense *chol_Z, *chol_E, *chol_Y, *chol_R;

    void *chol_Ap_tmp, *chol_Ai_tmp, *chol_Ax_tmp, *chol_b_tmp;

    // supernode Etree variables
    std::vector<int> super_parents;
    std::vector<int> tree_ptr;
    std::vector<int> tree_set;


    ///@breif Graph with no diagonal for Parth
    std::vector<int> Gp;
    std::vector<int> Gi;

    ///@breif Parallelization variables
    int parallel_levels = 0;
    std::vector<int> level_ptr;
    std::vector<int> part_ptr;
    std::vector<int> supernode_idx;
    std::vector<int> super_nz;

    ParthSolverAPI(void);
    ~ParthSolverAPI(void);
    //===========================================================================
    //========================== Analysis Functions  ============================
    //===========================================================================

    ///--------------------------------------------------------------------------
    /// analyze - the entry point for matrix inspection before factorization
    ///--------------------------------------------------------------------------
    void analyze(std::vector<int>& user_defined_perm);
    /// --------------------------------------------------------------------------
    /// cholmod_analyze_custom:  order and analyze (simplicial or supernodal) */
    /// --------------------------------------------------------------------------
    /// Orders and analyzes A, AA', PAP', or PAA'P' and returns a symbolic factor
    /// that can later be passed to cholmod_factorize.
    cholmod_factor *cholmod_analyze_custom(
        /* ---- input ---- */
        cholmod_sparse *A, /* matrix to order and analyze */
        /* --------------- */
        cholmod_common *Common);

    /// ==========================================================================
    /// === cholmod_analyze_p ====================================================
    /// ==========================================================================
    /// */ Orders and analyzes A, AA', PAP', PAA'P', FF', or PFF'P and returns a
    /// symbolic factor that can later be passed to cholmod_factorize, where
    /// F = A(:,fset) if fset is not NULL and A->stype is zero.
    /// UserPerm is tried if non-NULL.
    cholmod_factor *cholmod_analyze_p_custom(
        /* ---- input ---- */
        cholmod_sparse *A, /* matrix to order and analyze */
        int *UserPerm,     /* user-provided permutation, size A->nrow */
        int *fset,         /* subset of 0:(A->ncol)-1 */
        size_t fsize,      /* size of fset */
        /* --------------- */
        cholmod_common *Common);

    /// --------------------------------------------------------------------------
    /// cholmod_analyze_p2:  analyze for sparse Cholesky or sparse QR */
    /// --------------------------------------------------------------------------
    cholmod_factor *cholmod_analyze_p2_custom(
        /* ---- input ---- */
        int for_whom,      /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                           FOR_CHOLESKY (1): for Cholesky (GPU or not)
                           FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
        cholmod_sparse *A, /* matrix to order and analyze */
        int *UserPerm,     /* user-provided permutation, size A->nrow */
        int *fset,         /* subset of 0:(A->ncol)-1 */
        size_t fsize,      /* size of fset */
        /* --------------- */
        cholmod_common *Common);

    int cholmod_analyze_ordering_custom(
        /* ---- input ---- */
        cholmod_sparse *A, /* matrix to analyze */
        int ordering,      /* ordering method used */
        int *Perm,         /* size n, fill-reducing permutation to analyze */
        int *fset,         /* subset of 0:(A->ncol)-1 */
        size_t fsize,      /* size of fset */
        /* ---- output --- */
        int *Parent,   /* size n, elimination tree */
        int *Post,     /* size n, postordering of elimination tree */
        int *ColCount, /* size n, nnz in each column of L */
        /* ---- workspace  */
        int *First, /* size n workspace for cholmod_postorder */
        int *Level, /* size n workspace for cholmod_postorder */
        /* --------------- */
        cholmod_common *Common);

    /* --------------------------------------------------------------------------
     */
    /* cholmod_super_symbolic */
    /* --------------------------------------------------------------------------
     */

    /* Analyzes A, AA', or A(:,f)*A(:,f)' in preparation for a supernodal numeric
     * factorization.  The user need not call this directly; cholmod_analyze is
     * a "simple" wrapper for this routine.
     */

    int cholmod_super_symbolic_custom(
        /* ---- input ---- */
        cholmod_sparse *A, /* matrix to analyze */
        cholmod_sparse *F, /* F = A' or A(:,f)' */
        int *Parent,       /* elimination tree */
        /* ---- in/out --- */
        cholmod_factor *L, /* simplicial symbolic on input,
                            * supernodal symbolic on output */
        /* --------------- */
        cholmod_common *Common);

    /* --------------------------------------------------------------------------
     */
    /* cholmod_super_symbolic2 */
    /* --------------------------------------------------------------------------
     */

    /* Analyze for supernodal Cholesky or multifrontal QR */

    int cholmod_super_symbolic2_custom(
        /* ---- input ---- */
        int for_whom,      /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                                   FOR_CHOLESKY (1): for Cholesky (GPU or not)
                                   FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
        cholmod_sparse *A, /* matrix to analyze */
        cholmod_sparse *F, /* F = A' or A(:,f)' */
        int *Parent,       /* elimination tree */
        /* ---- in/out --- */
        cholmod_factor *L, /* simplicial symbolic on input,
                            * supernodal symbolic on output */
        /* --------------- */
        cholmod_common *Common);

    void computeFirstchild(int n, int *Ap, int *Ai);

    void createParallelSchedule();

    void createSpTRSVParallelSchedule();

    void createReuseParallelSchedule();

    int computeETreeCost(int *super, int *tree_ptr, int *tree_set,
                         int current_node);

    //===========================================================================
    //======================= Factorization Functions  ==========================
    //===========================================================================
    ///--------------------------------------------------------------------------
    /// factorize - the entry point for matrix factorization
    ///--------------------------------------------------------------------------
    bool factorize();

    ///---------------------------------------------------------------------
    /// These are main supernodal functions adapted from cholmod
    ///---------------------------------------------------------------------
    int cholmod_factorize_custom(
        cholmod_sparse *A,     ///<[in] matrix to factorize
        cholmod_factor *L,     ///<[out] resulting factorization
        cholmod_common *Common ///<[in/out]
    );

    ///---------------------------------------------------------------------
    /// These are main supernodal functions adapted from cholmod
    ///---------------------------------------------------------------------
    int cholmod_factorize_p_custom(
        cholmod_sparse *A,     ///<[in] matrix to factorize */
        double beta[2],        ///<[in] factorize beta*I+A or beta*I+A'*A */
        int *fset,             ///<[in] subset of 0:(A->ncol)-1 */
        size_t fsize,          ///<[in] size of fset */
        cholmod_factor *L,     ///<[in/out] resulting factorization */
        cholmod_common *Common ///<[in/out]
    );

    int cholmod_super_numeric_custom(
        cholmod_sparse *A, ///<[in] matrix to factorize */
        cholmod_sparse *F, ///<[in] F = A' or A(:,f)' */
        double
            beta[2], ///<[in] beta*I is added to diagonal of matrix to factorize
        cholmod_factor *L,     ///<[in/out]factorization */
        cholmod_common *Common ///<[in/out]
    );



    //===========================================================================
    //============================ Solve Functions  =============================
    //===========================================================================

    cholmod_dense *cholmod_solve_custom(
        /* ---- input ---- */
        int sys,           /* system to solve */
        cholmod_factor *L, /* factorization to use */
        cholmod_dense *B,  /* right-hand-side */
        /* --------------- */
        cholmod_common *Common);

    int cholmod_solve2_custom /* returns TRUE on success, FALSE on failure */
        (
            /* ---- input ---- */
            int sys,           /* system to solve */
            cholmod_factor *L, /* factorization to use */
            cholmod_dense *B,  /* right-hand-side */
            cholmod_sparse *Bset,
            /* ---- output --- */
            cholmod_dense **X_Handle, /* solution, allocated if need be */
            cholmod_sparse **Xset_Handle,
            /* ---- workspace  */
            cholmod_dense **Y_Handle, /* workspace, or NULL */
            cholmod_dense **E_Handle, /* workspace, or NULL */
            /* --------------- */
            cholmod_common *Common);

    int cholmod_super_ltsolve_custom /* TRUE if OK, FALSE if BLAS overflow occured
                                      */
        (
            /* ---- input ---- */
            cholmod_factor *L, /* factor to use for the backsolve */
            /* ---- output ---- */
            cholmod_dense *X, /* b on input, solution to L'x=b on output */
            /* ---- workspace ---- */
            cholmod_dense *E, /* workspace of size nrhs*(L->maxesize) */
            /* --------------- */
            cholmod_common *Common);

    int cholmod_super_lsolve_custom /* TRUE if OK, FALSE if BLAS overflow occured
                                     */
        (
            /* ---- input ---- */
            cholmod_factor *L, /* factor to use for the forward solve */
            /* ---- output ---- */
            cholmod_dense *X, /* b on input, solution to Lx=b on output */
            /* ---- workspace ---- */
            cholmod_dense *E, /* workspace of size nrhs*(L->maxesize) */
            /* --------------- */
            cholmod_common *Common);

    ///--------------------------------------------------------------------------
    /// solve - Entry point for solve function
    ///--------------------------------------------------------------------------
    void solve(Eigen::VectorXd &rhs,   ///<[in] right hand side of Ax=b
               Eigen::VectorXd &result ///<[out] the unknown x in Ax=b
    );

    ///--------------------------------------------------------------------------
    /// cholmodSolve - basic cholmod solve
    ///--------------------------------------------------------------------------
    void cholmodSolve(Eigen::VectorXd &rhs,   ///<[in] right hand side of Ax=b
                      Eigen::VectorXd &result ///<[out] the unknown x in Ax=b
    );

    void parth_solver_clean_memory();
    void setMatrix(int* p, int* i, double* x, int A_N, int NNZ);
    void removeDiagonal(int N,
            int* Ap, int* Ai, std::vector<int>& Gp, std::vector<int>& Gi);
};
} // namespace PARTH