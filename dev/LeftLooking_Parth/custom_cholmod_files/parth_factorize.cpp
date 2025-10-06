#include <unsupported/Eigen/SparseExtra>

#include "cholmod_cholesky.h"
#include "cholmod_core.h"
#include "cholmod_internal.h"
#include "parth_solver.h"

/* ========================================================================== */
/* === cholmod_factorize ==================================================== */
/* ========================================================================== */

/* Factorizes PAP' (or PAA'P' if A->stype is 0), using a factor obtained
 * from cholmod_analyze.  The analysis can be re-used simply by calling this
 * routine a second time with another matrix.  A must have the same nonzero
 * pattern as that passed to cholmod_analyze. */

namespace PARTH {

int ParthSolver::cholmod_factorize_custom(
    /* ---- input ---- */
    cholmod_sparse* A, /* matrix to factorize */
    /* ---- in/out --- */
    cholmod_factor* L, /* resulting factorization */
    /* --------------- */
    cholmod_common* Common)
{
    double zero[2];
    zero[0] = 0;
    zero[1] = 0;
    return cholmod_factorize_p_custom(A, zero, NULL, 0, L, Common);
}

/* ========================================================================== */
/* === cholmod_factorize_p ================================================== */
/* ========================================================================== */

/* Same as cholmod_factorize, but with more options. */

int ParthSolver::cholmod_factorize_p_custom(
    /* ---- input ---- */
    cholmod_sparse* A, /* matrix to factorize */
    double beta[2], /* factorize beta*I+A or beta*I+A'*A */
    Int* fset, /* subset of 0:(A->ncol)-1 */
    size_t fsize, /* size of fset */
    /* ---- in/out --- */
    cholmod_factor* L, /* resulting factorization */
    /* --------------- */
    cholmod_common* Common)
{
    cholmod_sparse *S, *F, *A1, *A2;
    Int nrow, ncol, stype, convert, n, nsuper, grow2, status;
    size_t s, t, uncol;
    int ok = TRUE;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    RETURN_IF_NULL_COMMON(FALSE);
    RETURN_IF_NULL(A, FALSE);
    RETURN_IF_NULL(L, FALSE);
    RETURN_IF_XTYPE_INVALID(A, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE);
    RETURN_IF_XTYPE_INVALID(L, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, FALSE);
    nrow = A->nrow;
    ncol = A->ncol;
    n = L->n;
    stype = A->stype;
    if (L->n != A->nrow) {
        ERROR(CHOLMOD_INVALID, "A and L dimensions do not match");
        return (FALSE);
    }
    if (stype != 0 && nrow != ncol) {
        ERROR(CHOLMOD_INVALID, "matrix invalid");
        return (FALSE);
    }
    DEBUG(CHOLMOD(dump_sparse)(A, "A for cholmod_factorize", Common));
    Common->status = CHOLMOD_OK;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace */
    /* ---------------------------------------------------------------------- */

    nsuper = (L->is_super ? L->nsuper : 0);
    uncol = ((stype != 0) ? 0 : ncol);

    /* s = 2*nrow + MAX (uncol, 2*nsuper) */
    s = CHOLMOD(mult_size_t)(nsuper, 2, &ok);
    s = MAX(uncol, s);
    t = CHOLMOD(mult_size_t)(nrow, 2, &ok);
    s = CHOLMOD(add_size_t)(s, t, &ok);
    if (!ok) {
        ERROR(CHOLMOD_TOO_LARGE, "problem too large");
        return (FALSE);
    }

    CHOLMOD(allocate_work)
    (nrow, s, 0, Common);
    if (Common->status < CHOLMOD_OK) {
        return (FALSE);
    }

    S = NULL;
    F = NULL;
    A1 = NULL;
    A2 = NULL;

    /* convert to another form when done, if requested */
    convert = !(Common->final_asis);

    /* ---------------------------------------------------------------------- */
    /* perform supernodal LL' or simplicial LDL' factorization */
    /* ---------------------------------------------------------------------- */

    if (L->is_super) {

#ifndef NSUPERNODAL

        /* ------------------------------------------------------------------ */
        /* supernodal factorization */
        /* ------------------------------------------------------------------ */

        if (L->ordering == CHOLMOD_NATURAL) {

            /* -------------------------------------------------------------- */
            /* natural ordering */
            /* -------------------------------------------------------------- */

            if (stype > 0) {
                /* S = tril (A'), F not needed */
                /* workspace: Iwork (nrow) */
                A1 = CHOLMOD(ptranspose)(A, 2, NULL, NULL, 0, Common);
                S = A1;
            }
            else if (stype < 0) {
                /* This is the fastest option for the natural ordering */
                /* S = A; F not needed */
                S = A;
            }
            else {
                /* F = A(:,f)' */
                /* workspace: Iwork (nrow) */
                /* workspace: Iwork (nrow if no fset; MAX (nrow,ncol) if fset)*/
                A1 = CHOLMOD(ptranspose)(A, 2, NULL, fset, fsize, Common);
                F = A1;
                /* S = A */
                S = A;
            }
        }
        else {

            /* -------------------------------------------------------------- */
            /* permute the input matrix before factorization */
            /* -------------------------------------------------------------- */

            if (stype > 0) {
                /* This is the fastest option for factoring a permuted matrix */
                /* S = tril (PAP'); F not needed */
                /* workspace: Iwork (2*nrow) */
                A1 = CHOLMOD(ptranspose)(A, 2, (int*)L->Perm, NULL, 0, Common);
                S = A1;
            }
            else if (stype < 0) {
                /* A2 = triu (PAP') */
                /* workspace: Iwork (2*nrow) */
                A2 = CHOLMOD(ptranspose)(A, 2, (int*)L->Perm, NULL, 0, Common);
                /* S = tril (A2'); F not needed */
                /* workspace: Iwork (nrow) */
                A1 = CHOLMOD(ptranspose)(A2, 2, NULL, NULL, 0, Common);
                S = A1;
                CHOLMOD(free_sparse)
                (&A2, Common);

                // Compute the first child and its distance
                ASSERT(A2 == NULL);
            }
            else {
                /* F = A(p,f)' */
                /* workspace: Iwork (nrow if no fset; MAX (nrow,ncol) if fset)*/
                A1 = CHOLMOD(ptranspose)(A, 2, (int*)L->Perm, fset, fsize, Common);
                F = A1;
                /* S = F' */
                /* workspace: Iwork (nrow) */
                A2 = CHOLMOD(ptranspose)(F, 2, NULL, NULL, 0, Common);
                S = A2;
            }
        }

        /* ------------------------------------------------------------------ */
        /* supernodal factorization */
        /* ------------------------------------------------------------------ */

        /* workspace: Flag (nrow), Head (nrow+1), Iwork (2*nrow+2*nsuper) */
        if (Common->status == CHOLMOD_OK) {
            cholmod_super_numeric_custom(S, F, beta, L, Common);
        }
        status = Common->status;
        ASSERT(IMPLIES(status >= CHOLMOD_OK, L->xtype != CHOLMOD_PATTERN));

        /* ------------------------------------------------------------------ */
        /* convert to final form, if requested */
        /* ------------------------------------------------------------------ */

        if (Common->status >= CHOLMOD_OK && convert) {
            /* workspace: none */
            ok = CHOLMOD(change_factor)(L->xtype, Common->final_ll,
                Common->final_super, Common->final_pack,
                Common->final_monotonic, L, Common);
            if (ok && Common->final_resymbol && !(L->is_super)) {
                /* workspace: Flag (nrow), Head (nrow+1),
                 *	if symmetric:   Iwork (2*nrow)
                 *	if unsymmetric: Iwork (2*nrow+ncol) */
                CHOLMOD(resymbol_noperm)
                (S, fset, fsize, Common->final_pack, L, Common);
            }
        }

#else

        /* ------------------------------------------------------------------ */
        /* CHOLMOD Supernodal module not installed */
        /* ------------------------------------------------------------------ */

        status = CHOLMOD_NOT_INSTALLED;
        ERROR(CHOLMOD_NOT_INSTALLED, "Supernodal module not installed");

#endif
    }
    else {

        /* ------------------------------------------------------------------ */
        /* simplicial LDL' factorization */
        /* ------------------------------------------------------------------ */

        /* Permute the input matrix A if necessary.  cholmod_rowfac requires
         * triu(A) in column form for the symmetric case, and A in column form
         * for the unsymmetric case (the matrix S).  The unsymmetric case
         * requires A in row form, or equivalently A' in column form (the
         * matrix F).
         */

        if (L->ordering == CHOLMOD_NATURAL) {

            /* -------------------------------------------------------------- */
            /* natural ordering */
            /* -------------------------------------------------------------- */

            if (stype > 0) {
                /* F is not needed, S = A */
                S = A;
            }
            else if (stype < 0) {
                /* F is not needed, S = A' */
                /* workspace: Iwork (nrow) */
                A2 = CHOLMOD(ptranspose)(A, 2, NULL, NULL, 0, Common);
                S = A2;
            }
            else {
                /* F = A (:,f)' */
                /* workspace: Iwork (nrow if no fset; MAX (nrow,ncol) if fset)*/
                A1 = CHOLMOD(ptranspose)(A, 2, NULL, fset, fsize, Common);
                F = A1;
                S = A;
            }
        }
        else {

            /* -------------------------------------------------------------- */
            /* permute the input matrix before factorization */
            /* -------------------------------------------------------------- */

            if (stype > 0) {
                /* F = tril (A (p,p)') */
                /* workspace: Iwork (2*nrow) */
                A1 = CHOLMOD(ptranspose)(A, 2, (int*)L->Perm, NULL, 0, Common);
                /* A2 = triu (F') */
                /* workspace: Iwork (nrow) */
                A2 = CHOLMOD(ptranspose)(A1, 2, NULL, NULL, 0, Common);
                /* the symmetric case does not need F, free it and set to NULL*/
                CHOLMOD(free_sparse)
                (&A1, Common);
            }
            else if (stype < 0) {
                /* A2 = triu (A (p,p)'), F not needed.  This is the fastest
                 * way to factorize a matrix using the simplicial routine
                 * (cholmod_rowfac). */
                /* workspace: Iwork (2*nrow) */
                A2 = CHOLMOD(ptranspose)(A, 2, (int*)L->Perm, NULL, 0, Common);
            }
            else {
                /* F = A (p,f)' */
                /* workspace: Iwork (nrow if no fset; MAX (nrow,ncol) if fset)*/
                A1 = CHOLMOD(ptranspose)(A, 2, (int*)L->Perm, fset, fsize, Common);
                F = A1;
                /* A2 = F' */
                /* workspace: Iwork (nrow) */
                A2 = CHOLMOD(ptranspose)(F, 2, NULL, NULL, 0, Common);
            }
            S = A2;
        }

        /* ------------------------------------------------------------------ */
        /* simplicial LDL' or LL' factorization */
        /* ------------------------------------------------------------------ */

        /* factorize beta*I+S (symmetric) or beta*I+F*F' (unsymmetric) */
        /* workspace: Flag (nrow), W (nrow), Iwork (2*nrow) */
        if (Common->status == CHOLMOD_OK) {
            grow2 = Common->grow2;
            L->is_ll = BOOLEAN(Common->final_ll);
            if (L->xtype == CHOLMOD_PATTERN && Common->final_pack) {
                /* allocate a factor with exactly the space required */
                Common->grow2 = 0;
            }
            CHOLMOD(rowfac)
            (S, F, beta, 0, nrow, L, Common);
            Common->grow2 = grow2;
        }
        status = Common->status;

        /* ------------------------------------------------------------------ */
        /* convert to final form, if requested */
        /* ------------------------------------------------------------------ */

        if (Common->status >= CHOLMOD_OK && convert) {
            /* workspace: none */
            CHOLMOD(change_factor)
            (L->xtype, L->is_ll, FALSE, Common->final_pack, Common->final_monotonic,
                L, Common);
        }
    }

    /* ---------------------------------------------------------------------- */
    /* free A1 and A2 if they exist */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(free_sparse)
    (&A1, Common);
    CHOLMOD(free_sparse)
    (&A2, Common);
    Common->status = MAX(Common->status, status);
    return (Common->status >= CHOLMOD_OK);
}

} // namespace PARTH
