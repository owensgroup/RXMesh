#pragma once

#include "rxmesh/matrix/direct_solver.h"

namespace rxmesh {

template <typename SpMatT, int DenseMatOrder = Eigen::ColMajor>
struct LUSolver : public DirectSolver<SpMatT, DenseMatOrder>
{
    using T = typename SpMatT::Type;

    LUSolver(SpMatT* mat, PermuteMethod perm = PermuteMethod::NSTDIS)
        : DirectSolver<SpMatT, DenseMatOrder>(mat, perm)
    {
    }

    virtual ~LUSolver()
    {
    }

    virtual void pre_solve(RXMeshStatic& rx) override
    {
    }


    /**
     * @brief solve using cuSolver High-Level API that permute, factorize, and
     * solve the matrix in one API call. This API is slows specially when
     * solving multiple rhs. Additionally, this solves the system on the CPU.
     * Thus, the caller must make sure that the matrix, X, and B are updated on
     * the host. After solving, moving B to the device is the caller's
     * responsiblity
     */
    virtual void solve(DenseMatrix<Type, DenseMatOrder>& B_mat,
                       DenseMatrix<Type, DenseMatOrder>& X_mat,
                       cudaStream_t                      stream = NULL) override
    {
        CUSOLVER_ERROR(cusolverSpSetStream(m_cusolver_sphandle, stream));

        // TODO this could be handled more cleanly using reshape operator on the
        // user side
        if (m_mat->cols() == X_mat.rows() && m_mat->rows() == B_mat.rows() &&
            X_mat.cols() == B_mat.cols()) {
            for (int i = 0; i < B_mat.cols(); ++i) {
                cusolver_lu(B_mat.col_data(i, HOST), X_mat.col_data(i, HOST));
            }
        } else if (m_mat->cols() == X_mat.rows() * X_mat.cols() &&
                   m_mat->rows() == B_mat.rows() * B_mat.cols()) {
            // the case where we flatten X and B and do one solve
            cusolver_lu(B_mat.col_data(0, HOST), X_mat.col_data(0, HOST));
        } else {
            RXMESH_ERROR(
                "LUSolver::solve() The sparse matrix size ({}, {}) does not "
                "match with the rhs size ({}, {}) and the unknown size ({}, "
                "{})",
                m_mat->rows(),
                m_mat->cols(),
                B_mat.rows(),
                B_mat.cols(),
                X_mat.rows(),
                X_mat.cols());
        }
    }

    virtual std::string name() override
    {
        return std::string("LU");
    }

   protected:
    /**
     * @brief wrapper for cuSolver API for solving linear systems using cuSolver
     * High-level API
     */
    void cusolver_lu(T* h_b, T* h_x)
    {
        double tol = 1.e-12;

        // -1 if A is invertible under tol.
        int singularity = 0;


        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrlsvluHost(m_cusolver_sphandle,
                                                   m_mat->rows(),
                                                   m_mat->non_zeros(),
                                                   m_descr,
                                                   m_mat->val_ptr(HOST),
                                                   m_mat->row_ptr(HOST),
                                                   m_mat->col_idx(HOST),
                                                   h_b,
                                                   tol,
                                                   permute_to_int(),
                                                   h_x,
                                                   &singularity));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            CUSOLVER_ERROR(cusolverSpCcsrlsvluHost(m_cusolver_sphandle,
                                                   m_mat->rows(),
                                                   m_mat->non_zeros(),
                                                   m_descr,
                                                   m_mat->val_ptr(HOST),
                                                   m_mat->row_ptr(HOST),
                                                   m_mat->col_idx(HOST),
                                                   h_b,
                                                   tol,
                                                   permute_to_int(),
                                                   h_x,
                                                   &singularity));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrlsvluHost(m_cusolver_sphandle,
                                                   m_mat->rows(),
                                                   m_mat->non_zeros(),
                                                   m_descr,
                                                   m_mat->val_ptr(HOST),
                                                   m_mat->row_ptr(HOST),
                                                   m_mat->col_idx(HOST),
                                                   h_b,
                                                   tol,
                                                   permute_to_int(),
                                                   h_x,
                                                   &singularity));
        }
        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            CUSOLVER_ERROR(cusolverSpZcsrlsvluHost(m_cusolver_sphandle,
                                                   m_mat->rows(),
                                                   m_mat->non_zeros(),
                                                   m_descr,
                                                   m_mat->val_ptr(HOST),
                                                   m_mat->row_ptr(HOST),
                                                   m_mat->col_idx(HOST),
                                                   h_b,
                                                   tol,
                                                   permute_to_int(),
                                                   h_x,
                                                   &singularity));
        }


        if (0 <= singularity) {
            RXMESH_WARN(
                "LUSolver::cusolver_lu() The matrix is "
                "singular at row {} under tol ({})",
                singularity,
                tol);
        }
    }

    csrqrInfo_t m_qr_info;

    size_t m_internalDataInBytes;
    size_t m_workspaceInBytes;

    void* m_solver_buffer;
};

}  // namespace rxmesh