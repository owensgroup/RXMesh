#include "rxmesh/matrix/direct_solver.h"

namespace rxmesh {

template <typename SpMatT>
struct LUSolver : public DirectSolver<SpMatT>
{
    using T = typename SpMatT::Type;

    LUSolver(SpMatT* mat, PermuteMethod perm = PermuteMethod::NSTDIS)
        : DirectSolver<SpMatT>(mat, perm)
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
    virtual void solve(const DenseMatrix<Type>& B_mat,
                       DenseMatrix<Type>&       X_mat,
                       cudaStream_t             stream = NULL) override
    {
        CUSOLVER_ERROR(cusolverSpSetStream(m_cusolver_sphandle, stream));
        for (int i = 0; i < B_mat.cols(); ++i) {
            cusolver_qr(m_cusolver_sphandle,
                        B_mat.col_data(i, HOST),
                        X_mat.col_data(i, HOST),
                        stream);
        }
    }


   protected:
    /**
     * @brief wrapper for cuSolver API for solving linear systems using cuSolver
     * High-level API
     */
    void cusolver_lu(const T* d_b, T* d_x, cudaStream_t stream)
    {
        CUSOLVER_ERROR(cusolverSpSetStream(m_cusolver_sphandle, stream));

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
                                                   d_b,
                                                   tol,
                                                   permute_to_int(),
                                                   d_x,
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
                                                   d_b,
                                                   tol,
                                                   permute_to_int(),
                                                   d_x,
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
                                                   d_b,
                                                   tol,
                                                   permute_to_int(),
                                                   d_x,
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
                                                   d_b,
                                                   tol,
                                                   permute_to_int(),
                                                   d_x,
                                                   &singularity));
        }


        if (0 <= singularity) {
            RXMESH_WARN(
                "QRkySolver::cusolver_qr() The matrix is "
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