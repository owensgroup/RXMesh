#pragma once

#include "rxmesh/matrix/direct_solver.h"

#include "cusolverSp_LOWLEVEL_PREVIEW.h"

namespace rxmesh {

template <typename SpMatT, int DenseMatOrder = Eigen::ColMajor>
struct CholeskySolver : public DirectSolver<SpMatT, DenseMatOrder>
{
    using T = typename SpMatT::Type;

    CholeskySolver(SpMatT* mat, PermuteMethod perm = PermuteMethod::NSTDIS)
        : DirectSolver<SpMatT, DenseMatOrder>(mat, perm),
          m_internalDataInBytes(0),
          m_workspaceInBytes(0),
          m_solver_buffer(nullptr),
          m_first_pre_solve(true)
    {
        CUSOLVER_ERROR(cusolverSpCreateCsrcholInfo(&m_chol_info));
    }

    virtual ~CholeskySolver()
    {
        GPU_FREE(m_solver_buffer);
        CUSOLVER_ERROR(cusolverSpDestroyCsrcholInfo(m_chol_info));
    }

    /**
     * @brief pre_solve should be called before calling the solve() method.
     * and it should be called every time the matrix is updated
     */
    virtual void pre_solve(RXMeshStatic& rx) override
    {
        if (m_first_pre_solve) {
            m_first_pre_solve = false;
            this->permute_alloc();
            this->permute(rx);
            this->premute_value_ptr();
            analyze_pattern();
            post_analyze_alloc();
            factorize();
        } else {
            this->premute_value_ptr();
            factorize();
        }
    }


    virtual void solve(DenseMatrix<Type, DenseMatOrder>& B_mat,
                       DenseMatrix<Type, DenseMatOrder>& X_mat,
                       cudaStream_t                      stream = NULL) override
    {
        if (m_first_pre_solve) {
            RXMESH_ERROR(
                "CholeskySolver::solver pre_solve() method should be called "
                "before calling the solve() method. Returning without solving "
                "anything.");
            return;
        }
        CUSOLVER_ERROR(cusolverSpSetStream(m_cusolver_sphandle, stream));


        // TODO this could be handled more cleanly using reshape operator on the
        // user side

        //  the case where we solve for multiple rhs
        if (m_mat->cols() == X_mat.rows() && m_mat->rows() == B_mat.rows() &&
            X_mat.cols() == B_mat.cols()) {
            for (int i = 0; i < B_mat.cols(); ++i) {
                solve(B_mat.col_data(i, DEVICE), X_mat.col_data(i, DEVICE));
            }
        } else if (m_mat->cols() == X_mat.rows() * X_mat.cols() &&
                   m_mat->rows() == B_mat.rows() * B_mat.cols()) {
            // the case where we flatten X and B and do one solve
            solve(B_mat.col_data(0, DEVICE), X_mat.col_data(0, DEVICE));
        } else {
            RXMESH_ERROR(
                "CholeskySolver::solve() The sparse matrix size ({}, {}) does "
                "not match with the rhs size ({}, {}) and the unknown size "
                "({}, {})",
                m_mat->rows(),
                m_mat->cols(),
                B_mat.rows(),
                B_mat.cols(),
                X_mat.rows(),
                X_mat.cols());
        }
    }


    /**
     * @brief solve using cuSolver High-Level API that permute, factorize, and
     * solve the matrix in one API call. This API is slows specially when
     * solving multiple rhs.
     */
    void solve_hl_api(const DenseMatrix<T, DenseMatOrder>& B_mat,
                      DenseMatrix<T, DenseMatOrder>&       X_mat,
                      cudaStream_t                         stream = NULL)
    {
        CUSOLVER_ERROR(cusolverSpSetStream(m_cusolver_sphandle, stream));

        // the case where we solve for multiple rhs
        if (m_mat->cols() == X_mat.rows() && m_mat->rows() == B_mat.rows() &&
            X_mat.cols() == B_mat.cols()) {
            for (int i = 0; i < B_mat.cols(); ++i) {
                cusolver_cholesky(B_mat.col_data(i, DEVICE),
                                  X_mat.col_data(i, DEVICE));
            }
        } else if (m_mat->cols() == X_mat.rows() * X_mat.cols() &&
                   m_mat->rows() == B_mat.rows() * B_mat.cols()) {
            // the case where we flatten X and B and do one solve
            cusolver_cholesky(B_mat.col_data(0, DEVICE),
                              X_mat.col_data(0, DEVICE));
        } else {
            RXMESH_ERROR(
                "CholeskySolver::solve_hl_api() The sparse matrix size ({}, "
                "{}) does not match with the rhs size ({}, {}) and the unknown "
                "size ({}, {})",
                m_mat->rows(),
                m_mat->cols(),
                B_mat.rows(),
                B_mat.cols(),
                X_mat.rows(),
                X_mat.cols());
        }
    }

    /**
     * @brief The lower level API of matrix analysis. Generating a member value
     * of type csrcholInfo_t for cuSolver.
     */
    virtual void analyze_pattern()
    {

        CUSOLVER_ERROR(cusolverSpXcsrcholAnalysis(m_cusolver_sphandle,
                                                  m_mat->rows(),
                                                  m_mat->non_zeros(),
                                                  m_descr,
                                                  this->m_d_solver_row_ptr,
                                                  this->m_d_solver_col_idx,
                                                  m_chol_info));
    }


    /**
     * @brief The lower level API of matrix factorization buffer calculation and
     * allocation. The buffer is a member variable.
     */
    virtual void post_analyze_alloc()
    {
        m_internalDataInBytes = 0;
        m_workspaceInBytes    = 0;

        GPU_FREE(m_solver_buffer);


        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholBufferInfo(m_cusolver_sphandle,
                                                        m_mat->rows(),
                                                        m_mat->non_zeros(),
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        &m_internalDataInBytes,
                                                        &m_workspaceInBytes));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            CUSOLVER_ERROR(cusolverSpCcsrcholBufferInfo(m_cusolver_sphandle,
                                                        m_mat->rows(),
                                                        m_mat->non_zeros(),
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        &m_internalDataInBytes,
                                                        &m_workspaceInBytes));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholBufferInfo(m_cusolver_sphandle,
                                                        m_mat->rows(),
                                                        m_mat->non_zeros(),
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        &m_internalDataInBytes,
                                                        &m_workspaceInBytes));
        }

        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            CUSOLVER_ERROR(cusolverSpZcsrcholBufferInfo(m_cusolver_sphandle,
                                                        m_mat->rows(),
                                                        m_mat->non_zeros(),
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        &m_internalDataInBytes,
                                                        &m_workspaceInBytes));
        }


        RXMESH_TRACE(
            "CholeskySolver::post_analyze_alloc() internalDataInBytes= {}, "
            "workspaceInBytes= {}",
            m_internalDataInBytes,
            m_workspaceInBytes);
        CUDA_ERROR(cudaMalloc((void**)&m_solver_buffer, m_workspaceInBytes));
    }

    /**
     * @brief The lower level api of matrix factorization and save the
     * factorization result in to the buffer.
     */
    virtual void factorize()
    {
        m_first_pre_solve = false;

        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholFactor(m_cusolver_sphandle,
                                                    m_mat->rows(),
                                                    m_mat->non_zeros(),
                                                    m_descr,
                                                    m_d_solver_val,
                                                    m_d_solver_row_ptr,
                                                    m_d_solver_col_idx,
                                                    m_chol_info,
                                                    m_solver_buffer));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            CUSOLVER_ERROR(cusolverSpCcsrcholFactor(m_cusolver_sphandle,
                                                    m_mat->rows(),
                                                    m_mat->non_zeros(),
                                                    m_descr,
                                                    m_d_solver_val,
                                                    m_d_solver_row_ptr,
                                                    m_d_solver_col_idx,
                                                    m_chol_info,
                                                    m_solver_buffer));
        }
        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholFactor(m_cusolver_sphandle,
                                                    m_mat->rows(),
                                                    m_mat->non_zeros(),
                                                    m_descr,
                                                    m_d_solver_val,
                                                    m_d_solver_row_ptr,
                                                    m_d_solver_col_idx,
                                                    m_chol_info,
                                                    m_solver_buffer));
        }
        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            CUSOLVER_ERROR(cusolverSpZcsrcholFactor(m_cusolver_sphandle,
                                                    m_mat->rows(),
                                                    m_mat->non_zeros(),
                                                    m_descr,
                                                    m_d_solver_val,
                                                    m_d_solver_row_ptr,
                                                    m_d_solver_col_idx,
                                                    m_chol_info,
                                                    m_solver_buffer));
        }


        double tol = 1.0e-8;
        int    singularity;


        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholZeroPivot(
                m_cusolver_sphandle, m_chol_info, tol, &singularity));
        }
        if constexpr (std::is_same_v<T, cuComplex>) {
            CUSOLVER_ERROR(cusolverSpCcsrcholZeroPivot(
                m_cusolver_sphandle, m_chol_info, tol, &singularity));
        }
        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholZeroPivot(
                m_cusolver_sphandle, m_chol_info, tol, &singularity));
        }
        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            CUSOLVER_ERROR(cusolverSpZcsrcholZeroPivot(
                m_cusolver_sphandle, m_chol_info, tol, &singularity));
        }


        if (0 <= singularity) {
            RXMESH_WARN(
                "CholeskySolver::factorize() The matrix is singular at row {} "
                "under tol ({})",
                singularity,
                tol);
        }
    }


    /**
     * @brief The lower level API of solving the linear system after using
     * factorization. The format follows Ax=b to solve x, where A is the sparse
     * matrix, x and b are device array. As long as A doesn't change. This
     * function could be called for many different b and x.
     * @param d_b: right hand side
     * @param d_x: output solution
     */
    virtual void solve(T* d_b, T* d_x)
    {
        T* d_solver_b;
        T* d_solver_x;

        if (this->m_use_permute) {
            // permute b and x
            d_solver_b = m_d_solver_b;
            d_solver_x = m_d_solver_x;
            permute_gather(m_d_permute, d_b, d_solver_b, m_mat->cols());
            permute_gather(m_d_permute, d_x, d_solver_x, m_mat->cols());
        } else {
            d_solver_b = d_b;
            d_solver_x = d_x;
        }


        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholSolve(m_cusolver_sphandle,
                                                   m_mat->rows(),
                                                   d_solver_b,
                                                   d_solver_x,
                                                   m_chol_info,
                                                   m_solver_buffer));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            CUSOLVER_ERROR(cusolverSpCcsrcholSolve(m_cusolver_sphandle,
                                                   m_mat->rows(),
                                                   d_solver_b,
                                                   d_solver_x,
                                                   m_chol_info,
                                                   m_solver_buffer));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholSolve(m_cusolver_sphandle,
                                                   m_mat->rows(),
                                                   d_solver_b,
                                                   d_solver_x,
                                                   m_chol_info,
                                                   m_solver_buffer));
        }
        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            CUSOLVER_ERROR(cusolverSpZcsrcholSolve(m_cusolver_sphandle,
                                                   m_mat->rows(),
                                                   d_solver_b,
                                                   d_solver_x,
                                                   m_chol_info,
                                                   m_solver_buffer));
        }


        if (this->m_use_permute) {
            permute_scatter(m_d_permute, d_solver_x, d_x, m_mat->rows());
            permute_scatter(m_d_permute, d_solver_b, d_b, m_mat->rows());
        }
    }

    virtual std::string name() override
    {
        return std::string("Cholesky");
    }

   protected:
    /**
     * @brief wrapper for cuSolver API for solving linear systems using cuSolver
     * High-level API
     */
    void cusolver_cholesky(const T* d_b, T* d_x)
    {

        double tol = 1.e-12;

        // -1 if A is invertible under tol.
        int singularity = 0;


        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrlsvchol(m_cusolver_sphandle,
                                                 m_mat->rows(),
                                                 m_mat->non_zeros(),
                                                 m_descr,
                                                 m_mat->val_ptr(DEVICE),
                                                 m_mat->row_ptr(DEVICE),
                                                 m_mat->col_idx(DEVICE),
                                                 d_b,
                                                 tol,
                                                 permute_to_int(),
                                                 d_x,
                                                 &singularity));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            CUSOLVER_ERROR(cusolverSpCcsrlsvchol(m_cusolver_sphandle,
                                                 m_mat->rows(),
                                                 m_mat->non_zeros(),
                                                 m_descr,
                                                 m_mat->val_ptr(DEVICE),
                                                 m_mat->row_ptr(DEVICE),
                                                 m_mat->col_idx(DEVICE),
                                                 d_b,
                                                 tol,
                                                 permute_to_int(),
                                                 d_x,
                                                 &singularity));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrlsvchol(m_cusolver_sphandle,
                                                 m_mat->rows(),
                                                 m_mat->non_zeros(),
                                                 m_descr,
                                                 m_mat->val_ptr(DEVICE),
                                                 m_mat->row_ptr(DEVICE),
                                                 m_mat->col_idx(DEVICE),
                                                 d_b,
                                                 tol,
                                                 permute_to_int(),
                                                 d_x,
                                                 &singularity));
        }
        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            CUSOLVER_ERROR(cusolverSpZcsrlsvchol(m_cusolver_sphandle,
                                                 m_mat->rows(),
                                                 m_mat->non_zeros(),
                                                 m_descr,
                                                 m_mat->val_ptr(DEVICE),
                                                 m_mat->row_ptr(DEVICE),
                                                 m_mat->col_idx(DEVICE),
                                                 d_b,
                                                 tol,
                                                 permute_to_int(),
                                                 d_x,
                                                 &singularity));
        }


        if (0 <= singularity) {
            RXMESH_WARN(
                "CholeskySolver::cusolver_cholesky() The matrix is "
                "singular at row {} under tol ({})",
                singularity,
                tol);
        }
    }

    csrcholInfo_t m_chol_info;

    size_t m_internalDataInBytes;
    size_t m_workspaceInBytes;

    void* m_solver_buffer;

    bool m_first_pre_solve;
};

}  // namespace rxmesh