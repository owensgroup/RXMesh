#pragma once
#ifdef USE_CUDSS

#include "rxmesh/matrix/direct_solver.h"

#include <cudss.h>

namespace rxmesh {

template <typename SpMatT, int DenseMatOrder = Eigen::ColMajor>
struct cuDSSCholeskySolver : public DirectSolver<SpMatT, DenseMatOrder>
{
    using IndexT = typename SpMatT::IndexT;
    using T      = typename SpMatT::Type;

    cuDSSCholeskySolver()
        : DirectSolver<SpMatT, DenseMatOrder>(), m_first_pre_solve(false)
    {
    }

    cuDSSCholeskySolver(SpMatT* mat, PermuteMethod perm = PermuteMethod::NSTDIS)
        : DirectSolver<SpMatT, DenseMatOrder>(mat, perm),
          m_A(mat->get_cudss_matrix()),
          m_first_pre_solve(true)
    {
        CUDSS_ERROR(cudssCreate(&m_cudss_handle));
        CUDSS_ERROR(cudssConfigCreate(&m_cudss_config));
        CUDSS_ERROR(cudssDataCreate(m_cudss_handle, &m_cudss_data));
    }

    virtual ~cuDSSCholeskySolver()
    {
        CUDSS_ERROR(cudssConfigDestroy(m_cudss_config));
        CUDSS_ERROR(cudssDataDestroy(m_cudss_handle, m_cudss_data));
        CUDSS_ERROR(cudssDestroy(m_cudss_handle));
    }


    virtual void pre_solve(RXMeshStatic& rx) override
    {
        RXMESH_ERROR(
            "cuDSSCholeskySolver::pre_solve() this pre_solve method for cuDSS "
            "Cholesky Solver is not implemented. Please call pre_solve(B,X)");
    }

    /**
     * @brief pre_solve should be called before calling the solve() method.
     * and it should be called every time the matrix is updated
     */
    virtual void pre_solve(RXMeshStatic&                  rx,
                           DenseMatrix<T, DenseMatOrder>& B_mat,
                           DenseMatrix<T, DenseMatOrder>& X_mat)
    {
        if (m_first_pre_solve) {
            permute(rx, B_mat, X_mat);
            analyze_pattern();
            factorize();
            m_first_pre_solve = false;
        } else {
            factorize();
        }
    }

    /**
     * @brief permute the matrix to reduce the non-zero fill-in
     */
    virtual void permute(RXMeshStatic&                  rx,
                         DenseMatrix<T, DenseMatOrder>& B_mat,
                         DenseMatrix<T, DenseMatOrder>& X_mat)
    {
        if (m_first_pre_solve) {
            m_B = B_mat.get_cudss_matrix();
            m_X = X_mat.get_cudss_matrix();
        }
        // set reorder type
        cudssAlgType_t reorder_alg;
        switch (this->m_perm) {
            case PermuteMethod::SYMAMD: {
                // an approximate minimum degree (AMD) reordering.
                reorder_alg = CUDSS_ALG_3;
                break;
            }
            case PermuteMethod::NSTDIS: {
                // a customized nested dissection algorithm based on METIS.
                reorder_alg = CUDSS_ALG_DEFAULT;
                break;
            }
            case PermuteMethod::SYMRCM: {
                // a custom combination of block triangular reordering and
                // COLAMD
                reorder_alg = CUDSS_ALG_1;
                break;
            }
            case PermuteMethod::GPUND: {
                reorder_alg = CUDSS_ALG_DEFAULT;
                DirectSolver<SpMatT, DenseMatOrder>::permute(rx);
                CUDSS_ERROR(
                    cudssDataSet(m_cudss_handle,
                                 m_cudss_data,
                                 CUDSS_DATA_USER_PERM,
                                 this->m_d_permute,
                                 size_t(m_mat->rows() * sizeof(IndexT))));
                break;
            }
            case PermuteMethod::GPUMGND:
            default:
                reorder_alg = CUDSS_ALG_DEFAULT;
        }

        CUDSS_ERROR(cudssConfigSet(m_cudss_config,
                                   CUDSS_CONFIG_REORDERING_ALG,
                                   &reorder_alg,
                                   sizeof(cudssAlgType_t)));


        CUDSS_ERROR(cudssExecute(m_cudss_handle,
                                 CUDSS_PHASE_REORDERING,
                                 m_cudss_config,
                                 m_cudss_data,
                                 m_A,
                                 m_X,
                                 m_B));
    }

    /**
     * @brief symbolic factorization
     */
    virtual void analyze_pattern()
    {
        // Symbolic factorization
        CUDSS_ERROR(cudssExecute(m_cudss_handle,
                                 CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
                                 m_cudss_config,
                                 m_cudss_data,
                                 m_A,
                                 m_X,
                                 m_B));
    }

    /**
     * @brief numerical (re-)factorization
     */
    virtual void factorize()
    {
        if (m_first_pre_solve) {
            // Numerical factorization
            CUDSS_ERROR(cudssExecute(m_cudss_handle,
                                     CUDSS_PHASE_FACTORIZATION,
                                     m_cudss_config,
                                     m_cudss_data,
                                     m_A,
                                     m_X,
                                     m_B));
        } else {
            // Re-factorize
            CUDSS_ERROR(cudssExecute(m_cudss_handle,
                                     CUDSS_PHASE_REFACTORIZATION,
                                     m_cudss_config,
                                     m_cudss_data,
                                     m_A,
                                     m_X,
                                     m_B));
        }


        size_t sizeInBytes = sizeof(int);
        size_t sizeWritten;
        int    singularity = 0;
        CUDSS_ERROR(cudssDataGet(m_cudss_handle,
                                 m_cudss_data,
                                 CUDSS_DATA_INFO,
                                 &singularity,
                                 sizeInBytes,
                                 &sizeWritten));

        if (0 < singularity) {
            RXMESH_WARN(
                "cuDSSCholeskySolver::factorize() The matrix is singular "
                "at row {}",
                singularity - 1);
        }
    }


    /**
     * @brief solve the system (should be called only after factorization, i.e.,
     * pre_solve())
     */
    virtual void solve(DenseMatrix<T, DenseMatOrder>& B_mat,
                       DenseMatrix<T, DenseMatOrder>& X_mat,
                       cudaStream_t                   stream = NULL) override
    {
        CUDSS_ERROR(cudssSetStream(m_cudss_handle, stream));

        // Solving the system
        CUDSS_ERROR(cudssExecute(m_cudss_handle,
                                 CUDSS_PHASE_SOLVE,
                                 m_cudss_config,
                                 m_cudss_data,
                                 m_A,
                                 m_X,
                                 m_B));
    }


    virtual std::string name() override
    {
        return std::string("cuDSSCholesky");
    }

   protected:
    bool m_first_pre_solve;

    cudssHandle_t m_cudss_handle;
    cudssConfig_t m_cudss_config;
    cudssData_t   m_cudss_data;

    cudssMatrix_t m_A;
    cudssMatrix_t m_B;
    cudssMatrix_t m_X;
};

}  // namespace rxmesh
#endif