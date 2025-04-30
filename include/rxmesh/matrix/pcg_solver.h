#pragma once
#include "rxmesh/matrix/iterative_solver.h"

#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"

namespace rxmesh {

/**
 * @brief Jacobi-preconditioned CG
 */
template <typename T, int DenseMatOrder = Eigen::ColMajor>
struct PCGSolver : public CGSolver<T, DenseMatOrder>
{
    using DenseMatT = DenseMatrix<T, DenseMatOrder>;

    PCGSolver(SparseMatrix<T>& sys,
              int              unknown_dim,  // num rhs vectors
              int              max_iter,
              T                abs_tol = 1e-6,
              T                rel_tol = 0.0,
              int reset_residual_freq  = std::numeric_limits<int>::max())
        : CGSolver<T, DenseMatOrder>(sys,
                                     unknown_dim,
                                     max_iter,
                                     abs_tol,
                                     rel_tol,
                                     reset_residual_freq)
    {
    }

    virtual void pre_solve(const DenseMatT& B,
                           DenseMatT&       X,
                           cudaStream_t     stream = NULL) override
    {
        S.reset(0.0, rxmesh::DEVICE, stream);
        P.reset(0.0, rxmesh::DEVICE, stream);
        R.reset(0.0, rxmesh::DEVICE, stream);


        // init S
        // S = Ax
        A->multiply(X, S, false, false, 1, 0, stream);

        // init R
        // R = B - S
        init_R(B, S, R, stream);

        // init P
        // P = inv(M) * R
        precond(R, P, stream);

        delta_new = R.dot(P, false, stream);
    }

    virtual void solve(const DenseMatT& B,
                       DenseMatT&       X,
                       cudaStream_t     stream = NULL) override
    {

        if (A->cols() != X.rows() || A->rows() != B.rows() ||
            X.cols() != B.cols()) {
            RXMESH_ERROR(
                "CGSolver::solver mismatch in the input/output size. A ({}, "
                "{}), X ({}, {}), B ({}, {})",
                A->rows(),
                A->cols(),
                X.rows(),
                X.cols(),
                B.rows(),
                B.cols());
            return;
        }

        m_start_residual = delta_new;

        m_iter_taken = 0;

        while (m_iter_taken < m_max_iter) {
            // s = Ap
            A->multiply(P, S, false, false, 1, 0, stream);

            // alpha = delta_new / <S,P>
            alpha = S.dot(P, false, stream);
            alpha = delta_new / alpha;

            // X =  alpha*P + X
            axpy(X, P, alpha, T(1.), stream);

            // reset residual
            if (m_iter_taken > 0 && m_iter_taken % m_reset_residual_freq == 0) {
                // s= Ax
                A->multiply(X, S, false, false, 1, 0, stream);

                // r = b-s
                subtract(R, B, S, stream);
            } else {
                // r = - alpha*s + r
                axpy(R, S, -alpha, T(1.), stream);
            }

            // S = inv(M) *R
            precond(R, S, stream);

            // delta_old = delta_new
            CUDA_ERROR(cudaStreamSynchronize(stream));
            delta_old = delta_new;

            // delta_new = <r,s>
            delta_new = R.dot(S, stream);


            // exit if error is getting too low across three coordinates
            if (is_converged(m_start_residual, delta_new)) {
                m_final_residual = delta_new;
                return;
            }

            // beta = delta_new/delta_old
            beta = delta_new / delta_old;

            // p = beta*p + s
            axpy(P, S, T(1.), beta, stream);

            m_iter_taken++;
        }
        m_final_residual = delta_new;
    }

    virtual std::string name() override
    {
        return std::string("PCG");
    }

    virtual ~PCGSolver()
    {
    }

    /**
     * @brief implement out = inv(M) * in
     * where in and out are dense matrices and M is the preconditioner
     * Here the preconditioner is Jacobi preconditioner. Thus, inv(M) is simply
     * a diagonal matrix where M(i,i) is 1/A(i,i)
     */
    virtual void precond(const DenseMatT& in,
                         DenseMatT&       out,
                         cudaStream_t     stream = NULL)
    {
        const int rows = in.rows();
        const int cols = in.cols();

        SparseMatrix<T> Amat = *A;

        const int blockThreads = 512;

        const int blocks = DIVIDE_UP(rows, blockThreads);

        for_each_item<<<blocks, blockThreads, 0, stream>>>(
            rows,
            [in, out, Amat, cols] __device__(int i) mutable {
                const T diag = T(1) / Amat(i, i);

                for (int j = 0; j < cols; ++j) {
                    out(i, j) = diag * in(i, j);
                }
            }

        );
    }

    /**
     * @brief initialize the residual
     */
    void init_R(const DenseMatT& B,
                const DenseMatT& S,
                DenseMatT&       R,
                cudaStream_t     stream = NULL)
    {
        const int rows = B.rows();
        const int cols = B.cols();

        const int blockThreads = 512;

        const int blocks = DIVIDE_UP(rows, blockThreads);

        for_each_item<<<blocks, blockThreads, 0, stream>>>(
            rows,
            [B, S, R, cols] __device__(int i) mutable {
                for (int j = 0; j < cols; ++j) {
                    R(i, j) = B(i, j) - S(i, j);
                }
            }

        );
    };
};

}  // namespace rxmesh
