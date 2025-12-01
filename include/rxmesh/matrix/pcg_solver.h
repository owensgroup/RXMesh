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
        this->S.reset(0.0, rxmesh::DEVICE, stream);
        this->P.reset(0.0, rxmesh::DEVICE, stream);
        this->R.reset(0.0, rxmesh::DEVICE, stream);


        // init S
        // S = Ax
        this->A->multiply(X, this->S, false, false, 1, 0, stream);

        // init R
        // R = B - S
        init_R(B, this->S, this->R, stream);

        // init P
        // P = inv(M) * R
        precond(this->R, this->P, stream);

        this->delta_new = this->R.dot(this->P, false, stream);
    }

    virtual void solve(DenseMatT&   B,
                       DenseMatT&   X,
                       cudaStream_t stream = NULL) override
    {

        if (this->A->cols() != X.rows() || this->A->rows() != B.rows() ||
            X.cols() != B.cols()) {
            RXMESH_ERROR(
                "CGSolver::solver mismatch in the input/output size. A ({}, "
                "{}), X ({}, {}), B ({}, {})",
                this->A->rows(),
                this->A->cols(),
                X.rows(),
                X.cols(),
                B.rows(),
                B.cols());
            return;
        }

        this->m_start_residual = this->delta_new;

        this->m_iter_taken = 0;

        while (this->m_iter_taken < this->m_max_iter) {
            // s = Ap
            this->A->multiply(this->P, this->S, false, false, 1, 0, stream);

            // alpha = this->delta_new / <S,P>
            this->alpha = this->S.dot(this->P, false, stream);
            this->alpha = this->delta_new / this->alpha;

            // X =  alpha*P + X
            this->axpy(X, this->P, this->alpha, T(1.), stream);

            // reset residual
            if (this->m_iter_taken > 0 &&
                this->m_iter_taken % this->m_reset_residual_freq == 0) {
                // s= Ax
                this->A->multiply(X, this->S, false, false, 1, 0, stream);

                // r = b-s
                this->subtract(this->R, B, this->S, stream);
            } else {
                // r = - alpha*s + r
                this->axpy(this->R, this->S, -this->alpha, T(1.), stream);
            }

            // S = inv(M) *R
            precond(this->R, this->S, stream);

            // delta_old = this->delta_new
            CUDA_ERROR(cudaStreamSynchronize(stream));
            this->delta_old = this->delta_new;

            // this->delta_new = <r,s>
            this->delta_new = this->R.dot(this->S, stream);


            // exit if error is getting too low across three coordinates
            if (this->is_converged(this->m_start_residual, this->delta_new)) {
                this->m_final_residual = this->delta_new;
                return;
            }

            // beta = this->delta_new/delta_old
            this->beta = this->delta_new / this->delta_old;

            // p = beta*p + s
            this->axpy(this->P, this->S, T(1.), this->beta, stream);

            this->m_iter_taken++;
        }
        RXMESH_WARN(
            "PCGSolver::solve() did not converge after {} "
            "iterations. Residual "
            "= {}",
            this->m_iter_taken,
            delta_new);

        this->m_final_residual = this->delta_new;
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

        SparseMatrix<T> Amat = *(this->A);

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
