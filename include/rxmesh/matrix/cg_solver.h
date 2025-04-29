#pragma once
#include "rxmesh/matrix/iterative_solver.h"

#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"

namespace rxmesh {

/**
 * @brief (Un-preconditioned) CG
 */
template <typename T, int DenseMatOrder = Eigen::ColMajor>
struct CGSolver : public IterativeSolver<T, DenseMatrix<T, DenseMatOrder>>
{
    using DenseMatT = DenseMatrix<T, DenseMatOrder>;

    CGSolver(SparseMatrix<T>& sys,
             int              unknown_dim,  // num rhs vectors
             int              max_iter,
             T                abs_tol = 1e-6,
             T                rel_tol = 0.0,
             int reset_residual_freq  = std::numeric_limits<int>::max())
        : IterativeSolver<T, DenseMatT>(max_iter, abs_tol, rel_tol),
          A(&sys),
          alpha(0),
          beta(0),
          delta_new(0),
          delta_old(0),
          m_reset_residual_freq(reset_residual_freq),
          S(DenseMatT(sys.rows(), unknown_dim, DEVICE)),
          P(DenseMatT(sys.rows(), unknown_dim, DEVICE)),
          R(DenseMatT(sys.rows(), unknown_dim, DEVICE))
    {
        A->alloc_multiply_buffer(S, P);
    }

    virtual void pre_solve(const DenseMatT& B,
                           DenseMatT&       X,
                           cudaStream_t     stream = NULL) override
    {

        S.reset(0.0, rxmesh::DEVICE, stream);
        P.reset(0.0, rxmesh::DEVICE, stream);
        R.reset(0.0, rxmesh::DEVICE, stream);


        // init S
        mat_vec(X, S, stream);

        init_PR(B, S, R, P, stream);

        delta_new = R.norm2(stream);
        delta_new *= delta_new;
    }

    virtual void solve(const DenseMatT& B,
                       DenseMatT&       X,
                       cudaStream_t     stream = NULL) override
    {
        m_start_residual = delta_new;

        m_iter_taken = 0;

        while (m_iter_taken < m_max_iter) {
            // s = Ap
            mat_vec(P, S, stream);

            // alpha = delta_new / <S,P>
            alpha = S.dot(P, false, stream);
            alpha = delta_new / alpha;

            // X =  alpha*P + X
            axpy(X, P, alpha, T(1.), stream);

            // reset residual
            if (m_iter_taken > 0 && m_iter_taken % m_reset_residual_freq == 0) {
                // s= Ax
                mat_vec(X, S, stream);

                // r = b-s
                subtract(R, B, S, stream);
            } else {
                // r = - alpha*s + r
                axpy(R, S, -alpha, T(1.), stream);
            }

            // delta_old = delta_new
            CUDA_ERROR(cudaStreamSynchronize(stream));
            delta_old = delta_new;

            // delta_new = <r,r>
            delta_new = R.norm2(stream);
            delta_new *= delta_new;

            // exit if error is getting too low across three coordinates
            if (is_converged(m_start_residual, delta_new)) {
                m_final_residual = delta_new;
                return;
            }

            // beta = delta_new/delta_old
            beta = delta_new / delta_old;

            // p = beta*p + r
            axpy(P, R, T(1.), beta, stream);

            m_iter_taken++;
        }
        m_final_residual = delta_new;
    }

    virtual std::string name() override
    {
        return std::string("CG");
    }

    virtual ~CGSolver()
    {
    }


    /**
     * @brief implement Y = alpha*X + beta*Y
     */
    void axpy(DenseMatT&       y,
              const DenseMatT& x,
              const T          alpha,
              const T          beta,
              cudaStream_t     stream)
    {
        const int rows = x.rows();
        const int cols = y.cols();

        const int blockThreads = 512;

        const int blocks = DIVIDE_UP(rows, blockThreads);

        for_each_item<<<blocks, blockThreads, 0, stream>>>(
            rows,
            [y, x, alpha, beta, cols] __device__(int i) mutable {
                for (int j = 0; j < cols; ++j) {
                    y(i, j) = alpha * x(i, j) + beta * y(i, j);
                }
            }

        );
    }

    /**
     * @brief implement r = b - s
     */
    void subtract(DenseMatT&       r,
                  const DenseMatT& b,
                  const DenseMatT& s,
                  cudaStream_t     stream)
    {
        const int rows = r.rows();
        const int cols = r.cols();

        const int blockThreads = 512;

        const int blocks = DIVIDE_UP(rows, blockThreads);

        for_each_item<<<blocks, blockThreads, 0, stream>>>(
            rows,
            [r, b, s, cols] __device__(int i) mutable {
                for (int j = 0; j < cols; ++j) {
                    r(i, j) = b(i, j) - s(i, j);
                }
            }

        );
    }


    /**
     * @brief initialize the residual and the direction in CG method
     */
    void init_PR(const DenseMatT& B,
                 const DenseMatT& S,
                 DenseMatT&       R,
                 DenseMatT&       P,
                 cudaStream_t     stream = NULL)
    {
        const int rows = B.rows();
        const int cols = B.cols();

        const int blockThreads = 512;

        const int blocks = DIVIDE_UP(rows, blockThreads);

        for_each_item<<<blocks, blockThreads, 0, stream>>>(
            rows,
            [P, B, R, S, cols] __device__(int i) mutable {
                for (int j = 0; j < cols; ++j) {
                    R(i, j) = B(i, j) - S(i, j);
                    P(i, j) = R(i, j);
                }
            }

        );
    };

   protected:
    virtual void mat_vec(const DenseMatT& in,
                         DenseMatT&       out,
                         cudaStream_t     stream)
    {
        if (A->cols() != in.rows() || A->rows() != out.rows() ||
            in.cols() != out.cols()) {
            RXMESH_ERROR(
                "CGSolver::mat_vec mismatch in the input/output size. A ({}, "
                "{}), In ({}, {}), Out ({}, {})",
                A->rows(),
                A->cols(),
                in.rows(),
                in.cols(),
                out.rows(),
                out.cols());
            return;
        }


        A->multiply(in, out, false, false, 1, 0, stream);
    }

    CGSolver(int num_rows,
             int unknown_dim,
             int max_iter,
             T   abs_tol,
             T   rel_tol,
             int reset_residual_freq)
        : IterativeSolver<T, DenseMatT>(max_iter, abs_tol, rel_tol),
          A(nullptr),
          alpha(0),
          beta(0),
          delta_new(0),
          delta_old(0),
          m_reset_residual_freq(reset_residual_freq),
          S(DenseMatT(num_rows, unknown_dim, DEVICE)),
          P(DenseMatT(num_rows, unknown_dim, DEVICE)),
          R(DenseMatT(num_rows, unknown_dim, DEVICE))
    {
    }

    SparseMatrix<T>* A;
    DenseMatT        S, P, R;
    T                alpha, beta, delta_new, delta_old;
    int              m_reset_residual_freq;
};

}  // namespace rxmesh
