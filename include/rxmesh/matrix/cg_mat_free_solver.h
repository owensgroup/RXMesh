#pragma once

#include "rxmesh/matrix/cg_solver.h"


namespace rxmesh {

/**
 * @brief (Un-preconditioned) matrix-free CG
 */
template <typename T, int DenseMatOrder = Eigen::ColMajor>
struct CGMatFreeSolver : public CGSolver<T, DenseMatOrder>
{
    using DenseMatT = DenseMatrix<T, DenseMatOrder>;

    using MatVecT =
        std::function<void(const DenseMatT&, DenseMatT&, cudaStream_t)>;

    MatVecT m_mat_vec;

    CGMatFreeSolver(int num_rows,
                    int unkown_dim,  // 1D, 2D, 3D
                    int max_iter,
                    T   abs_tol             = 1e-6,
                    T   rel_tol             = 0.0,
                    int reset_residual_freq = std::numeric_limits<int>::max())
        : CGSolver<T, DenseMatOrder>(num_rows,
                                     unkown_dim,
                                     max_iter,
                                     abs_tol,
                                     rel_tol,
                                     reset_residual_freq)
    {
    }

    virtual std::string name() override
    {
        return std::string("CG Matrix Free");
    }

    virtual ~CGMatFreeSolver()
    {
    }

   protected:
    virtual void mat_vec(const DenseMatT& in,
                         DenseMatT&       out,
                         cudaStream_t     stream) override
    {
        m_mat_vec(in, out, stream);
    }
};

}  // namespace rxmesh