#pragma once

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"

#include <Eigen/Dense>
namespace rxmesh {

template <typename SpMatT, int DenseMatOrder = Eigen::ColMajor>
struct SolverBase
{
    using Type = typename SpMatT::Type;

    SolverBase(SpMatT* mat) : m_mat(mat)
    {
    }

    virtual ~SolverBase()
    {
    }

    virtual void pre_solve(RXMeshStatic& rx) = 0;

    virtual void solve(DenseMatrix<Type, DenseMatOrder>& B_mat,
                       DenseMatrix<Type, DenseMatOrder>& X_mat,
                       cudaStream_t                      stream = NULL) = 0;

    virtual std::string name() = 0;

   protected:
    SpMatT* m_mat;
};

}  // namespace rxmesh