#pragma once

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/hessian_sparse_matrix.h"
#include "rxmesh/diff/term.h"
#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/types.h"

namespace rxmesh {

/**
 * @brief Definition of differentaion problem
 * @tparam T the underlying (passive) type of the problem, e.g., float or double
 * @tparam HandleT the type of the mesh element with respect to which the
 * differentiation is being performed (e.g., VertexHandle for mesh
 * paramterization)
 * @tparam VariableDim the dimensions of the active variable defined on each
 * mesh element under consideration (e.g., 2 for mesh paramterization)
 */
template <typename T, int VariableDim, typename HandleT>
struct DiffProblem
{

    DenseMatrix<T, Eigen::RowMajor>     grad;
    HessianSparseMatrix<T, VariableDim> hess;
    DenseMatrix<T, Eigen::RowMajor>     dir;
    std::vector<std::shared_ptr<Term>>  terms;

    // TODO NElements ??
    using ScalarT = Scalar<T, 1 /*NElements*/, true>;

    template <bool Active>
    using ActiveT = std::conditional_t<Active, ScalarT, T>;

    DiffProblem(RXMeshStatic& rx)
        : grad(DenseMatrix<T, Eigen::RowMajor>(rx,
                                               rx.get_num_elements<HandleT>(),
                                               VariableDim)),
          hess(HessianSparseMatrix<T, VariableDim>(rx)),
          dir(DenseMatrix<T, Eigen::RowMajor>(rx,
                                              rx.get_num_elements<HandleT>(),
                                              VariableDim))
    {
    }


    template <Op       op,
              bool     ProjectHess,
              uint32_t blockThreads = 256,
              typename LambdaT      = void>
    void add_term(RXMeshStatic& rx, LambdaT t, bool oreinted = false)
    {

        if constexpr (op == Op::VV || op == Op::VE || op == Op::VF) {
            auto new_term =
                std::make_shared<TemplatedTerm<VertexHandle,
                                               blockThreads,
                                               op,
                                               ScalarT,
                                               ProjectHess,
                                               VariableDim,
                                               LambdaT>>(rx, t, oreinted);
            terms.push_back(std::dynamic_pointer_cast<Term>(new_term));
        }

        if constexpr (op == Op::EV || op == Op::EE || op == Op::EF) {
            auto new_term =
                std::make_shared<TemplatedTerm<EdgeHandle,
                                               blockThreads,
                                               op,
                                               ScalarT,
                                               ProjectHess,
                                               VariableDim,
                                               LambdaT>>(rx, t, oreinted);
            terms.push_back(std::dynamic_pointer_cast<Term>(new_term));
        }

        if constexpr (op == Op::FV || op == Op::FE || op == Op::FF) {
            auto new_term =
                std::make_shared<TemplatedTerm<FaceHandle,
                                               blockThreads,
                                               op,
                                               ScalarT,
                                               ProjectHess,
                                               VariableDim,
                                               LambdaT>>(rx, t, oreinted);
            terms.push_back(std::dynamic_pointer_cast<Term>(new_term));
        }
    }


    void eval(cudaStream_t stream = NULL)
    {
        grad.reset(0, DEVICE);
        hess.reset(0, DEVICE);
    }
};

}  // namespace rxmesh