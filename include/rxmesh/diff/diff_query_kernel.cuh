#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/context.h"
#include "rxmesh/iterator.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/util/meta.h"

#include "rxmesh/diff/hessian_projection.h"
#include "rxmesh/diff/hessian_sparse_matrix.h"
#include "rxmesh/diff/scalar.h"
#include "rxmesh/matrix/dense_matrix.h"

#include "rxmesh/diff/diff_handle.h"
#include "rxmesh/diff/diff_iterator.h"

namespace rxmesh {
namespace detail {
template <uint32_t blockThreads,
          typename LossHandleT,
          typename ObjHandleT,
          Op op,
          typename ScalarT,
          bool Active,
          bool ProjectHess,
          int  VariableDim,
          typename LambdaT>
__global__ static void diff_kernel(
    const Context                                                   context,
    DenseMatrix<typename ScalarT::PassiveType, Eigen::RowMajor>     grad,
    HessianSparseMatrix<typename ScalarT::PassiveType, VariableDim> hess,
    Attribute<typename ScalarT::PassiveType, LossHandleT>           loss,
    Attribute<typename ScalarT::PassiveType, ObjHandleT>            objective,
    const bool                                                      oriented,
    LambdaT                                                         user_func)
{

    using IteratorT = typename IteratorType<op>::type;

    using IterHandleT = typename IteratorT::Handle;

    using PassiveT = typename ScalarT::PassiveType;

    constexpr bool WithHessian = ScalarT::WithHessian_;

    auto block = cooperative_groups::this_thread_block();


    auto eval = [&](const LossHandleT& fh, const IteratorT& iter) {
        if constexpr (Active) {
            // eval the objective function

            DiffHandle<ScalarT, LossHandleT> diff_handle(fh);

            ScalarT res = user_func(diff_handle, iter, objective);

            // function
            loss(fh) = res.val;

            // gradient
            for (uint16_t i = 0; i < iter.size(); ++i) {
                for (int local = 0; local < VariableDim; ++local) {

                    ::atomicAdd(&grad(iter[i], local),
                                res.grad[index_mapping<VariableDim>(i, local)]);
                }
            }

            if constexpr (WithHessian) {
                // project Hessian to PD matrix
                if constexpr (ProjectHess) {
                    project_positive_definite(res.Hess);
                }

                // Hessian
                for (int i = 0; i < iter.size(); ++i) {
                    const IterHandleT vi = iter[i];

                    for (int j = 0; j < iter.size(); ++j) {
                        const IterHandleT vj = iter[j];

                        for (int local_i = 0; local_i < VariableDim;
                             ++local_i) {

                            for (int local_j = 0; local_j < VariableDim;
                                 ++local_j) {

                                ::atomicAdd(&hess(vi, vj, local_i, local_j),
                                            res.Hess(index_mapping<VariableDim>(
                                                         i, local_i),
                                                     index_mapping<VariableDim>(
                                                         j, local_j)));
                            }
                        }
                    }
                }
            }
        } else {

            DiffHandle<PassiveT, LossHandleT> diff_handle(fh);

            PassiveT res = user_func(diff_handle, iter, objective);

            loss(fh) = res;
        }
    };

    Query<blockThreads> query(context);

    ShmemAllocator shrd_alloc;

    query.dispatch<op>(block, shrd_alloc, eval, oriented);
}
}  // namespace detail
}  // namespace rxmesh