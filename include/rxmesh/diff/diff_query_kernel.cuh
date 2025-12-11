#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/context.h"
#include "rxmesh/iterator.cuh"
#include "rxmesh/kernels/for_each.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/util/meta.h"

#include "rxmesh/diff/hessian_projection.h"
#include "rxmesh/diff/hessian_sparse_matrix.h"
#include "rxmesh/diff/jacobian_sparse_matrix.h"
#include "rxmesh/diff/scalar.h"

#include "rxmesh/matrix/dense_matrix.h"

#include "rxmesh/diff/candidate_pairs.h"
#include "rxmesh/diff/diff_handle.h"
#include "rxmesh/diff/diff_iterator.h"

namespace rxmesh {
namespace detail {
// ============================== Scalar Kernels ==============================
// ============= Scalar Mat-vec
template <uint32_t blockThreads,
          typename LossHandleT,
          typename ObjHandleT,
          Op op,
          typename ScalarT,
          bool ProjectHess,
          int  VariableDim,
          typename LambdaT>
__global__ static void hess_matvec_scalar_kernel(
    const Context context,
    const DenseMatrix<typename ScalarT::PassiveType, Eigen::RowMajor>
                                                                input_vector,
    DenseMatrix<typename ScalarT::PassiveType, Eigen::RowMajor> output_vector,
    const Attribute<typename ScalarT::PassiveType, ObjHandleT>  objective,
    const bool                                                  oriented,
    LambdaT                                                     user_func)
{

    using IteratorT = typename IteratorType<op>::type;

    using IterHandleT = typename IteratorT::Handle;

    using PassiveT = typename ScalarT::PassiveType;

    assert(ScalarT::WithHessian_);

    auto block = cooperative_groups::this_thread_block();

    auto get_indices = [&](const ObjHandleT& row,
                           const ObjHandleT& col,
                           const int         local_i,
                           const int         local_j) {
        // this mimics how we calculate the strides in the sparse matrix
        const int r_id = context.linear_id(row) * VariableDim + local_i;

        const int c_id = context.linear_id(col) * VariableDim + local_j;

        return std::pair<int, int>(r_id, c_id);
    };

    // Unary queries
    if constexpr (op == Op::V || op == Op::E || op == Op::F) {

        for_each<op, blockThreads>(context, [&](const LossHandleT& fh) {
            // eval the objective function
            DiffHandle<ScalarT, LossHandleT> diff_handle(fh);

            ScalarT res = user_func(diff_handle, objective);

            // project Hessian to PD matrix
            if constexpr (ProjectHess) {
                project_positive_definite(res.hess());
            }

            for (int local_i = 0; local_i < VariableDim; ++local_i) {

                for (int local_j = 0; local_j < VariableDim; ++local_j) {

                    std::pair<int, int> ids =
                        get_indices(fh, fh, local_i, local_j);

                    // TODO we now assume solving single col vector
                    PassiveT p = res.hess()(local_i, local_j) *
                                 input_vector(ids.second, 0);

                    ::atomicAdd(&output_vector(ids.first, 0), p);
                }
            }
        });
    } else {
        // Binary query
        auto eval = [&](const LossHandleT& fh, const IteratorT& iter) {
            // eval the objective function
            DiffHandle<ScalarT, LossHandleT> diff_handle(fh);

            ScalarT res = user_func(diff_handle, iter, objective);


            // project Hessian to PD matrix
            if constexpr (ProjectHess) {
                project_positive_definite(res.hess());
            }

            // Hessian
            for (int i = 0; i < iter.size(); ++i) {
                const IterHandleT vi = iter[i];


                for (int j = 0; j < iter.size(); ++j) {
                    const IterHandleT vj = iter[j];


                    for (int local_i = 0; local_i < VariableDim; ++local_i) {

                        for (int local_j = 0; local_j < VariableDim;
                             ++local_j) {

                            std::pair<int, int> ids =
                                get_indices(vi, vj, local_i, local_j);

                            // TODO we now assume solving single col
                            // vector
                            PassiveT p =
                                res.hess()(
                                    index_mapping(VariableDim, i, local_i),
                                    index_mapping(VariableDim, j, local_j)) *
                                input_vector(ids.second, 0);

                            ::atomicAdd(&output_vector(ids.first, 0), p);
                        }
                    }
                }
            }
        };

        Query<blockThreads> query(context);

        ShmemAllocator shrd_alloc;

        query.dispatch<op>(block, shrd_alloc, eval, oriented);
    }
}

// ============= Scalar Passive
template <uint32_t blockThreads,
          typename LossHandleT,
          typename ObjHandleT,
          Op op,
          typename ScalarT,
          typename LambdaT>
__global__ static void diff_scalar_kernel_passive(
    const Context                                         context,
    Attribute<typename ScalarT::PassiveType, LossHandleT> loss,
    Attribute<typename ScalarT::PassiveType, ObjHandleT>  objective,
    const bool                                            oriented,
    LambdaT                                               user_func)
{

    using IteratorT = typename IteratorType<op>::type;

    using PassiveT = typename ScalarT::PassiveType;

    auto block = cooperative_groups::this_thread_block();

    // Unary queries
    if constexpr (op == Op::V || op == Op::E || op == Op::F) {

        for_each<op, blockThreads>(context, [&](const LossHandleT& fh) {
            DiffHandle<PassiveT, LossHandleT> diff_handle(fh);

            PassiveT res = user_func(diff_handle, objective);

            loss(fh) = res;
        });
    } else {
        // Binary query
        auto eval = [&](const LossHandleT& fh, const IteratorT& iter) {
            DiffHandle<PassiveT, LossHandleT> diff_handle(fh);

            PassiveT res = user_func(diff_handle, iter, objective);

            loss(fh) = res;
        };

        Query<blockThreads> query(context);

        ShmemAllocator shrd_alloc;

        query.dispatch<op>(block, shrd_alloc, eval, oriented);
    }
}

// ============= Scalar Active
template <uint32_t blockThreads,
          typename LossHandleT,
          typename ObjHandleT,
          Op op,
          typename ScalarT,
          bool ProjectHess,
          int  VariableDim,
          typename LambdaT>
__global__ static void diff_scalar_kernel_active(
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

    // Unary queries
    if constexpr (op == Op::V || op == Op::E || op == Op::F) {

        for_each<op, blockThreads>(context, [&](const LossHandleT& fh) {
            // eval the objective function
            DiffHandle<ScalarT, LossHandleT> diff_handle(fh);

            ScalarT res = user_func(diff_handle, objective);

            // function
            loss(fh) = res.val();

            // gradient
            for (int local = 0; local < VariableDim; ++local) {
                // we don't need atomics here since each thread update
                // the gradient of one element so there is no data race
                grad(fh, local) += res.grad()[local];
            }

            // Hessian
            if constexpr (WithHessian) {
                // project Hessian to PD matrix
                if constexpr (ProjectHess) {
                    project_positive_definite(res.hess());
                }

                for (int local_i = 0; local_i < VariableDim; ++local_i) {

                    for (int local_j = 0; local_j < VariableDim; ++local_j) {

                        hess(fh, fh, local_i, local_j) +=
                            res.hess()(local_i, local_j);
                    }
                }
            }
        });
    } else {
        // Binary query
        auto eval = [&](const LossHandleT& fh, const IteratorT& iter) {
            // eval the objective function
            DiffHandle<ScalarT, LossHandleT> diff_handle(fh);

            ScalarT res = user_func(diff_handle, iter, objective);

            // function
            loss(fh) = res.val();

            // gradient
            for (int i = 0; i < iter.size(); ++i) {
                for (int local = 0; local < VariableDim; ++local) {

                    ::atomicAdd(
                        &grad(iter[i], local),
                        res.grad()[index_mapping(VariableDim, i, local)]);
                }
            }

            if constexpr (WithHessian) {
                // project Hessian to PD matrix
                if constexpr (ProjectHess) {
                    project_positive_definite(res.hess());
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

                                ::atomicAdd(
                                    &hess(vi, vj, local_i, local_j),
                                    res.hess()(
                                        index_mapping(VariableDim, i, local_i),
                                        index_mapping(
                                            VariableDim, j, local_j)));
                            }
                        }
                    }
                }
            }
        };

        Query<blockThreads> query(context);

        ShmemAllocator shrd_alloc;

        query.dispatch<op>(block, shrd_alloc, eval, oriented);
    }
}

// ============= Scalar Passive Pairs
template <uint32_t blockThreads,
          typename LossHandleT,
          typename ObjHandleT,
          typename HandleT0,
          typename HandleT1,
          typename HessMatT,
          typename ScalarT,
          typename LambdaT>
__global__ static void diff_scalar_kernel_passive_pair(
    CandidatePairs<HandleT0, HandleT1, HessMatT>          pairs,
    Attribute<typename ScalarT::PassiveType, LossHandleT> loss,
    Attribute<typename ScalarT::PassiveType, ObjHandleT>  objective,
    LambdaT                                               user_func)
{
    static_assert(std::is_same_v<HandleT0, HandleT1>);

    using PassiveT = typename ScalarT::PassiveType;

    const uint32_t stride = blockThreads * gridDim.x;

    const int size = pairs.num_pairs();

    for (int id = threadIdx.x + blockThreads * blockIdx.x; id < size;
         id += stride) {

        uint64_t id64(id);

        DiffHandle<PassiveT, LossHandleT> diff_handle(id64);

        std::pair<HandleT0, HandleT1> pair = pairs.get_pair(id);

        PairIterator<HandleT0> iter(pair.first, pair.second);

        PassiveT res = user_func(diff_handle, iter, objective);

        //???? not sure which vertex should take the loss
        loss(pair.first) = res;
    }
}

// ============= Scalar Active Pairs
template <uint32_t blockThreads,
          typename LossHandleT,
          typename ObjHandleT,
          typename HandleT0,
          typename HandleT1,
          typename HessMatT,
          typename ScalarT,
          bool ProjectHess,
          int  VariableDim,
          typename LambdaT>
__global__ static void diff_scalar_kernel_active_pair(
    CandidatePairs<HandleT0, HandleT1, HessMatT>                    pairs,
    DenseMatrix<typename ScalarT::PassiveType, Eigen::RowMajor>     grad,
    HessianSparseMatrix<typename ScalarT::PassiveType, VariableDim> hess,
    Attribute<typename ScalarT::PassiveType, LossHandleT>           loss,
    Attribute<typename ScalarT::PassiveType, ObjHandleT>            objective,
    LambdaT                                                         user_func)
{
    static_assert(std::is_same_v<HandleT0, HandleT1>);

    using PassiveT = typename ScalarT::PassiveType;

    constexpr bool WithHessian = ScalarT::WithHessian_;

    const uint32_t stride = blockThreads * gridDim.x;

    const int size = pairs.num_pairs();

    for (int id = threadIdx.x + blockThreads * blockIdx.x; id < size;
         id += stride) {

        // hijacking DiffHandle to 1) pass the pair index to the user, 2)
        // allow the user to extract the type of the ScalarT (using
        // ACTIVE_TYPE(id))
        uint64_t id64(id);

        DiffHandle<ScalarT, LossHandleT> diff_handle(id64);

        std::pair<HandleT0, HandleT1> pair = pairs.get_pair(id);

        assert(hess.is_non_zero(pair.first, pair.second));

        PairIterator<HandleT0> iter(pair.first, pair.second);

        ScalarT res = user_func(diff_handle, iter, objective);

        //???? not sure which vertex should take the loss
        loss(pair.first) = res.val();

        // gradient
        for (int i = 0; i < iter.size(); ++i) {
            for (int local = 0; local < VariableDim; ++local) {
                ::atomicAdd(&grad(iter[i], local),
                            res.grad()[index_mapping(VariableDim, i, local)]);
            }
        }


        if constexpr (WithHessian) {
            // project Hessian to PD matrix
            if constexpr (ProjectHess) {
                project_positive_definite(res.hess());
            }

            // Hessian
            for (int i = 0; i < iter.size(); ++i) {
                const HandleT0 vi = iter[i];

                for (int j = 0; j < iter.size(); ++j) {
                    const HandleT0 vj = iter[j];

                    for (int local_i = 0; local_i < VariableDim; ++local_i) {

                        for (int local_j = 0; local_j < VariableDim;
                             ++local_j) {

                            ::atomicAdd(
                                &hess(vi, vj, local_i, local_j),
                                res.hess()(
                                    index_mapping(VariableDim, i, local_i),
                                    index_mapping(VariableDim, j, local_j)));
                        }
                    }
                }
            }
        }
    }
}


// ============================== Vector Kernels ==============================
// ============= Vector Passive
template <uint32_t blockThreads,
          typename LossHandleT,
          typename ObjHandleT,
          Op  op,
          int InputDim,
          typename ScalarT,
          typename LambdaT>
__global__ static void diff_vector_kernel_passive(
    const Context                                               context,
    DenseMatrix<typename ScalarT::PassiveType, Eigen::RowMajor> residual,
    Attribute<typename ScalarT::PassiveType, ObjHandleT>        objective,
    const bool                                                  oriented,
    LambdaT                                                     user_func)
{

    using PassiveT = typename ScalarT::PassiveType;

    auto block = cooperative_groups::this_thread_block();

    const int num_input_elements = context.get_num<LossHandleT>();

    assert(residual.cols() == 1);

    assert(residual.rows() == num_input_elements * InputDim);

    residual.reshape(num_input_elements, InputDim);

    // Unary queries
    if constexpr (op == Op::V || op == Op::E || op == Op::F) {

        for_each<op, blockThreads>(context, [&](const LossHandleT& fh) {
            DiffHandle<PassiveT, LossHandleT> diff_handle(fh);

            Eigen::Vector<PassiveT, InputDim> res =
                user_func(diff_handle, objective);

            for (int i = 0; i < InputDim; ++i) {
                residual(fh, i) = res[i];
            }
        });
    } else {
        // Binary query
        using IteratorT = typename IteratorType<op>::type;

        auto eval = [&](const LossHandleT& fh, const IteratorT& iter) {
            DiffHandle<PassiveT, LossHandleT> diff_handle(fh);

            Eigen::Vector<PassiveT, InputDim> res =
                user_func(diff_handle, iter, objective);

            for (int i = 0; i < InputDim; ++i) {
                residual(fh, i) = res[i];
            }
        };

        Query<blockThreads> query(context);

        ShmemAllocator shrd_alloc;

        query.dispatch<op>(block, shrd_alloc, eval, oriented);
    }
}


// ============= Vector Active
template <uint32_t blockThreads,
          typename LossHandleT,
          typename ObjHandleT,
          Op  op,
          int InputDim,
          typename ScalarT,
          int VariableDim,
          typename LambdaT>
__global__ static void diff_vector_kernel_active(
    const Context                                               context,
    JacobianSparseMatrix<typename ScalarT::PassiveType>         jac,
    DenseMatrix<typename ScalarT::PassiveType, Eigen::RowMajor> residual,
    Attribute<typename ScalarT::PassiveType, ObjHandleT>        objective,
    const bool                                                  oriented,
    LambdaT                                                     user_func)
{

    using IteratorT = typename IteratorType<op>::type;

    using PassiveT = typename ScalarT::PassiveType;

    auto block = cooperative_groups::this_thread_block();

    const int num_input_elements = context.get_num<LossHandleT>();

    assert(residual.cols() == 1);


    assert(residual.rows() == num_input_elements * InputDim);

    residual.reshape(num_input_elements, InputDim);


    // Unary queries
    if constexpr (op == Op::V || op == Op::E || op == Op::F) {

        for_each<op, blockThreads>(context, [&](const LossHandleT& fh) {
            // eval the objective function
            DiffHandle<ScalarT, LossHandleT> diff_handle(fh);

            Eigen::Vector<ScalarT, InputDim> res =
                user_func(diff_handle, objective);

            // residual
            for (int i = 0; i < InputDim; ++i) {
                residual(fh, i) = res[i].val();
            }

            // TODO jacobian
            for (int local = 0; local < VariableDim; ++local) {
                // we don't need atomics here since each thread update
                // the gradient of one element so there is no data race
                // grad(fh, local) += res.grad()[local];
            }
        });
    } else {
        // Binary query
        auto eval = [&](const LossHandleT& fh, const IteratorT& iter) {
            // eval the objective function
            DiffHandle<ScalarT, LossHandleT> diff_handle(fh);

            Eigen::Vector<ScalarT, InputDim> res =
                user_func(diff_handle, iter, objective);

            // residual
            for (int i = 0; i < InputDim; ++i) {
                residual(fh, i) = res[i].val();
            }


            // TODO jacobian
            for (int i = 0; i < iter.size(); ++i) {
                for (int local = 0; local < VariableDim; ++local) {

                    //::atomicAdd(
                    //    &grad(iter[i], local),
                    //    res.grad()[index_mapping(VariableDim, i, local)]);
                }
            }
        };

        Query<blockThreads> query(context);

        ShmemAllocator shrd_alloc;

        query.dispatch<op>(block, shrd_alloc, eval, oriented);
    }
}

}  // namespace detail
}  // namespace rxmesh