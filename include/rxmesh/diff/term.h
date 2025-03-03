#pragma once

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/attribute.h"
#include "rxmesh/reduce_handle.h"

#include "rxmesh/diff/diff_query_kernel.cuh"
#include "rxmesh/diff/hessian_sparse_matrix.h"
#include "rxmesh/matrix/dense_matrix.cuh"


namespace rxmesh {

/**
 * @brief
 */
template <typename T, typename ObjHandleT>
struct Term
{
    virtual void eval_active(Attribute<T, ObjHandleT>& obj,
                             cudaStream_t              stream)  = 0;
    virtual void eval_passive(Attribute<T, ObjHandleT>& obj,
                              cudaStream_t              stream) = 0;
    virtual T    get_loss(cudaStream_t stream)     = 0;
};

/**
 * @brief
 */
template <typename LossHandleT,
          typename ObjHandleT,
          uint32_t blockThreads,
          Op       op,
          typename ScalarT,
          bool ProjectHess,
          int  VariableDim,
          typename LambdaT>
struct TemplatedTerm : public Term<typename ScalarT::PassiveType, ObjHandleT>
{
    using T = typename ScalarT::PassiveType;

    /**
     * @brief
     */
    TemplatedTerm(RXMeshStatic&                        rx,
                  LambdaT                              t,
                  bool                                 oreinted,
                  DenseMatrix<T, Eigen::RowMajor>&     grad,
                  HessianSparseMatrix<T, VariableDim>& hess)
        : term(t), rx(rx), grad(grad), hess(hess), oreinted(oreinted)
    {
        // TODO is it always 1
        loss = rx.add_attribute<T, LossHandleT>("Loss", 1);

        reducer = std::make_shared<ReduceHandle<T, LossHandleT>>(*loss);

        rx.prepare_launch_box({op},
                              lb_active,
                              (void*)detail::diff_kernel<blockThreads,
                                                         LossHandleT,
                                                         ObjHandleT,
                                                         op,
                                                         ScalarT,
                                                         true,
                                                         ProjectHess,
                                                         VariableDim,
                                                         LambdaT>,
                              oreinted);

        rx.prepare_launch_box({op},
                              lb_passive,
                              (void*)detail::diff_kernel<blockThreads,
                                                         LossHandleT,
                                                         ObjHandleT,
                                                         op,
                                                         ScalarT,
                                                         false,
                                                         ProjectHess,
                                                         VariableDim,
                                                         LambdaT>,
                              oreinted);
    }

    /**
     * @brief
     */
    void eval_active(Attribute<T, ObjHandleT>& obj, cudaStream_t stream)
    {
        rx.run_kernel(lb_active,
                      detail::diff_kernel<blockThreads,
                                          LossHandleT,
                                          ObjHandleT,
                                          op,
                                          ScalarT,
                                          true,
                                          ProjectHess,
                                          VariableDim,
                                          LambdaT>,
                      stream,
                      grad,
                      hess,
                      *loss,
                      obj,
                      oreinted,
                      term);
    }

    /**
     * @brief
     */
    void eval_passive(Attribute<T, ObjHandleT>& obj, cudaStream_t stream)
    {
        rx.run_kernel(lb_passive,
                      detail::diff_kernel<blockThreads,
                                          LossHandleT,
                                          ObjHandleT,
                                          op,
                                          ScalarT,
                                          false,
                                          ProjectHess,
                                          VariableDim,
                                          LambdaT>,
                      stream,
                      grad,
                      hess,
                      *loss,
                      obj,
                      oreinted,
                      term);
    }

    /**
     * @brief
     */
    T get_loss(cudaStream_t stream = NULL)
    {
        return reducer->reduce(*loss, cub::Sum(), 0, INVALID32, stream);
    }

    LambdaT term;

    std::shared_ptr<Attribute<T, LossHandleT>>    loss;
    std::shared_ptr<ReduceHandle<T, LossHandleT>> reducer;
    LaunchBox<blockThreads>                       lb_active;
    LaunchBox<blockThreads>                       lb_passive;

    bool oreinted;

    RXMeshStatic&                        rx;
    DenseMatrix<T, Eigen::RowMajor>&     grad;
    HessianSparseMatrix<T, VariableDim>& hess;
};
}  // namespace rxmesh