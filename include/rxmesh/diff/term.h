#pragma once

#include <sstream>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/attribute.h"
#include "rxmesh/reduce_handle.h"

#include "rxmesh/diff/diff_query_kernel.cuh"
#include "rxmesh/diff/hessian_sparse_matrix.h"
#include "rxmesh/matrix/dense_matrix.h"


namespace rxmesh {

/**
 * @brief pure virtual class used as interface for all energy terms (without
 * specifying the which type of mesh elements it is specified on, number of
 * variables). This is used to store the all energies inside DiffScalarProblem
 */
template <typename T, typename ObjHandleT>
struct Term
{
    virtual void eval_active(Attribute<T, ObjHandleT>& obj,
                             cudaStream_t              stream) = 0;

    virtual void eval_active_grad_only(Attribute<T, ObjHandleT>& obj,
                                       cudaStream_t              stream) = 0;

    virtual void eval_passive(Attribute<T, ObjHandleT>& obj,
                              cudaStream_t              stream) = 0;

    virtual void eval_active_matvec(
        Attribute<T, ObjHandleT>&              obj,
        const DenseMatrix<T, Eigen::RowMajor>& input,
        DenseMatrix<T, Eigen::RowMajor>&       output,
        cudaStream_t                           stream) = 0;

    virtual T get_loss(cudaStream_t stream) = 0;
};

/**
 * @brief concrete class that defines energy terms for specific mesh type,
 * specific number of variable, etc.
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

    // Scalar type that only store the 1st derivative
    // This will be the same as ScalarT if ScalarT has WithHessian=false
    using ScalarGradOnlyT = Scalar<T, ScalarT::k_, false>;


    TemplatedTerm(RXMeshStatic&                        rx,
                  LambdaT                              t,
                  bool                                 oreinted,
                  DenseMatrix<T, Eigen::RowMajor>&     grad,
                  HessianSparseMatrix<T, VariableDim>& hess)
        : term(t), rx(rx), grad(grad), hess(hess), oreinted(oreinted)
    {
        // TODO is it always 1

        // To avoid the clash that happens from adding many losses.
        std::ostringstream address;
        address << (void const*)this;
        std::string name = address.str();

        loss = rx.add_attribute<T, LossHandleT>("Loss" + name, 1);

        reducer = std::make_shared<ReduceHandle<T, LossHandleT>>(*loss);

        rx.prepare_launch_box({op},
                              lb_active,
                              (void*)detail::diff_kernel_active<blockThreads,
                                                                LossHandleT,
                                                                ObjHandleT,
                                                                op,
                                                                ScalarT,
                                                                ProjectHess,
                                                                VariableDim,
                                                                LambdaT>,
                              oreinted);

        rx.prepare_launch_box({op},
                              lb_passive,
                              (void*)detail::diff_kernel_passive<blockThreads,
                                                                 LossHandleT,
                                                                 ObjHandleT,
                                                                 op,
                                                                 ScalarT,
                                                                 LambdaT>,
                              oreinted);


        rx.prepare_launch_box({op},
                              lb_active_matvec,
                              (void*)detail::hess_matvec_kernel<blockThreads,
                                                                LossHandleT,
                                                                ObjHandleT,
                                                                op,
                                                                ScalarT,
                                                                ProjectHess,
                                                                VariableDim,
                                                                LambdaT>,
                              oreinted);

        rx.prepare_launch_box({op},
                              lb_active_grad_only,
                              (void*)detail::diff_kernel_active<blockThreads,
                                                                LossHandleT,
                                                                ObjHandleT,
                                                                op,
                                                                ScalarGradOnlyT,
                                                                ProjectHess,
                                                                VariableDim,
                                                                LambdaT>,
                              oreinted);
    }

    /**
     * @brief Evaluate the energy term using active/differentiable type
     */
    void eval_active(Attribute<T, ObjHandleT>& obj, cudaStream_t stream)
    {
        rx.run_kernel(lb_active,
                      detail::diff_kernel_active<blockThreads,
                                                 LossHandleT,
                                                 ObjHandleT,
                                                 op,
                                                 ScalarT,
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
     * @brief Evaluate the energy term using active/differentiable type for the
     * 1st derivative only
     */
    void eval_active_grad_only(Attribute<T, ObjHandleT>& obj,
                               cudaStream_t              stream)
    {
        rx.run_kernel(lb_active_grad_only,
                      detail::diff_kernel_active<blockThreads,
                                                 LossHandleT,
                                                 ObjHandleT,
                                                 op,
                                                 ScalarGradOnlyT,
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
     * @brief Evaluate the energy term using active/differentiable type but
     * without constructing the Hessian. Instead, we do matvec with the Hessian
     */
    void eval_active_matvec(Attribute<T, ObjHandleT>&              obj,
                            const DenseMatrix<T, Eigen::RowMajor>& input,
                            DenseMatrix<T, Eigen::RowMajor>&       output,
                            cudaStream_t                           stream)
    {
        if (!ScalarT::WithHessian_) {
            RXMESH_ERROR(
                "TemplatedTerm::eval_active_matvec() can not run with scalar "
                "type that does not have Hessians. Returning without "
                "evolution.");
            return;
        }
        rx.run_kernel(lb_active,
                      detail::hess_matvec_kernel<blockThreads,
                                                 LossHandleT,
                                                 ObjHandleT,
                                                 op,
                                                 ScalarT,
                                                 ProjectHess,
                                                 VariableDim,
                                                 LambdaT>,
                      stream,
                      input,
                      output,
                      obj,
                      oreinted,
                      term);
    }

    /**
     * @brief Evaluate the energy term using non-active/non-differentiable type
     */
    void eval_passive(Attribute<T, ObjHandleT>& obj, cudaStream_t stream)
    {
        rx.run_kernel(lb_passive,
                      detail::diff_kernel_passive<blockThreads,
                                                  LossHandleT,
                                                  ObjHandleT,
                                                  op,
                                                  ScalarT,
                                                  LambdaT>,
                      stream,
                      *loss,
                      obj,
                      oreinted,
                      term);
    }

    /**
     * @brief get the current loss of the energy. Should be called evaluating
     * the term using active type
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
    LaunchBox<blockThreads>                       lb_active_matvec;
    LaunchBox<blockThreads>                       lb_active_grad_only;


    bool oreinted;

    RXMeshStatic&                        rx;
    DenseMatrix<T, Eigen::RowMajor>&     grad;
    HessianSparseMatrix<T, VariableDim>& hess;
};
}  // namespace rxmesh