#pragma once

#include <sstream>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/attribute.h"
#include "rxmesh/diff/diff_query_kernel.cuh"
#include "rxmesh/diff/jacobian_sparse_matrix.h"
#include "rxmesh/matrix/dense_matrix.h"


namespace rxmesh {

/**
 * @brief pure virtual class used as interface for all energy terms (without
 * specifying the which type of mesh elements it is specified on, number of
 * variables). This is used to store the all energies inside DiffScalarProblem
 */
template <typename T, typename ObjHandleT>
struct VectorTerm
{
    virtual void eval_active(int                              term_id,
                             Attribute<T, ObjHandleT>&        obj,
                             DenseMatrix<T, Eigen::RowMajor>& residual,
                             JacobianSparseMatrix<T>&         jac,
                             cudaStream_t                     stream) = 0;

    virtual void eval_passive(Attribute<T, ObjHandleT>&        obj,
                              DenseMatrix<T, Eigen::RowMajor>& residual,
                              cudaStream_t                     stream) = 0;
};

/**
 * @brief concrete class that defines energy terms for specific mesh type,
 * specific number of variable, etc. Used to define energy terms that require
 * local query operations, e.g., FV
 */
template <typename LossHandleT,
          typename ObjHandleT,
          uint32_t blockThreads,
          Op       op,
          typename ScalarT,
          int InputDim,
          int VariableDim,
          typename LambdaT>
struct TemplatedVectorTerm
    : public VectorTerm<typename ScalarT::PassiveType, ObjHandleT>
{
    using T = typename ScalarT::PassiveType;


    TemplatedVectorTerm(RXMeshStatic& rx, LambdaT t, bool oreinted)
        : term(t), rx(rx), oreinted(oreinted)
    {
        // To avoid the clash that happens from adding many losses.
        std::ostringstream address;
        address << (void const*)this;
        std::string name = address.str();

        rx.prepare_launch_box(
            {op},
            lb_active,
            (void*)detail::diff_vector_kernel_active<blockThreads,
                                                     LossHandleT,
                                                     ObjHandleT,
                                                     op,
                                                     InputDim,
                                                     ScalarT,
                                                     VariableDim,
                                                     LambdaT>,
            oreinted);

        rx.prepare_launch_box(
            {op},
            lb_passive,
            (void*)detail::diff_vector_kernel_passive<blockThreads,
                                                      LossHandleT,
                                                      ObjHandleT,
                                                      op,
                                                      InputDim,
                                                      ScalarT,
                                                      LambdaT>,
            oreinted);
    }

    /**
     * @brief Evaluate the energy term using active/differentiable type
     */
    void eval_active(int                              term_id,
                     Attribute<T, ObjHandleT>&        obj,
                     DenseMatrix<T, Eigen::RowMajor>& residual,
                     JacobianSparseMatrix<T>&         jac,
                     cudaStream_t                     stream)
    {
        rx.run_kernel(lb_active,
                      detail::diff_vector_kernel_active<blockThreads,
                                                        LossHandleT,
                                                        ObjHandleT,
                                                        op,
                                                        InputDim,
                                                        ScalarT,
                                                        VariableDim,
                                                        LambdaT>,
                      stream,
                      term_id,
                      jac,
                      residual,
                      obj,
                      oreinted,
                      term);
    }


    /**
     * @brief Evaluate the energy term using non-active/non-differentiable type
     */
    void eval_passive(Attribute<T, ObjHandleT>&        obj,
                      DenseMatrix<T, Eigen::RowMajor>& residual,
                      cudaStream_t                     stream)
    {
        rx.run_kernel(lb_passive,
                      detail::diff_vector_kernel_passive<blockThreads,
                                                         LossHandleT,
                                                         ObjHandleT,
                                                         op,
                                                         InputDim,
                                                         ScalarT,
                                                         LambdaT>,
                      stream,
                      residual,
                      obj,
                      oreinted,
                      term);
    }


    LambdaT term;


    LaunchBox<blockThreads> lb_active;
    LaunchBox<blockThreads> lb_passive;

    bool oreinted;

    RXMeshStatic& rx;
};


}  // namespace rxmesh