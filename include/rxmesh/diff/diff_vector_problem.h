#pragma once

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/element_valence.h"
#include "rxmesh/diff/jacobian_sparse_matrix.h"
#include "rxmesh/diff/vector_term.h"
#include "rxmesh/matrix/dense_matrix.h"


namespace rxmesh {

template <typename T, int VariableDim, typename OptVarHandleT>
struct DiffVectorProblem
{
    using DenseMatT = DenseMatrix<T, Eigen::RowMajor>;
    using JacSpMatT = JacobianSparseMatrix<T>;

    using IndexT = typename DenseMatT::IndexT;

    RXMeshStatic& rx;

    std::shared_ptr<Attribute<T, OptVarHandleT>> opt_var;
    std::shared_ptr<DenseMatT>                   residual;
    std::shared_ptr<DenseMatT>                   grad;

    // residual_reshaped is a view in residual for different terms
    std::vector<DenseMatT>     residual_reshaped;
    std::shared_ptr<JacSpMatT> jac;
    std::vector<Op>            ops;
    std::vector<BlockShape>    block_shapes;
    std::vector<std::shared_ptr<VectorTerm<T, OptVarHandleT>>> terms;
    bool is_prep_eval_called;


    DiffVectorProblem(RXMeshStatic& rx)
        : rx(rx),
          opt_var(rx.add_attribute<T, OptVarHandleT>("opt_var", VariableDim)),
          is_prep_eval_called(false)
    {
    }

    /**
     * @brief add a (energy) vector term to the loss function that depends on
     * local query operation (e.g., FV)
     */
    template <Op       op,
              int      InputDim,
              uint32_t blockThreads = 256,
              typename LambdaT      = void>
    void add_term(LambdaT t, bool oreinted = false)
    {

        constexpr int ElementValence = element_valence<op>();

        constexpr int NElements =
            std::max(VariableDim * ElementValence, Eigen::Dynamic);

        using ScalarT = Scalar<T, NElements, false>;

        if constexpr (op == Op::VV || op == Op::VE || op == Op::VF ||
                      op == Op::V) {
            auto new_term =
                std::make_shared<TemplatedVectorTerm<VertexHandle,
                                                     OptVarHandleT,
                                                     blockThreads,
                                                     op,
                                                     ScalarT,
                                                     InputDim,
                                                     VariableDim,
                                                     LambdaT>>(rx, t, oreinted);
            terms.push_back(
                std::dynamic_pointer_cast<VectorTerm<T, OptVarHandleT>>(
                    new_term));
        }

        if constexpr (op == Op::EV || op == Op::EE || op == Op::EF ||
                      op == Op::E || op == Op::EVDiamond) {
            auto new_term =
                std::make_shared<TemplatedVectorTerm<EdgeHandle,
                                                     OptVarHandleT,
                                                     blockThreads,
                                                     op,
                                                     ScalarT,
                                                     InputDim,
                                                     VariableDim,
                                                     LambdaT>>(rx, t, oreinted);
            terms.push_back(
                std::dynamic_pointer_cast<VectorTerm<T, OptVarHandleT>>(
                    new_term));
        }

        if constexpr (op == Op::FV || op == Op::FE || op == Op::FF ||
                      op == Op::F) {
            auto new_term =
                std::make_shared<TemplatedVectorTerm<FaceHandle,
                                                     OptVarHandleT,
                                                     blockThreads,
                                                     op,
                                                     ScalarT,
                                                     InputDim,
                                                     VariableDim,
                                                     LambdaT>>(rx, t, oreinted);
            terms.push_back(
                std::dynamic_pointer_cast<VectorTerm<T, OptVarHandleT>>(
                    new_term));
        }

        ops.push_back(op);
        block_shapes.push_back({InputDim, VariableDim});
    }


    /**
     * @brief add a (energy) vector term to the loss function that depends on
     * local query operation (e.g., FV)
     */
    /* template <Op       op,
               int      InputDim,
               int      OutputDimStart,
               int      OutputDimEnd,
               uint32_t blockThreads = 256,
               typename LambdaT      = void>
     void add_term(LambdaT t, bool oreinted = false)
     {
         constexpr int OutputVariableDim = OutputDimEnd - OutputDimStart;

         constexpr int ElementValence = element_valence<op>();

         constexpr int NElements =
             std::max(OutputVariableDim * ElementValence, Eigen::Dynamic);

         using ScalarT = Scalar<T, NElements, false>;

         if constexpr (op == Op::VV || op == Op::VE || op == Op::VF ||
                       op == Op::V) {
             auto new_term =
                 std::make_shared<TemplatedVectorTerm<VertexHandle,
                                                      OptVarHandleT,
                                                      blockThreads,
                                                      op,
                                                      ScalarT,
                                                      InputDim,
                                                      OutputVariableDim,
                                                      LambdaT>>(rx, t,
     oreinted); terms.push_back( std::dynamic_pointer_cast<VectorTerm<T,
     OptVarHandleT>>( new_term));
         }

         if constexpr (op == Op::EV || op == Op::EE || op == Op::EF ||
                       op == Op::E || op == Op::EVDiamond) {
             auto new_term =
                 std::make_shared<TemplatedVectorTerm<EdgeHandle,
                                                      OptVarHandleT,
                                                      blockThreads,
                                                      op,
                                                      ScalarT,
                                                      InputDim,
                                                      OutputVariableDim,
                                                      LambdaT>>(rx, t,
     oreinted); terms.push_back( std::dynamic_pointer_cast<VectorTerm<T,
     OptVarHandleT>>( new_term));
         }

         if constexpr (op == Op::FV || op == Op::FE || op == Op::FF ||
                       op == Op::F) {
             auto new_term =
                 std::make_shared<TemplatedVectorTerm<FaceHandle,
                                                      OptVarHandleT,
                                                      blockThreads,
                                                      op,
                                                      ScalarT,
                                                      InputDim,
                                                      OutputVariableDim,
                                                      LambdaT>>(rx, t,
     oreinted); terms.push_back( std::dynamic_pointer_cast<VectorTerm<T,
     OptVarHandleT>>( new_term));
         }

         ops.push_back(op);
         block_shapes.push_back({InputDim, OutputVariableDim});
     }*/

    /**
     * @brief prepare the Jacobian for evaluation (i.e., allocate memory). this
     * function should be called before eval.
     */
    void prep_eval()
    {

        is_prep_eval_called = true;

        jac = std::make_shared<JacSpMatT>(rx, ops, block_shapes);

        // allocate residual
        residual =
            std::make_shared<DenseMatT>(rx, jac->rows(), 1, LOCATION_ALL);

        grad = std::make_shared<DenseMatT>(rx, jac->cols(), 1, LOCATION_ALL);

        // populate residual_reshaped
        for (int i = 0; i < jac->get_num_terms(); ++i) {

            auto [st, end] = jac->get_term_rows_range(i);

            residual_reshaped.emplace_back(residual->segment_range(st, end));
        }

        jac->alloc_multiply_buffer(*residual, *grad, true, false);
    }

    /**
     * @brief evaluate all terms in active mode and calculate the Jacobian
     */
    void eval_terms(cudaStream_t stream = NULL)
    {
        if (!is_prep_eval_called) {
            prep_eval();
        }

        jac->reset(0, DEVICE, stream);

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_active(
                i, *opt_var, residual_reshaped[i], *jac, stream);
        }
    }

    /**
     * @brief evaluate all terms in active mode and calculate Jacobin and grad
     * where grad = scale*Jac^T*res
     */
    void eval_terms_sum_of_squares(T scale = T(1.), cudaStream_t stream = NULL)
    {
        eval_terms(stream);

        jac->multiply(*residual, *grad, true, false, scale, T(0.), stream);
    }

    /**
     * @brief compute the current loss as norm2 of the residual
     */
    T get_current_loss(cudaStream_t stream = NULL)
    {
        return residual->norm2(stream);
    }


    /**
     * @brief evaluate all terms in passive mode
     */
    void eval_terms_passive(Attribute<T, OptVarHandleT>* opt_var_in = nullptr,
                            cudaStream_t                 stream     = NULL)
    {
        if (!is_prep_eval_called) {
            prep_eval();
        }

        for (size_t i = 0; i < terms.size(); ++i) {
            if (opt_var_in) {
                terms[i]->eval_passive(
                    *opt_var_in, residual_reshaped[i], stream);
            } else {
                terms[i]->eval_passive(*opt_var, residual_reshaped[i], stream);
            }
        }
    }
};

}  // namespace rxmesh
