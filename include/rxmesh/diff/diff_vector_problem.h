#pragma once

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/element_valence.h"
#include "rxmesh/diff/jacobian_sparse_matrix.h"
#include "rxmesh/diff/vector_term.h"
#include "rxmesh/matrix/dense_matrix.h"


namespace rxmesh {

template <typename T, int VariableDim, typename ObjHandleT>
struct DiffVectorProblem
{
    using DenseMatT = DenseMatrix<T, Eigen::RowMajor>;
    using JacSpMatT = JacobianSparseMatrix<T>;

    using IndexT = typename DenseMatT::IndexT;

    RXMeshStatic& rx;

    std::shared_ptr<Attribute<T, ObjHandleT>>               objective;
    std::shared_ptr<DenseMatT>                              residual;
    std::vector<DenseMatT>                                  residual_reshaped;
    std::shared_ptr<JacSpMatT>                              jac;
    std::vector<Op>                                         ops;
    std::vector<BlockShape>                                 block_shapes;
    std::vector<std::shared_ptr<VectorTerm<T, ObjHandleT>>> terms;
    bool                                                    is_prep_eval_called;


    DiffVectorProblem(RXMeshStatic& rx)
        : rx(rx),
          objective(rx.add_attribute<T, ObjHandleT>("objective", VariableDim)),
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
                                                     ObjHandleT,
                                                     blockThreads,
                                                     op,
                                                     ScalarT,
                                                     InputDim,
                                                     VariableDim,
                                                     LambdaT>>(rx, t, oreinted);
            terms.push_back(
                std::dynamic_pointer_cast<VectorTerm<T, ObjHandleT>>(new_term));
        }

        if constexpr (op == Op::EV || op == Op::EE || op == Op::EF ||
                      op == Op::E || op == Op::EVDiamond) {
            auto new_term =
                std::make_shared<TemplatedVectorTerm<EdgeHandle,
                                                     ObjHandleT,
                                                     blockThreads,
                                                     op,
                                                     ScalarT,
                                                     InputDim,
                                                     VariableDim,
                                                     LambdaT>>(rx, t, oreinted);
            terms.push_back(
                std::dynamic_pointer_cast<VectorTerm<T, ObjHandleT>>(new_term));
        }

        if constexpr (op == Op::FV || op == Op::FE || op == Op::FF ||
                      op == Op::F) {
            auto new_term =
                std::make_shared<TemplatedVectorTerm<FaceHandle,
                                                     ObjHandleT,
                                                     blockThreads,
                                                     op,
                                                     ScalarT,
                                                     InputDim,
                                                     VariableDim,
                                                     LambdaT>>(rx, t, oreinted);
            terms.push_back(
                std::dynamic_pointer_cast<VectorTerm<T, ObjHandleT>>(new_term));
        }

        ops.push_back(op);
        block_shapes.push_back({InputDim, VariableDim});
    }

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

        // populate residual_reshaped
        for (int i = 0; i < jac->get_num_terms(); ++i) {

            auto [st, end] = jac->get_term_rows_range(i);

            residual_reshaped.emplace_back(residual->segment_range(st, end));
        }
    }

    /**
     * @brief evaluate all terms in active mode
     */
    void eval_terms(cudaStream_t stream = NULL)
    {
        if (!is_prep_eval_called) {
            prep_eval();
        }

        jac->reset(0, DEVICE, stream);

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_active(
                *objective, residual_reshaped[i], *jac, stream);

            {
                CUDA_ERROR(cudaGetLastError());
                CUDA_ERROR(cudaDeviceSynchronize());
            }
        }        
    }


    /**
     * @brief evaluate all terms in passive mode
     */
    void eval_terms_passive(Attribute<T, ObjHandleT>* obj    = nullptr,
                            cudaStream_t              stream = NULL)
    {
        if (!is_prep_eval_called) {
            prep_eval();
        }

        for (size_t i = 0; i < terms.size(); ++i) {
            if (obj) {
                terms[i]->eval_passive(*obj, residual_reshaped[i], stream);
            } else {
                terms[i]->eval_passive(
                    *objective, residual_reshaped[i], stream);
            }
        }
    }
};

}  // namespace rxmesh
