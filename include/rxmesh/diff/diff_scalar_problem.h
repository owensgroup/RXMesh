#pragma once

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/element_valence.h"
#include "rxmesh/diff/hessian_sparse_matrix.h"
#include "rxmesh/diff/term.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/types.h"

namespace rxmesh {

/**
 * @brief Definition of differentiation problem
 * @tparam T the underlying (passive) type of the problem, e.g., float or double
 * @tparam ObjHandleT the type of the mesh element with respect to which the
 * differentiation is being performed (e.g., VertexHandle for mesh
 * parametrization)
 * @tparam VariableDim the dimensions of the active variable defined on each
 * mesh element under consideration (e.g., 2 for mesh parametrization)
 */
template <typename T, int VariableDim, typename ObjHandleT, bool WithHess>
struct DiffScalarProblem
{

    // TODO use ObjHandleT to define the Hessian matrix sparsity
    // right now, we always assume VV sparsity pattern but we can derive
    // different sparsity, e.g., FF
    using HessMatT  = HessianSparseMatrix<T, VariableDim>;
    using DenseMatT = DenseMatrix<T, Eigen::RowMajor>;

    static constexpr bool WithHessian = WithHess;


    RXMeshStatic&                                     rx;
    DenseMatT                                         grad;
    HessMatT                                          hess;
    std::shared_ptr<Attribute<T, ObjHandleT>>         objective;
    std::vector<std::shared_ptr<Term<T, ObjHandleT>>> terms;


    DiffScalarProblem(RXMeshStatic& rx, bool assmble_hessian = true)
        : rx(rx),
          grad(DenseMatT(rx, rx.get_num_elements<ObjHandleT>(), VariableDim)),
          objective(rx.add_vertex_attribute<T>("objective", VariableDim))

    {
        grad.reset(0, LOCATION_ALL);

        if constexpr (WithHessian) {
            if (assmble_hessian) {
                hess = HessMatT(rx);
                hess.reset(0, LOCATION_ALL);
            }
        }
    }

    /**
     * @brief add an term to the loss function
     */
    template <Op       op,
              bool     ProjectHess  = false,
              uint32_t blockThreads = 256,
              typename LambdaT      = void>
    void add_term(LambdaT t, bool oreinted = false)
    {

        constexpr int ElementValence = element_valence<op>();

        constexpr int NElements = VariableDim * ElementValence;

        using ScalarT = Scalar<T, NElements, WithHessian>;

        if constexpr (op == Op::VV || op == Op::VE || op == Op::VF ||
                      op == Op::V) {
            auto new_term = std::make_shared<TemplatedTerm<VertexHandle,
                                                           ObjHandleT,
                                                           blockThreads,
                                                           op,
                                                           ScalarT,
                                                           ProjectHess,
                                                           VariableDim,
                                                           LambdaT>>(
                rx, t, oreinted, grad, hess);
            terms.push_back(
                std::dynamic_pointer_cast<Term<T, ObjHandleT>>(new_term));
        }

        if constexpr (op == Op::EV || op == Op::EE || op == Op::EF ||
                      op == Op::E) {
            auto new_term = std::make_shared<TemplatedTerm<EdgeHandle,
                                                           ObjHandleT,
                                                           blockThreads,
                                                           op,
                                                           ScalarT,
                                                           ProjectHess,
                                                           VariableDim,
                                                           LambdaT>>(
                rx, t, oreinted, grad, hess);
            terms.push_back(
                std::dynamic_pointer_cast<Term<T, ObjHandleT>>(new_term));
        }

        if constexpr (op == Op::FV || op == Op::FE || op == Op::FF ||
                      op == Op::F) {
            auto new_term = std::make_shared<TemplatedTerm<FaceHandle,
                                                           ObjHandleT,
                                                           blockThreads,
                                                           op,
                                                           ScalarT,
                                                           ProjectHess,
                                                           VariableDim,
                                                           LambdaT>>(
                rx, t, oreinted, grad, hess);
            terms.push_back(
                std::dynamic_pointer_cast<Term<T, ObjHandleT>>(new_term));
        }
    }

    /**
     * @brief evaluate all terms
     */
    void eval_terms(cudaStream_t stream = NULL)
    {
        grad.reset(0, DEVICE, stream);

        if constexpr (WithHessian) {
            hess.reset(0, DEVICE, stream);
        }

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_active(*objective, stream);
        }
    }


    /**
     * @brief evaluate all terms
     */
    void eval_terms_grad_only(cudaStream_t stream = NULL)
    {
        grad.reset(0, DEVICE, stream);

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_active_grad_only(*objective, stream);
        }
    }

    /**
     * @brief Hessian-vector product
     */
    void eval_matvec(const DenseMatrix<T, Eigen::RowMajor>& input,
                     DenseMatrix<T, Eigen::RowMajor>&       output,
                     cudaStream_t                           stream = NULL)
    {
        output.reset(0, DEVICE, stream);

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_active_matvec(*objective, input, output, stream);
        }
    }

    /**
     * @brief return the current loss/energy
     */
    T get_current_loss(cudaStream_t stream = NULL)
    {
        T sum = 0;

        for (size_t i = 0; i < terms.size(); ++i) {
            sum += terms[i]->get_loss(stream);
        }
        return sum;
    }


    /**
     * @brief evaluate all terms in
     */
    void eval_terms_passive(Attribute<T, ObjHandleT>* obj    = nullptr,
                            cudaStream_t              stream = NULL)
    {
        for (size_t i = 0; i < terms.size(); ++i) {
            if (obj) {
                terms[i]->eval_passive(*obj, stream);
            } else {
                terms[i]->eval_passive(*objective, stream);
            }
        }
    }
};

}  // namespace rxmesh