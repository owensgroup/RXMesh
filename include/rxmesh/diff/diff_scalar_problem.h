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

    using IndexT = typename HessMatT::IndexT;

    static constexpr bool WithHessian = WithHess;


    RXMeshStatic&                                     rx;
    DenseMatT                                         grad;
    std::unique_ptr<HessMatT>                         hess;
    std::unique_ptr<HessMatT>                         hess_new;
    std::shared_ptr<Attribute<T, ObjHandleT>>         objective;
    std::vector<std::shared_ptr<Term<T, ObjHandleT>>> terms;


    /**
     * @brief Constructor
     * @param rx is the instance of RXMeshStatic
     * @param assmble_hessian should allocate the Hessian
     * @param capacity_factor we allow the Hessian to change its sparsity which
     * might increase the number of NNZ. We use the number of calculate the max
     * nnz as capactiy_factor*nnz0 where nnz0 is the nnz of the hessian from the
     * topology of the mesh
     */
    DiffScalarProblem(RXMeshStatic& rx,
                      bool          assmble_hessian = true,
                      const float   capacity_factor = 1.0f)
        : rx(rx),
          grad(DenseMatT(rx, rx.get_num_elements<ObjHandleT>(), VariableDim)),
          objective(rx.add_vertex_attribute<T>("objective", VariableDim))

    {
        grad.reset(0, LOCATION_ALL);

        if constexpr (WithHessian) {
            if (assmble_hessian) {
                hess = std::make_unique<HessMatT>(rx, capacity_factor);
                hess->reset(0, LOCATION_ALL);

                hess_new = std::make_unique<HessMatT>(rx, capacity_factor);
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

        constexpr int NElements =
            std::max(VariableDim * ElementValence, Eigen::Dynamic);

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
                rx, t, oreinted, grad, *hess);
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
                rx, t, oreinted, grad, *hess);
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
                rx, t, oreinted, grad, *hess);
            terms.push_back(
                std::dynamic_pointer_cast<Term<T, ObjHandleT>>(new_term));
        }
    }


    void update_hessian(const IndexT  size,
                        const IndexT* d_new_rows,
                        const IndexT* d_new_cols)
    {
        hess_new->insert(rx, *hess, size, d_new_rows, d_new_cols);
        hess_new.swap(hess);        
    }

    /**
     * @brief evaluate all terms
     */
    void eval_terms(cudaStream_t stream = NULL)
    {
        grad.reset(0, DEVICE, stream);

        if constexpr (WithHessian) {
            hess->reset(0, DEVICE, stream);
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

    /**
     * @brief evaluate all terms
     */
    void eval_terms_grad_only(Attribute<T, ObjHandleT>* obj,
                              cudaStream_t              stream = NULL)
    {
        grad.reset(0, DEVICE, stream);

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_active_grad_only(*obj, stream);
        }
    }
};

}  // namespace rxmesh