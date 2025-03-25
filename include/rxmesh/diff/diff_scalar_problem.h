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
template <typename T, int VariableDim, typename ObjHandleT>
struct DiffScalarProblem
{
    RXMeshStatic&                                     rx;
    DenseMatrix<T, Eigen::RowMajor>                   grad;
    HessianSparseMatrix<T, VariableDim>               hess;
    std::shared_ptr<Attribute<T, ObjHandleT>>         objective;
    std::vector<std::shared_ptr<Term<T, ObjHandleT>>> terms;


    /**
     * @brief
     */
    DiffScalarProblem(RXMeshStatic& rx)
        : rx(rx),
          grad(
              DenseMatrix<T, Eigen::RowMajor>(rx,
                                              rx.get_num_elements<ObjHandleT>(),
                                              VariableDim)),
          hess(HessianSparseMatrix<T, VariableDim>(rx)),
          objective(rx.add_vertex_attribute<T>("objective", VariableDim))

    {
        grad.reset(0, LOCATION_ALL);
        hess.reset(0, LOCATION_ALL);
    }

    /**
     * @brief
     * @tparam LambdaT
     * @tparam op
     * @tparam ProjectHess
     * @tparam blockThreads
     * @param rx
     * @param t
     * @param oreinted
     */
    template <Op       op,
              bool     ProjectHess,
              uint32_t blockThreads = 256,
              typename LambdaT      = void>
    void add_term(RXMeshStatic& rx, LambdaT t, bool oreinted = false)
    {

        constexpr int ElementValence = element_valence<op>();

        constexpr int NElements = VariableDim * ElementValence;

        using ScalarT = Scalar<T, NElements, true>;

        if constexpr (op == Op::VV || op == Op::VE || op == Op::VF) {
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

        if constexpr (op == Op::EV || op == Op::EE || op == Op::EF) {
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

        if constexpr (op == Op::FV || op == Op::FE || op == Op::FF) {
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
     * @brief
     */
    void eval_terms(cudaStream_t stream = NULL)
    {
        grad.reset(0, DEVICE, stream);
        hess.reset(0, DEVICE, stream);

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_active(*objective, stream);
        }
    }


    /**
     * @brief
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
     * @brief
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