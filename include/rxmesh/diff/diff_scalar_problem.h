#pragma once

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/candidate_pairs.h"
#include "rxmesh/diff/element_valence.h"
#include "rxmesh/diff/ev_diamond_interaction.h"
#include "rxmesh/diff/hessian_sparse_matrix.h"
#include "rxmesh/diff/scalar_term.h"
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

    bool ev_diamond_interaction_added = false;

    RXMeshStatic&                                           rx;
    DenseMatT                                               grad;
    std::unique_ptr<HessMatT>                               hess;
    std::unique_ptr<HessMatT>                               hess_new;
    std::shared_ptr<Attribute<T, ObjHandleT>>               objective;
    std::vector<std::shared_ptr<ScalarTerm<T, ObjHandleT>>> terms;

    // TODO we might need other types of candidate pairs
    CandidatePairsVV<HessMatT> vv_pairs;
    CandidatePairsVF<HessMatT> vf_pairs;


    /**
     * @brief Constructor
     * @param rx is the instance of RXMeshStatic
     * @param assmble_hessian should allocate the Hessian
     */
    DiffScalarProblem(RXMeshStatic& rx,
                      bool          assmble_hessian,
                      int           expected_vv_candidate_pairs = 0,
                      int           expected_vf_candidate_pairs = 0)
        : rx(rx),
          ev_diamond_interaction_added(false),
          grad(DenseMatT(rx,
                         rx.get_num_elements<ObjHandleT>(),
                         VariableDim,
                         LOCATION_ALL)),
          objective(rx.add_attribute<T, ObjHandleT>("objective", VariableDim))
    {
        grad.reset(0, LOCATION_ALL);

        if constexpr (WithHessian) {
            if (assmble_hessian) {

                // every VV interaction pairs will add a 2 (because of
                // symmetry) blocks of (VariableDim x VariableDim) into the
                // Hessian
                // every VF interaction pairs will add 3 (because of three
                // triangles vertices) x 2 (because of symmetry) blocks of
                // (VariableDim x VariableDim)
                int expect_new_entries_in_hess =
                    expected_vv_candidate_pairs * VariableDim * VariableDim *
                        2 +
                    expected_vf_candidate_pairs * VariableDim * VariableDim *
                        2 * 3;

                hess =
                    std::make_unique<HessMatT>(rx, expect_new_entries_in_hess);
                hess->reset(0, LOCATION_ALL);

                hess_new =
                    std::make_unique<HessMatT>(rx, expect_new_entries_in_hess);

                vv_pairs = CandidatePairsVV<HessMatT>(
                    expected_vv_candidate_pairs, *hess, rx.get_context());

                vf_pairs = CandidatePairsVF<HessMatT>(
                    expected_vf_candidate_pairs, *hess, rx.get_context());

            } else {
                hess = std::make_unique<HessMatT>();
            }
        } else {
            hess = std::make_unique<HessMatT>();
        }
    }

    /**
     * @brief add a (energy) term to the loss function that depends on local
     * query operation (e.g., FV)
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
            auto new_term = std::make_shared<TemplatedScalarTerm<VertexHandle,
                                                                 ObjHandleT,
                                                                 blockThreads,
                                                                 op,
                                                                 ScalarT,
                                                                 ProjectHess,
                                                                 VariableDim,
                                                                 LambdaT>>(
                rx, t, oreinted, &grad, hess.get());
            terms.push_back(
                std::dynamic_pointer_cast<ScalarTerm<T, ObjHandleT>>(new_term));
        }

        if constexpr (op == Op::EV || op == Op::EE || op == Op::EF ||
                      op == Op::E || op == Op::EVDiamond) {
            auto new_term = std::make_shared<TemplatedScalarTerm<EdgeHandle,
                                                                 ObjHandleT,
                                                                 blockThreads,
                                                                 op,
                                                                 ScalarT,
                                                                 ProjectHess,
                                                                 VariableDim,
                                                                 LambdaT>>(
                rx, t, oreinted, &grad, hess.get());
            terms.push_back(
                std::dynamic_pointer_cast<ScalarTerm<T, ObjHandleT>>(new_term));
            if (op == Op::EVDiamond && WithHess && hess) {
                if (!ev_diamond_interaction_added) {
                    detail::add_ev_diamond_interaction(rx, *this);
                    ev_diamond_interaction_added = true;
                }
            }
        }

        if constexpr (op == Op::FV || op == Op::FE || op == Op::FF ||
                      op == Op::F) {
            auto new_term = std::make_shared<TemplatedScalarTerm<FaceHandle,
                                                                 ObjHandleT,
                                                                 blockThreads,
                                                                 op,
                                                                 ScalarT,
                                                                 ProjectHess,
                                                                 VariableDim,
                                                                 LambdaT>>(
                rx, t, oreinted, &grad, hess.get());
            terms.push_back(
                std::dynamic_pointer_cast<ScalarTerm<T, ObjHandleT>>(new_term));
        }
    }


    /**
     * @brief add a (energy) term to the loss function that acts on candidate
     * pairs
     * TODO generalize this to other type of candidate pairs. For now, we assume
     * only VV or VF pairs
     */
    template <Op       op,
              bool     ProjectHess  = false,
              uint32_t blockThreads = 256,
              typename LambdaT      = void>
    void add_interaction_term(LambdaT t)
    {
        if constexpr (op == Op::VV) {

            // for VV interaction, the element valence is 2 because there are
            // 2 vertices involved in each interaction
            constexpr int ElementValence = 2;

            constexpr int NElements = VariableDim * ElementValence;

            using ScalarT = Scalar<T, NElements, WithHessian>;

            auto new_term =
                std::make_shared<TemplatedScalarTermPairs<VertexHandle,
                                                          ObjHandleT,
                                                          blockThreads,
                                                          VertexHandle,
                                                          VertexHandle,
                                                          HessMatT,
                                                          ScalarT,
                                                          ProjectHess,
                                                          VariableDim,
                                                          LambdaT>>(
                    rx, t, &grad, hess.get(), vv_pairs);

            terms.push_back(
                std::dynamic_pointer_cast<ScalarTerm<T, ObjHandleT>>(new_term));
        }

        if constexpr (op == Op::VF) {
            // for VF interaction, the element valence is 4 because there are
            // 4 vertices involved in each interaction, i.e., face's three
            // vertices and the other vertex
            constexpr int ElementValence = 4;

            constexpr int NElements = VariableDim * ElementValence;

            using ScalarT = Scalar<T, NElements, WithHessian>;

            auto new_term =
                std::make_shared<TemplatedScalarTermPairs<VertexHandle,
                                                          ObjHandleT,
                                                          blockThreads,
                                                          VertexHandle,
                                                          FaceHandle,
                                                          HessMatT,
                                                          ScalarT,
                                                          ProjectHess,
                                                          VariableDim,
                                                          LambdaT>>(
                    rx, t, &grad, hess.get(), vf_pairs);

            terms.push_back(
                std::dynamic_pointer_cast<ScalarTerm<T, ObjHandleT>>(new_term));
        }
    }

    /**
     * @brief update the sparse Hessian after adding contact
     */
    void update_hessian()
    {
        if (!hess) {
            return;
        }
        // TODO expand the indices for VF interactions

        if (hess_new->insert(rx,
                             *hess,
                             vv_pairs.num_index(),
                             vv_pairs.m_pairs_id.col_data(0),
                             vv_pairs.m_pairs_id.col_data(1))) {
            hess_new->swap(*hess);

            vv_pairs.m_hess = *hess;
        }
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
            T l = terms[i]->get_loss(stream);
            sum += l;
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

    /**
     * @brief add the interaction between the two opposite vertices of edge
     * diamond.
     */
};

}  // namespace rxmesh