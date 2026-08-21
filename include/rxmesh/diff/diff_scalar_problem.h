#pragma once

#include <stdexcept>
#include <string>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/candidate_pairs.h"
#include "rxmesh/diff/element_valence.h"
#include "rxmesh/diff/hessian_sparse_matrix.h"
#include "rxmesh/diff/interaction_impl.h"
#include "rxmesh/diff/scalar_term.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/types.h"

namespace rxmesh {

enum class OptVarStorage
{
    Owned,
    MetadataOnly,
    Absent
};

/**
 * @brief Storage policy for DiffScalarProblem's internal buffers.
 */
struct DiffProblemMemoryOptions
{
    locationT     gradient_location     = LOCATION_ALL;
    OptVarStorage opt_var_storage       = OptVarStorage::Owned;
    locationT     opt_var_location      = LOCATION_ALL;
    layoutT       opt_var_layout        = AoSoA;
    locationT     term_loss_location    = LOCATION_ALL;
    bool          unique_internal_names = false;
};


/**
 * @brief Definition of differentiation problem
 * @tparam T the underlying (passive) type of the problem, e.g., float or double
 * @tparam OptVarHandleT the type of the mesh element with respect to which the
 * differentiation is being performed (e.g., VertexHandle for mesh
 * parametrization)
 * @tparam VariableDim the dimensions of the active variable defined on each
 * mesh element under consideration (e.g., 2 for mesh parametrization)
 */
template <typename T, int VariableDim, typename OptVarHandleT, bool WithHess>
struct DiffScalarProblem
{
    // TODO use OptVarHandleT to define the Hessian matrix sparsity
    // right now, we always assume VV sparsity pattern but we can derive
    // different sparsity, e.g., FF
    using HessMatT  = HessianSparseMatrix<T, VariableDim>;
    using DenseMatT = DenseMatrix<T, Eigen::RowMajor>;

    using IndexT = typename HessMatT::IndexT;

    static constexpr bool WithHessian = WithHess;

    bool ev_diamond_interaction_added = false;

    RXMeshStatic&                                              rx;
    DiffProblemMemoryOptions                                   memory_options;
    DenseMatT                                                  grad;
    std::unique_ptr<HessMatT>                                  hess;
    std::unique_ptr<HessMatT>                                  hess_new;
    std::shared_ptr<Attribute<T, OptVarHandleT>>               opt_var;
    std::vector<std::shared_ptr<ScalarTerm<T, OptVarHandleT>>> terms;
    std::shared_ptr<FaceAttribute<VertexHandle>> face_interact_vertex;
    detail::RegisteredAttributeOwner             opt_var_registration;
    detail::RegisteredAttributeOwner             face_interact_registration;

    // TODO we might need other types of candidate pairs
    CandidatePairsVV<HessMatT> vv_pairs;
    CandidatePairsVF<HessMatT> vf_pairs;


    /**
     * @brief Constructor
     * @param rx is the instance of RXMeshStatic
     * @param assmble_hessian should allocate the Hessian
     */
    DiffScalarProblem(RXMeshStatic&      rx,
                      bool               assmble_hessian,
                      int                expected_vv_candidate_pairs = 0,
                      int                expected_vf_candidate_pairs = 0,
                      const std::string& opt_var_name         = "opt_var",
                      const DiffProblemMemoryOptions& options = {})
        : ev_diamond_interaction_added(false),
          rx(rx),
          memory_options(options),
          grad(DenseMatT(rx,
                         rx.get_num_elements<OptVarHandleT>(),
                         VariableDim,
                         options.gradient_location))
    {
        if (options.gradient_location != LOCATION_NONE) {
            grad.reset(T(0), options.gradient_location);
        }

        if (options.opt_var_storage != OptVarStorage::Absent) {
            const std::string internal_name =
                options.unique_internal_names ?
                    rx.make_unique_attribute_name(opt_var_name + ":rx:diff:") :
                    opt_var_name;
            const locationT opt_var_location =
                options.opt_var_storage == OptVarStorage::MetadataOnly ?
                    LOCATION_NONE :
                    options.opt_var_location;
            opt_var =
                rx.add_attribute<T, OptVarHandleT>(internal_name,
                                                   VariableDim,
                                                   opt_var_location,
                                                   options.opt_var_layout);
            opt_var_registration.bind(rx, opt_var.get());
            if (opt_var_location != LOCATION_NONE) {
                opt_var->reset(T(0), opt_var_location);
            }
        }

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
                    expected_vv_candidate_pairs +
                        3 * expected_vf_candidate_pairs,  // since we use
                                                          // vv_pairs to stage
                                                          // VF interaction as
                                                          // well
                    *hess,
                    rx.get_context());

                vf_pairs = CandidatePairsVF<HessMatT>(
                    expected_vf_candidate_pairs, *hess, rx.get_context());

                if (expected_vf_candidate_pairs > 0) {
                    const std::string interaction_name =
                        options.unique_internal_names ?
                            rx.make_unique_attribute_name(
                                "rx:diff:face_interaction:") :
                            "rx:FInteractV";
                    face_interact_vertex = rx.add_face_attribute<VertexHandle>(
                        interaction_name, 1);
                    face_interact_registration.bind(rx,
                                                    face_interact_vertex.get());
                }

            } else {
                hess = std::make_unique<HessMatT>();
            }
        } else {
            hess = std::make_unique<HessMatT>();
        }
    }

    /**
     * @brief Convenience overload placing the memory policy before optional
     * naming/candidate-pair parameters.
     */
    DiffScalarProblem(RXMeshStatic&                   rx,
                      bool                            assmble_hessian,
                      const DiffProblemMemoryOptions& options,
                      const std::string&              opt_var_name = "opt_var",
                      int expected_vv_candidate_pairs              = 0,
                      int expected_vf_candidate_pairs              = 0)
        : DiffScalarProblem(rx,
                            assmble_hessian,
                            expected_vv_candidate_pairs,
                            expected_vf_candidate_pairs,
                            opt_var_name,
                            options)
    {
    }

    ~DiffScalarProblem()
    {
        // Terms own registered loss Attributes and must disappear before the
        // problem releases the buffers their kernels reference.
        terms.clear();
        face_interact_registration.reset();
        face_interact_vertex.reset();
        opt_var_registration.reset();
        opt_var.reset();
        grad.release();
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
                                                                 OptVarHandleT,
                                                                 blockThreads,
                                                                 op,
                                                                 ScalarT,
                                                                 ProjectHess,
                                                                 VariableDim,
                                                                 LambdaT>>(
                rx,
                t,
                oreinted,
                &grad,
                hess.get(),
                memory_options.term_loss_location);
            terms.push_back(
                std::dynamic_pointer_cast<ScalarTerm<T, OptVarHandleT>>(
                    new_term));
        }

        if constexpr (op == Op::EV || op == Op::EE || op == Op::EF ||
                      op == Op::E || op == Op::EVDiamond) {
            auto new_term = std::make_shared<TemplatedScalarTerm<EdgeHandle,
                                                                 OptVarHandleT,
                                                                 blockThreads,
                                                                 op,
                                                                 ScalarT,
                                                                 ProjectHess,
                                                                 VariableDim,
                                                                 LambdaT>>(
                rx,
                t,
                oreinted,
                &grad,
                hess.get(),
                memory_options.term_loss_location);
            terms.push_back(
                std::dynamic_pointer_cast<ScalarTerm<T, OptVarHandleT>>(
                    new_term));
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
                                                                 OptVarHandleT,
                                                                 blockThreads,
                                                                 op,
                                                                 ScalarT,
                                                                 ProjectHess,
                                                                 VariableDim,
                                                                 LambdaT>>(
                rx,
                t,
                oreinted,
                &grad,
                hess.get(),
                memory_options.term_loss_location);
            terms.push_back(
                std::dynamic_pointer_cast<ScalarTerm<T, OptVarHandleT>>(
                    new_term));
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
                                                          OptVarHandleT,
                                                          blockThreads,
                                                          VertexHandle,
                                                          VertexHandle,
                                                          HessMatT,
                                                          ScalarT,
                                                          ProjectHess,
                                                          VariableDim,
                                                          LambdaT>>(
                    rx,
                    t,
                    &grad,
                    hess.get(),
                    vv_pairs,
                    memory_options.term_loss_location);

            terms.push_back(
                std::dynamic_pointer_cast<ScalarTerm<T, OptVarHandleT>>(
                    new_term));
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
                                                          OptVarHandleT,
                                                          blockThreads,
                                                          VertexHandle,
                                                          FaceHandle,
                                                          HessMatT,
                                                          ScalarT,
                                                          ProjectHess,
                                                          VariableDim,
                                                          LambdaT>>(
                    rx,
                    t,
                    &grad,
                    hess.get(),
                    vf_pairs,
                    memory_options.term_loss_location);

            terms.push_back(
                std::dynamic_pointer_cast<ScalarTerm<T, OptVarHandleT>>(
                    new_term));
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
        //  record the current vv_pair size and then reset it against to
        // this size. We do this because add_vf_pairs_to_vv_pairs stages the new
        // VV pairs (due to FV interaction) in vv_pair which would mess with the
        // user-defined vv_pairs later when they start using it to add energies

        int vv_prv_num_index = vv_pairs.num_index();
        int vv_prv_num_pairs = vv_pairs.num_pairs();

        // expand the indices for VF interactions
        if (face_interact_vertex) {
            detail::add_vf_pairs_to_vv_pairs(
                rx, *this, vf_pairs, vv_pairs, *face_interact_vertex);
        }

        if (hess_new->insert(rx,
                             *hess,
                             vv_pairs.num_index(),
                             vv_pairs.m_pairs_id.col_data(0),
                             vv_pairs.m_pairs_id.col_data(1))) {
            hess_new->swap(*hess);

            vv_pairs.m_hess = *hess;
        }

        if (face_interact_vertex) {
            vv_pairs.reset(vv_prv_num_pairs, vv_prv_num_index);
        }
#ifndef NDEBUG
        // hess->check_repeated_indices();
        // hess_new->check_repeated_indices();
#endif
    }

    /**
     * @brief evaluate all terms
     */
    void eval_terms(cudaStream_t stream = NULL)
    {
        if (!require_owned_opt_var("eval_terms") ||
            !require_internal_gradient("eval_terms")) {
            return;
        }
        grad.reset(0, DEVICE, stream);

        if constexpr (WithHessian) {
            hess->reset(0, DEVICE, stream);
        }

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_active(*opt_var, face_interact_vertex.get(), stream);
        }
    }


    /**
     * @brief evaluate all terms
     */
    void eval_terms_grad_only(cudaStream_t stream = NULL)
    {
        if (!require_owned_opt_var("eval_terms_grad_only")) {
            return;
        }
        eval_terms_grad_only(opt_var.get(), stream);
    }

    /**
     * @brief Hessian-vector product
     */
    void eval_matvec(const DenseMatrix<T, Eigen::RowMajor>& input,
                     DenseMatrix<T, Eigen::RowMajor>&       output,
                     cudaStream_t                           stream = NULL)
    {
        if (!require_owned_opt_var("eval_matvec")) {
            return;
        }
        output.reset(0, DEVICE, stream);

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_active_matvec(*opt_var, input, output, stream);
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
     * @brief Number of independently reduced scalar terms.
     */
    size_t get_num_terms() const
    {
        return terms.size();
    }

    /**
     * @brief Enqueue one device loss reduction per term. No device-to-host copy
     * or stream synchronization is done.
     */
    void get_current_loss_device(T*           device_term_losses,
                                 size_t       output_count,
                                 cudaStream_t stream = NULL)
    {
        if (output_count < terms.size()) {
            RXMESH_ERROR(
                "DiffScalarProblem::get_current_loss_device() received {} "
                "outputs for {} terms.",
                output_count,
                terms.size());
            return;
        }
        if (!terms.empty() && device_term_losses == nullptr) {
            RXMESH_ERROR(
                "DiffScalarProblem::get_current_loss_device() received a "
                "null output pointer.");
            return;
        }

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->get_loss_device(device_term_losses + i, stream);
        }
    }


    /**
     * @brief evaluate all terms in
     */
    void eval_terms_passive(Attribute<T, OptVarHandleT>* opt_var_in = nullptr,
                            cudaStream_t                 stream     = NULL)
    {
        Attribute<T, OptVarHandleT>* effective_opt_var = opt_var_in;
        if (effective_opt_var == nullptr) {
            if (!require_owned_opt_var("eval_terms_passive")) {
                return;
            }
            effective_opt_var = opt_var.get();
        } else if (!validate_device_opt_var(effective_opt_var,
                                            "eval_terms_passive")) {
            return;
        }

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_passive(
                *effective_opt_var, face_interact_vertex.get(), stream);
        }
    }

    /**
     * @brief evaluate all terms
     */
    void eval_terms_grad_only(Attribute<T, OptVarHandleT>* opt_var_in,
                              cudaStream_t                 stream = NULL)
    {
        if (!require_internal_gradient("eval_terms_grad_only")) {
            return;
        }
        eval_terms_grad_only(opt_var_in, grad, stream);
    }

    /**
     * @brief Evaluate the gradient only path into (row-major) device storage
     provided by the caller
     */
    void eval_terms_grad_only(Attribute<T, OptVarHandleT>* opt_var_in,
                              DenseMatT&                   gradient_output,
                              cudaStream_t                 stream = NULL)
    {
        const IndexT expected_rows =
            static_cast<IndexT>(rx.get_num_elements<OptVarHandleT>());
        if (!validate_device_opt_var(opt_var_in, "eval_terms_grad_only") ||
            gradient_output.rows() != expected_rows ||
            gradient_output.cols() != VariableDim ||
            !gradient_output.template has_compatible_context<OptVarHandleT>(
                rx)) {
            RXMESH_ERROR(
                "DiffScalarProblem::eval_terms_grad_only() received an "
                "incompatible input or gradient output.");
            return;
        }

        const bool output_empty = expected_rows == 0 || VariableDim == 0;
        if (output_empty) {
            return;
        }
        if (((gradient_output.get_allocated() & DEVICE) != DEVICE) ||
            gradient_output.data(DEVICE) == nullptr) {
            RXMESH_ERROR(
                "DiffScalarProblem::eval_terms_grad_only() gradient output "
                "must be allocated on the device.");
            return;
        }

        gradient_output.reset(0, DEVICE, stream);

        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]->eval_active_grad_only(*opt_var_in,
                                            face_interact_vertex.get(),
                                            gradient_output,
                                            stream);
        }
    }

    /**
     * @brief True only for the historical internally owned opt-var mode.
     */
    bool has_owned_opt_var() const
    {
        return memory_options.opt_var_storage == OptVarStorage::Owned &&
               opt_var != nullptr;
    }

    bool validate_device_opt_var(const Attribute<T, OptVarHandleT>* candidate,
                                 const char* operation) const
    {
        const size_t expected_rows = rx.get_num_elements<OptVarHandleT>();
        if (candidate == nullptr || candidate->rows() != expected_rows ||
            candidate->cols() != VariableDim) {
            RXMESH_ERROR(
                "DiffScalarProblem::{}() requires an optimization Attribute "
                "with shape ({}, {}).",
                operation,
                expected_rows,
                VariableDim);
            return false;
        }
        if (expected_rows != 0 && (!candidate->is_device_allocated() ||
                                   candidate->data(DEVICE) == nullptr)) {
            RXMESH_ERROR(
                "DiffScalarProblem::{}() requires device-backed optimization "
                "storage.",
                operation);
            return false;
        }
        return true;
    }

    bool require_owned_opt_var(const char* operation) const
    {
        if (!has_owned_opt_var()) {
            const char* mode =
                memory_options.opt_var_storage == OptVarStorage::MetadataOnly ?
                    "MetadataOnly" :
                    "Absent";
            const std::string message =
                "DiffScalarProblem::" + std::string(operation) +
                "() requires OptVarStorage::Owned, but this problem uses " +
                mode +
                ". Use an overload with an explicit optimization Attribute.";
            RXMESH_ERROR("{}", message);
            throw std::invalid_argument(message);
        }
        return validate_device_opt_var(opt_var.get(), operation);
    }

    bool require_internal_gradient(const char* operation) const
    {
        const size_t rows = rx.get_num_elements<OptVarHandleT>();
        if (rows != 0 && (((grad.get_allocated() & DEVICE) != DEVICE) ||
                          grad.data(DEVICE) == nullptr)) {
            const std::string message =
                "DiffScalarProblem::" + std::string(operation) +
                "() requires a device-backed internal gradient. Configure "
                "gradient_location with DEVICE or use the caller-output "
                "gradient overload.";
            RXMESH_ERROR("{}", message);
            throw std::invalid_argument(message);
        }
        return true;
    }

    /**
     * @brief add the interaction between the two opposite vertices of edge
     * diamond.
     */
};

}  // namespace rxmesh
