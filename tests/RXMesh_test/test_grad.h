#pragma once

#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/gradient_descent.h"
#include "rxmesh/diff/newton_solver.h"

#include <vector>


using namespace rxmesh;

template <typename ProblemT>
inline void add_smoothing_term(ProblemT& problem)
{
    problem.template add_term<Op::EV>([=] __device__(const auto& eh,
                                                     const auto& iter,
                                                     auto&       opt_var) {
        assert(iter.size() == 2);

        using ActiveT = ACTIVE_TYPE(eh);

        // pos
        Eigen::Vector3<ActiveT> d0 = opt_var.template active<3>(eh, iter, 0);
        Eigen::Vector3<ActiveT> d1 = opt_var.template active<3>(eh, iter, 1);

        Eigen::Vector3<ActiveT> dist = (d0 - d1);

        ActiveT dist_sq = dist.squaredNorm();

        return dist_sq;
    });
}

TEST(Diff, SmoothingNewton)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");
    // RXMeshStatic rx(rxmesh_args.obj_file_name);

    using T = float;

    constexpr int VariableDim = 3;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    ProblemT problem(rx, true);

    auto v_input_pos = *rx.get_input_vertex_coordinates();

    problem.opt_var->copy_from(v_input_pos, DEVICE, DEVICE);

    add_smoothing_term(problem);


    using HessMatT = typename ProblemT::HessMatT;

    LUSolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(problem.hess.get());

    NewtonSolver newton(problem, &solver);

    int num_iterations = 100;

    T convergence_eps = 1e-2;

    GPUTimer timer;
    timer.start();

    for (int iter = 0; iter < num_iterations; ++iter) {

        problem.eval_terms();


        float energy = problem.get_current_loss();

        RXMESH_INFO("Iteration = {}: Energy = {}", iter, energy);


        newton.compute_direction();

        RXMESH_INFO("newton.dir.norm2() = {}", newton.dir.norm2());
        RXMESH_INFO("problem.grad.norm2() = {}", problem.grad.norm2());

        if (0.5f * problem.grad.dot(newton.dir) < convergence_eps) {
            break;
        }

        newton.line_search();
    }
    timer.stop();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    RXMESH_INFO("Smoothing Newton RXMesh took {} ms, {} ms/iteration",
                timer.elapsed_millis(),
                timer.elapsed_millis() / float(num_iterations));


    // so newton method on this function should lead to a vertex position that
    // is just zero since the function is quadratic

    problem.opt_var->move(DEVICE, HOST);

    T f = (*problem.opt_var)(VertexHandle(0, 0), 0);

    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR((*problem.opt_var)(vh, 0), f, 1e-3);
        }
    });
}

template <typename VAttr>
void copy_x(RXMeshStatic& rx, const VAttr& pos, VAttr& val)
{
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        // val(vh, 0) = fabs(pos(vh, 0));
        val(vh, 0) = pos(vh, 0) * pos(vh, 0);
    });
}

template <typename VAttr, typename DenseMatT>
void verify_while_loop_x(RXMeshStatic&    rx,
                         const VAttr&     pos,
                         const VAttr&     opt_var,
                         const DenseMatT& grad,
                         float            tol)
{
    rx.for_each_vertex(HOST, [=](const VertexHandle& vh) {
        float expected_sqrt = fabs(pos(vh, 0));
        if (expected_sqrt > tol) {

            float expected_grad = 1.0f / (2.0f * expected_sqrt);

            ASSERT_NEAR(opt_var(vh, 0), expected_sqrt, tol);
            ASSERT_NEAR(grad(vh, 0), expected_grad, tol);
        }
    });
}


template <typename ProblemT>
inline void add_while_loop_term(ProblemT& problem, float tol)
{

    problem.template add_term<Op::V>(
        [=] __device__(const auto& vh, auto& opt_var) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            tol = tol;

            Eigen::Vector<ActiveT, 1> xx = opt_var.template active<1>(vh);

            ActiveT a = xx(0);

            if constexpr (is_scalar_v<ActiveT>) {

                // x_new = 0.5 * (x + a / x)
                if (a.val() > tol) {
                    do {
                        xx(0) = 0.5 * (xx(0) + a / xx(0));
                    } while (fabs(xx(0).val() * xx(0).val() - a.val()) > tol);
                } else {
                    xx(0) = ActiveT(0.0);
                }

                // hijacking the opt_var for storing the sqrt value.
                opt_var(vh, 0) = xx(0).val();
            }


            return xx(0);
        });
}

TEST(Diff, WhileLoop)
{

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere1.obj");
    // RXMeshStatic rx(rxmesh_args.obj_file_name);

    using T = float;

    constexpr int VariableDim = 1;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, false>;

    ProblemT problem(rx, false);

    auto v_input_pos = *rx.get_input_vertex_coordinates();

    copy_x(rx, v_input_pos, *problem.opt_var);

    T tol = std::numeric_limits<T>::epsilon();

    add_while_loop_term(problem, tol);


    problem.eval_terms();

    problem.opt_var->move(DEVICE, HOST);
    problem.grad.move(DEVICE, HOST);

    verify_while_loop_x(rx, v_input_pos, *problem.opt_var, problem.grad, 0.001);
}

TEST(Diff, UserGradientAndDeviceLoss)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere1.obj");
    using T                   = rx_coord_t;
    constexpr int VariableDim = 3;
    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, false>;

    DiffProblemMemoryOptions options;
    options.gradient_location     = LOCATION_NONE;
    options.unique_internal_names = true;
    ProblemT problem(rx, false, options);
        
    const int rows = static_cast<int>(rx.get_num_vertices());
    EXPECT_EQ(problem.grad.rows(), rows);
    EXPECT_EQ(problem.grad.cols(), VariableDim);
    EXPECT_EQ(problem.grad.get_allocated(), LOCATION_NONE);
    EXPECT_EQ(problem.grad.data(HOST), nullptr);
    EXPECT_EQ(problem.grad.data(DEVICE), nullptr);
    EXPECT_FALSE(problem.grad.has_library_resources());

    auto input_position = rx.get_input_vertex_coordinates();
    problem.opt_var->copy_from(*input_position, DEVICE, DEVICE);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    add_smoothing_term(problem);
    ASSERT_EQ(problem.get_num_terms(), 1u);

    const size_t gradient_bytes = size_t(rows) * VariableDim * sizeof(T);
    T*           gradient_ptr   = nullptr;
    T*           loss_ptr       = nullptr;
    CUDA_ERROR(cudaMalloc(&gradient_ptr, gradient_bytes));
    CUDA_ERROR(cudaMalloc(&loss_ptr, sizeof(T)));
    auto gradient_output =
        ProblemT::DenseMatT::device_view(rx, rows, VariableDim, gradient_ptr);

    cudaStream_t stream = nullptr;
    CUDA_ERROR(cudaStreamCreate(&stream));
    problem.eval_terms_grad_only(
        problem.opt_var.get(), gradient_output, stream);
    problem.get_current_loss_device(loss_ptr, 1, stream);

    std::vector<T> direct_gradient(size_t(rows) * VariableDim);
    T              direct_loss = 0;
    CUDA_ERROR(cudaMemcpyAsync(direct_gradient.data(),
                               gradient_ptr,
                               gradient_bytes,
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_ERROR(cudaMemcpyAsync(
        &direct_loss, loss_ptr, sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_ERROR(cudaStreamSynchronize(stream));

    // The caller output overload does not allocate or rebind problem.grad.
    EXPECT_EQ(problem.grad.get_allocated(), LOCATION_NONE);
    EXPECT_EQ(problem.grad.data(HOST), nullptr);
    EXPECT_EQ(problem.grad.data(DEVICE), nullptr);

    ProblemT reference_problem(rx, false);
    reference_problem.opt_var->copy_from(*input_position, DEVICE, DEVICE);
    add_smoothing_term(reference_problem);
    reference_problem.eval_terms_grad_only();
    const T reference_loss = reference_problem.get_current_loss();
    reference_problem.grad.move(DEVICE, HOST);
    EXPECT_NEAR(direct_loss, reference_loss, T(1e-5));
    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        const size_t row = rx.linear_id(vh);
        for (int c = 0; c < VariableDim; ++c) {
            EXPECT_NEAR(direct_gradient[row * VariableDim + c],
                        reference_problem.grad(vh, c),
                        T(1e-5));
        }
    });

    // The central internal output overload fails before touching a null
    // gradient and explains both supported remedies.
    try {
        problem.eval_terms_grad_only(problem.opt_var.get());
        FAIL() << "Expected internal-output evaluation to throw";
    } catch (const std::invalid_argument& error) {
        const std::string message(error.what());
        EXPECT_NE(message.find("eval_terms_grad_only"), std::string::npos);
        EXPECT_NE(message.find("gradient_location with DEVICE"),
                  std::string::npos);
        EXPECT_NE(message.find("caller-output gradient overload"),
                  std::string::npos);
    }
    EXPECT_EQ(problem.grad.get_allocated(), LOCATION_NONE);
    EXPECT_EQ(problem.grad.data(HOST), nullptr);
    EXPECT_EQ(problem.grad.data(DEVICE), nullptr);

    gradient_output.release();
    CUDA_ERROR(cudaStreamDestroy(stream));
    GPU_FREE(gradient_ptr);
    GPU_FREE(loss_ptr);
}

TEST(Diff, ConfigurableStorage)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere1.obj");
    using T        = rx_coord_t;
    using ProblemT = DiffScalarProblem<T, 3, VertexHandle, false>;

    const auto baseline_names = rx.get_attribute_names();

    {
        ProblemT problem(rx, false);
        ASSERT_NE(problem.opt_var, nullptr);
        EXPECT_TRUE(problem.has_owned_opt_var());
        EXPECT_EQ(problem.grad.get_allocated() & HOST, HOST);
        EXPECT_EQ(problem.grad.get_allocated() & DEVICE, DEVICE);
        EXPECT_EQ(problem.opt_var->get_allocated() & HOST, HOST);
        EXPECT_EQ(problem.opt_var->get_allocated() & DEVICE, DEVICE);
        EXPECT_EQ(problem.opt_var->get_layout(), AoSoA);

        add_smoothing_term(problem);
        ASSERT_EQ(problem.terms.size(), 1u);
        EXPECT_EQ(problem.terms[0]->get_loss_allocated() & HOST, HOST);
        EXPECT_EQ(problem.terms[0]->get_loss_allocated() & DEVICE, DEVICE);
        EXPECT_EQ(rx.get_attribute_names().size(), baseline_names.size() + 2);
    }
    EXPECT_EQ(rx.get_attribute_names(), baseline_names);

    DiffProblemMemoryOptions metadata_options;
    metadata_options.gradient_location     = DEVICE;
    metadata_options.opt_var_storage       = OptVarStorage::MetadataOnly;
    metadata_options.opt_var_location      = LOCATION_ALL;  // ignored by policy
    metadata_options.opt_var_layout        = SoA;
    metadata_options.term_loss_location    = DEVICE;
    metadata_options.unique_internal_names = true;
    {
        ProblemT problem(rx, false, metadata_options, "metadata_opt_var");
        ASSERT_NE(problem.opt_var, nullptr);
        EXPECT_FALSE(problem.has_owned_opt_var());
        EXPECT_EQ(problem.grad.get_allocated() & DEVICE, DEVICE);
        EXPECT_EQ(problem.grad.get_allocated() & HOST, LOCATION_NONE);
        EXPECT_EQ(problem.opt_var->get_allocated(), LOCATION_NONE);
        EXPECT_EQ(problem.opt_var->get_layout(), SoA);

        add_smoothing_term(problem);
        ASSERT_EQ(problem.terms.size(), 1u);
        EXPECT_EQ(problem.terms[0]->get_loss_allocated() & DEVICE, DEVICE);
        EXPECT_EQ(problem.terms[0]->get_loss_allocated() & HOST, LOCATION_NONE);

        // No-argument evaluation rejects metadata-only storage explicitly,
        // without dereferencing its null data pointer.
        try {
            problem.eval_terms_grad_only();
            FAIL() << "Expected metadata-only evaluation to throw";
        } catch (const std::invalid_argument& error) {
            const std::string message(error.what());
            EXPECT_NE(message.find("eval_terms_grad_only"), std::string::npos);
            EXPECT_NE(message.find("MetadataOnly"), std::string::npos);
        }
    }
    EXPECT_EQ(rx.get_attribute_names(), baseline_names);

    DiffProblemMemoryOptions absent_options = metadata_options;
    absent_options.opt_var_storage          = OptVarStorage::Absent;
    {
        ProblemT problem(rx, false, absent_options, "absent_opt_var");
        EXPECT_EQ(problem.opt_var, nullptr);
        EXPECT_FALSE(problem.has_owned_opt_var());
        EXPECT_EQ(rx.get_attribute_names(), baseline_names);
        try {
            problem.eval_terms_grad_only();
            FAIL() << "Expected absent-storage evaluation to throw";
        } catch (const std::invalid_argument& error) {
            const std::string message(error.what());
            EXPECT_NE(message.find("eval_terms_grad_only"), std::string::npos);
            EXPECT_NE(message.find("Absent"), std::string::npos);
        }
    }
    EXPECT_EQ(rx.get_attribute_names(), baseline_names);

    // Repeated construction exercises mesh monotonic names and exact
    // identity cleanup rather than relying on allocator addresses.
    for (int i = 0; i < 100; ++i) {
        ProblemT problem(rx, false, metadata_options, "cycle_opt_var");
        ASSERT_NE(problem.opt_var, nullptr);
        EXPECT_EQ(problem.opt_var->get_allocated(), LOCATION_NONE);
    }
    EXPECT_EQ(rx.get_attribute_names(), baseline_names);

    // Exact-identity cleanup must leave an earlier user registration intact,
    // even when backward-compatible mode uses the same visible name.
    auto user_attr = rx.add_vertex_attribute<T>("shared_opt_var_name", 1, HOST);
    const auto               names_with_user        = rx.get_attribute_names();
    DiffProblemMemoryOptions duplicate_name_options = metadata_options;
    duplicate_name_options.unique_internal_names    = false;
    {
        ProblemT problem(
            rx, false, duplicate_name_options, "shared_opt_var_name");
        ASSERT_NE(problem.opt_var, nullptr);
        EXPECT_STREQ(problem.opt_var->get_name(), "shared_opt_var_name");
        EXPECT_EQ(rx.get_attribute_names().size(), names_with_user.size() + 1);
    }
    EXPECT_EQ(rx.get_attribute_names(), names_with_user);
    rx.remove_attribute(user_attr.get());
    EXPECT_EQ(rx.get_attribute_names(), baseline_names);
}
