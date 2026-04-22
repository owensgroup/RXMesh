#pragma once

#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/gradient_descent.h"
#include "rxmesh/diff/newton_solver.h"


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
        Eigen::Vector3<ActiveT> d0 = iter_val<ActiveT, 3>(eh, iter, opt_var, 0);
        Eigen::Vector3<ActiveT> d1 = iter_val<ActiveT, 3>(eh, iter, opt_var, 1);

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

    NetwtonSolver newton(problem, &solver);

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

    problem.template add_term<Op::V>([=] __device__(const auto& vh,
                                                    auto& opt_var) mutable {
        using ActiveT = ACTIVE_TYPE(vh);

        tol = tol;

        Eigen::Vector<ActiveT, 1> xx = opt_var.template active<ActiveT, 1>(vh);

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
