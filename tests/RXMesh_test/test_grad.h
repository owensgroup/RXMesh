#pragma once

#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/gradient_descent.h"
#include "rxmesh/diff/newton_solver.h"


template <typename ProblemT>
inline void add_term(ProblemT& problem)
{
    using namespace rxmesh;

    problem.template add_term<Op::EV>(
        [=] __device__(const auto& eh, const auto& iter, auto& objective) {
            assert(iter.size() == 2);

            using ActiveT = ACTIVE_TYPE(eh);

            // pos
            Eigen::Vector3<ActiveT> d0 =
                iter_val<ActiveT, 3>(eh, iter, objective, 0);
            Eigen::Vector3<ActiveT> d1 =
                iter_val<ActiveT, 3>(eh, iter, objective, 1);

            Eigen::Vector3<ActiveT> dist = (d0 - d1);

            ActiveT dist_sq = dist.squaredNorm();

            return dist_sq;
        });
}

TEST(Diff, SmoothingNewton)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");
    // RXMeshStatic rx(rxmesh_args.obj_file_name);

    using T = float;

    constexpr int VariableDim = 3;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    ProblemT problem(rx);

    auto v_input_pos = *rx.get_input_vertex_coordinates();

    problem.objective->copy_from(v_input_pos, DEVICE, DEVICE);

    add_term(problem);


    using HessMatT = typename ProblemT::HessMatT;

    LUSolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(&problem.hess);

    NetwtonSolver newton(problem, &solver);

    int num_iterations = 100;

    T convergence_eps = 1e-2;

    GPUTimer timer;
    timer.start();

    for (int iter = 0; iter < num_iterations; ++iter) {

        problem.eval_terms();


        float energy = problem.get_current_loss();

        RXMESH_INFO("Iteration = {}: Energy = {}", iter, energy);


        newton.newton_direction();

        RXMESH_INFO("newton.dir.norm2() = {}", newton.dir.norm2());
        RXMESH_INFO("problem.grad.norm2() = {}", problem.grad.norm2());

        if (0.5f * problem.grad.dot(newton.dir) < convergence_eps) {
            break;
        }

        newton.line_search();
    }
    timer.stop();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::cout << "\nSmoothing Newton RXMesh: " << timer.elapsed_millis()
              << " (ms)," << timer.elapsed_millis() / float(num_iterations)
              << " ms per iteration\n";

    // so newton method on this function should lead to a vertex position that
    // is just zero since the function is quadratic

    problem.objective->move(DEVICE, HOST);

    T f = (*problem.objective)(VertexHandle(0, 0), 0);

    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR((*problem.objective)(vh, 0), f, 1e-3);
        }
    });
}
