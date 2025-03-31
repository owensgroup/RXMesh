#pragma once

#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/gradient_descent.h"

void smoothing_gd()
{
    using namespace rxmesh;

    // RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");
    RXMeshStatic rx(rxmesh_args.obj_file_name);

    using T = float;

    constexpr int VariableDim = 3;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle>;

    ProblemT problem(rx);

    auto v_input_pos = *rx.get_input_vertex_coordinates();

    problem.objective->copy_from(v_input_pos, DEVICE, DEVICE);


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


    float learning_rate = 0.01;

    GradientDescent gd(problem, learning_rate);

    int num_iterations = 100;

    GPUTimer timer;
    timer.start();

    for (int iter = 0; iter < num_iterations; ++iter) {

        problem.eval_terms();

        float energy = problem.get_current_loss();

        if (iter % 10 == 0) {
            RXMESH_INFO("Iteration = {}: Energy = {}", iter, energy);
        }

        gd.take_step();
    }
    timer.stop();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::cout << "\nSmoothing RXMesh: " << timer.elapsed_millis() << " (ms),"
              << timer.elapsed_millis() / float(num_iterations)
              << " ms per iteration\n";

#if USE_POLYSCOPE
    problem.objective->move(DEVICE, HOST);
    rx.get_polyscope_mesh()->updateVertexPositions(*problem.objective);
    // polyscope::show();
#endif
}

TEST(DiffAttribute, SmoothingGD)
{
    smoothing_gd();
}
