
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

using namespace rxmesh;


template <typename T>
void mass_spring(RXMeshStatic& rx)
{
    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    using HessMatT = typename ProblemT::HessMatT;

    T rho = 1000, k = 1e5, initial_stretch = 1.3, h = 0.004, tol = 0.01;

    glm::vec3 bb_lower(0), bb_upper(0);
    rx.bounding_box(bb_lower, bb_upper);
    glm::vec3 bb = bb_upper - bb_lower;

    // mass per vertex = rho * volume /num_vertices
    T mass = rho * bb[0] * bb[1] / rx.get_num_vertices();

    ProblemT problem(rx);

    LUSolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(&problem.hess);

    NetwtonSolver newton_solver(problem, &solver);

    auto rest_l = *rx.add_edge_attribute<T>("RestLen", 1);

    auto velocity = *rx.add_vertex_attribute<T>("Velocity", 3);

    auto coordinates = *rx.get_input_vertex_coordinates();


    // calc rest length
    rx.run_query_kernel<Op::EV, blockThreads>(
        [=] __device__(const EdgeHandle& eh, const VertexIterator& iter) {
            Eigen::Vector3<T> a = coordinates.to_eigen<3>(iter[0]);
            Eigen::Vector3<T> b = coordinates.to_eigen<3>(iter[1]);

            rest_l(eh) = (a - b).squaredNorm();
        });

    // apply initial stretch along the y direction
    rx.for_each_vertex(
        DEVICE,
        [initial_stretch, coordinates] __device__(const VertexHandle& vh) {
            coordinates(vh, 1) *= initial_stretch;
        });

    // add inertial energy term
    T half_mass = T(0.5) * mass;
    problem.template add_term<Op::EV, true>(
        [rest_l, half_mass] __device__(
            const auto& eh, const auto& iter, auto& obj) mutable {
            assert(iter[0].is_valid() && iter[1].is_valid());

            assert(iter.size() == 2);

            using ActiveT = ACTIVE_TYPE(eh);

            // tangent vectors at the triangle three vertices (a,b,c)
            Eigen::Vector3<ActiveT> a = iter_val<ActiveT, 3>(eh, iter, obj, 0);
            Eigen::Vector3<ActiveT> b = iter_val<ActiveT, 3>(eh, iter, obj, 1);

            ActiveT l = (a - b).squaredNorm();

            T r = rest_l(eh);

            ActiveT E = half_mass * (l - r) * (l - r);

            return E;
        });

    // add elastic potential energy term
    T half_k_times_h_sq = T(0.5) * k * h * h;
    problem.template add_term<Op::EV, true>(
        [rest_l, half_k_times_h_sq] __device__(
            const auto& eh, const auto& iter, auto& obj) mutable {
            assert(iter[0].is_valid() && iter[1].is_valid());

            assert(iter.size() == 2);

            using ActiveT = ACTIVE_TYPE(eh);

            // tangent vectors at the triangle three vertices (a,b,c)
            Eigen::Vector3<ActiveT> a = iter_val<ActiveT, 3>(eh, iter, obj, 0);
            Eigen::Vector3<ActiveT> b = iter_val<ActiveT, 3>(eh, iter, obj, 1);

            ActiveT l = (a - b).squaredNorm();

            T r = rest_l(eh);

            ActiveT s = (l / r) - T(1.0);

            ActiveT E = half_k_times_h_sq * r * s * s;

            return E;
        });


    int iter = 0;

    // GPUTimer timer;
    // timer.start();
    //
    // timer.stop();

    auto ps_callback = [&]() mutable {
        problem.objective->reset(0, DEVICE);

        problem.eval_terms();

        T f = problem.get_current_loss();

        RXMESH_INFO("Iteration= {}: Energy = {}", iter, f);

        newton_solver.newton_direction();

        T residual = newton_solver.dir.max() / h;
        if (residual <= tol) {
            return;
        }

        newton_solver.line_search();

#if USE_POLYSCOPE
        problem.objective->move(DEVICE, HOST);
        rx.get_polyscope_mesh()->updateVertexPositions(*problem.objective);
#endif
        iter++;
    };

#if USE_POLYSCOPE
    polyscope::state::userCallback = ps_callback;
    polyscope::show();
#endif

    RXMESH_INFO(
        "#Faces: {}, #Vertices: {}", rx.get_num_faces(), rx.get_num_vertices());

    // RXMESH_INFO(
    //     "Mass-spring: iterations ={}, time= {} (ms), "
    //     "timer/iteration= {} ms/iter",
    //     iter,
    //     timer.elapsed_millis(),
    //     timer.elapsed_millis() / float(num_iterations));
}

int main(int argc, char** argv)
{
    Log::init(spdlog::level::info);

    using T = float;

    std::vector<std::vector<float>>    verts;
    std::vector<std::vector<uint32_t>> fv;

    int n = 16;

    if (argc == 2) {
        n = atoi(argv[1]);
    }

    T dx = 1 / T(n - 1);

    create_plane(verts, fv, n, n, 2, dx);

    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(verts, "Coords");


    mass_spring<T>(rx);
}