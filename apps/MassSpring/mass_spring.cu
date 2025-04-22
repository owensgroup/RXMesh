
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

    const T rho             = 1000;  // density
    const T k               = 1e5;   // stiffness
    const T initial_stretch = 1.3;
    const T tol             = 0.01;
    const T h               = 0.004;  // time step
    const T inv_h           = T(1) / h;

    glm::vec3 bb_lower(0), bb_upper(0);
    rx.bounding_box(bb_lower, bb_upper);
    glm::vec3 bb = bb_upper - bb_lower;

    // mass per vertex = rho * volume /num_vertices
    T mass = rho * bb[0] * bb[1] / rx.get_num_vertices();

    ProblemT problem(rx);

    CholeskySolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(&problem.hess);

    NetwtonSolver newton_solver(problem, &solver);

    auto rest_l = *rx.add_edge_attribute<T>("RestLen", 1);

    auto velocity = *rx.add_vertex_attribute<T>("Velocity", 3);
    velocity.reset(DEVICE, 0);

    auto x = *rx.get_input_vertex_coordinates();

    auto& x_tilde = *problem.objective;


    // calc rest length
    rx.run_query_kernel<Op::EV, blockThreads>(
        [=] __device__(const EdgeHandle& eh, const VertexIterator& iter) {
            Eigen::Vector3<T> a = x.to_eigen<3>(iter[0]);
            Eigen::Vector3<T> b = x.to_eigen<3>(iter[1]);

            rest_l(eh) = (a - b).squaredNorm();
        });

    // apply initial stretch along the y direction
    rx.for_each_vertex(
        DEVICE,
        [initial_stretch, x, x_tilde] __device__(const VertexHandle& vh) {
            x(vh, 1) = x(vh, 1) * initial_stretch;
        });


    // add inertial energy term
    T half_mass = T(0.5) * mass;
    problem.template add_term<Op::V, true>(
        [x, half_mass] __device__(const auto& vh, auto& obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            Eigen::Vector3<ActiveT> x_tilda = iter_val<ActiveT, 3>(vh, obj);

            Eigen::Vector3<T> xx = x.to_eigen<3>(vh);

            Eigen::Vector3<ActiveT> l = xx - x_tilda;

            ActiveT E = half_mass * l.squaredNorm();

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

            const Eigen::Vector3<ActiveT> a =
                iter_val<ActiveT, 3>(eh, iter, obj, 0);
            const Eigen::Vector3<ActiveT> b =
                iter_val<ActiveT, 3>(eh, iter, obj, 1);

            const T r = rest_l(eh);

            const Eigen::Vector3<ActiveT> diff = a - b;

            const ActiveT ratio = diff.squaredNorm() / r;

            const ActiveT s = (ratio - T(1.0));

            const ActiveT E = half_k_times_h_sq * r * s * s;

            return E;
        });


    int time_step = 0;

    // GPUTimer timer;
    // timer.start();
    //
    // timer.stop();

    auto ps_callback = [&]() mutable {
        // update x_tilde
        rx.for_each_vertex(
            DEVICE,
            [x, x_tilde, velocity, h] __device__(VertexHandle vh) mutable {
                for (int i = 0; i < 3; ++i) {
                    x_tilde(vh, i) = x(vh, i) + h * velocity(vh, i);
                }
            });

        problem.eval_terms();

        T f = problem.get_current_loss();

        RXMESH_INFO("### Time step {} ###", time_step);

        RXMESH_INFO("*** E_last {}", f);

        newton_solver.newton_direction();

        T residual = newton_solver.dir.abs_max() / h;

        newton_solver.line_search();

        int iter = 0;
        if (residual > tol) {
            RXMESH_INFO("iter= {}:", iter);
            RXMESH_INFO("residual = {}", residual);

            newton_solver.line_search();

            problem.eval_terms();

            newton_solver.newton_direction();

            residual = newton_solver.dir.abs_max() / h;

            iter++;
        }

        RXMESH_INFO("iter= {}:", iter);
        RXMESH_INFO("residual = {}", residual);

        //  update velocity
        rx.for_each_vertex(
            DEVICE,
            [x, x_tilde, velocity, inv_h] __device__(VertexHandle vh) mutable {
                for (int i = 0; i < 3; ++i) {
                    velocity(vh, i) = inv_h * (x_tilde(vh, i) - x(vh, i));

                    x(vh, i) = x_tilde(vh, i);
                }
            });

#if USE_POLYSCOPE
        x_tilde.move(DEVICE, HOST);
        rx.get_polyscope_mesh()->updateVertexPositions(x_tilde);
#endif

        time_step++;
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

    int n = 160;

    if (argc == 2) {
        n = atoi(argv[1]);
    }

    T dx = 1 / T(n - 1);

    create_plane(verts, fv, n, n, 2, dx);

    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(verts, "Coords");


    mass_spring<T>(rx);
}