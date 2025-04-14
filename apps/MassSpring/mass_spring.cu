
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

using namespace rxmesh;


template <typename T>
void mass_spring(RXMeshStatic& rx)
{
    constexpr int VariableDim = 3;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    ProblemT problem(rx);

    using HessMatT = typename ProblemT::HessMatT;

    LUSolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(&problem.hess);

    NetwtonSolver newton_solver(problem, &solver);

    auto rest_l = *rx.add_edge_attribute<T>("Rest Len", 1);

    auto coordinates = *rx.get_input_vertex_coordinates();

    constexpr uint32_t blockThreads = 256;

    rx.run_query_kernel<Op::EV, blockThreads>(
        [=] __device__(const EdgeHandle& eh, const VertexIterator& iter) {
            Eigen::Vector3<T> a = coordinates.to_eigen<3>(iter[0]);
            Eigen::Vector3<T> b = coordinates.to_eigen<3>(iter[1]);

            rest_l(eh) = T(0.5) * (a - b).norm();
        });


    // add energy term
    problem.template add_term<Op::EV, true>(
        [rest_l] __device__(
            const auto& eh, const auto& iter, auto& obj) mutable {
            assert(iter[0].is_valid() && iter[1].is_valid());

            assert(iter.size() == 2);

            using ActiveT = ACTIVE_TYPE(eh);

            // tangent vectors at the triangle three vertices (a,b,c)
            Eigen::Vector3<ActiveT> a = iter_val<ActiveT, 3>(eh, iter, obj, 0);
            Eigen::Vector3<ActiveT> b = iter_val<ActiveT, 3>(eh, iter, obj, 1);

            ActiveT l = (a - b).norm();

            T r = rest_l(eh);

            ActiveT E = T(0.5) * (l - r) * (l - r);

            return E;
        });

    T convergence_eps = 1e-1;

    int num_iterations = 1000;
    int iter;

    GPUTimer timer;
    timer.start();

    for (iter = 0; iter < num_iterations; ++iter) {

        problem.objective->reset(0, DEVICE);

        problem.eval_terms();

        T f = problem.get_current_loss();

        // RXMESH_INFO("Iteration= {}: Energy = {}", iter, f);
        //
        // newton_solver.newton_direction();
        //
        //
        // if (0.5f * problem.grad.dot(newton_solver.dir) < convergence_eps) {
        //     break;
        // }
        //
        //
        // newton_solver.line_search();
    }

    timer.stop();


    RXMESH_INFO("#Faces: {}, #Vertices: {}",
                rx.get_num_faces(),
                rx.get_num_vertices());
    RXMESH_INFO(
        "Mass-spring: iterations ={}, time= {} (ms), "
        "timer/iteration= {} ms/iter",
        iter,
        timer.elapsed_millis(),
        timer.elapsed_millis() / float(num_iterations));

    // polyscope::show();
}

int main(int argc, char** argv)
{
    Log::init(spdlog::level::info);

    using T = float;

    std::vector<std::vector<float>>    verts;
    std::vector<std::vector<uint32_t>> fv;
    if (argc == 3) {
        create_plane(verts, fv, atoi(argv[1]), atoi(argv[2]));
    } else {
        create_plane(verts, fv, 100, 30);
    }
    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(verts, "Coords");


    mass_spring<T>(rx);
}