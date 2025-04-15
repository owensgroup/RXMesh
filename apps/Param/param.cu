// Reference Implementation
// https://github.com/patr-schm/TinyAD-Examples/blob/main/apps/parametrization_openmesh.cc
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/algo/tutte_embedding.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

using namespace rxmesh;


template <typename T>
void add_mesh_to_polyscope(RXMeshStatic&       rx,
                           VertexAttribute<T>& v,
                           std::string         name)
{
    if (v.get_num_attributes() == 3) {
        polyscope::registerSurfaceMesh(name, v, rx.get_polyscope_mesh()->faces);
    } else {
        auto v3 = *rx.add_vertex_attribute<T>(name, 3);

        rx.for_each_vertex(HOST, [&](const VertexHandle h) {
            v3(h, 0) = v(h, 0);
            v3(h, 1) = v(h, 1);
            v3(h, 2) = 0;
        });

        polyscope::registerSurfaceMesh(
            name, v3, rx.get_polyscope_mesh()->faces);

        rx.remove_attribute(name);
    }
}

template <typename T>
void parameterize(RXMeshStatic& rx)
{
    constexpr int VariableDim = 2;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    auto coordinates = *rx.get_input_vertex_coordinates();

    ProblemT problem(rx);

    using HessMatT = typename ProblemT::HessMatT;

    LUSolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(&problem.hess);
    // CholeskySolver<HessMatT, ProblemT::DenseMatT::OrderT>
    // solver(&problem.hess);


    NetwtonSolver newton_solver(problem, &solver);

    // TODO this is a AoS and should be converted into SoA
    auto rest_shape =
        *rx.add_face_attribute<Eigen::Matrix<T, 2, 2>>("fRestShape", 1);

    tutte_embedding(rx, coordinates, *problem.objective);

    // auto uv = *problem.objective;

    // uv(VertexHandle(0, 0), 0) = 1;
    // uv(VertexHandle(0, 0), 1) = 0;
    //
    // uv(VertexHandle(0, 1), 0) = 0.25;
    // uv(VertexHandle(0, 1), 1) = -0.25;
    //
    // uv(VertexHandle(0, 2), 0) = -1.83697e-16;
    // uv(VertexHandle(0, 2), 1) = -1;
    //
    // uv(VertexHandle(0, 3), 0) = -0.166667;
    // uv(VertexHandle(0, 3), 1) = -0.166667;
    //
    // uv(VertexHandle(0, 4), 0) = -1;
    // uv(VertexHandle(0, 4), 1) = 1.22465e-16;
    //
    // uv(VertexHandle(0, 5), 0) = -0.25;
    // uv(VertexHandle(0, 5), 1) = 0.25;
    //
    // uv(VertexHandle(0, 6), 0) = 6.12323e-17;
    // uv(VertexHandle(0, 6), 1) = 1;
    //
    // uv(VertexHandle(0, 7), 0) = 0.166667;
    // uv(VertexHandle(0, 7), 1) = 0.166667;
    // uv.move(HOST, DEVICE);

    rx.get_polyscope_mesh()->addVertexParameterizationQuantity(
        "uv_tutte", *problem.objective);

    add_mesh_to_polyscope(rx, *problem.objective, "tutte_mesh");

    constexpr uint32_t blockThreads = 256;

    // 1) compute rest shape
    rx.run_query_kernel<Op::FV, blockThreads>(
        [=] __device__(const FaceHandle& fh, const VertexIterator& iter) {
            const VertexHandle v0 = iter[0];
            const VertexHandle v1 = iter[1];
            const VertexHandle v2 = iter[2];

            assert(v0.is_valid() && v1.is_valid() && v2.is_valid());

            // 3d position
            Eigen::Vector3<T> ar_3d = coordinates.to_eigen<3>(v0);
            Eigen::Vector3<T> br_3d = coordinates.to_eigen<3>(v1);
            Eigen::Vector3<T> cr_3d = coordinates.to_eigen<3>(v2);

            // Local 2D coordinate system
            Eigen::Vector3<T> n  = (br_3d - ar_3d).cross(cr_3d - ar_3d);
            Eigen::Vector3<T> b1 = (br_3d - ar_3d).normalized();
            Eigen::Vector3<T> b2 = n.cross(b1).normalized();

            // Express a, b, c in local 2D coordinates system
            Eigen::Vector2<T> ar_2d(T(0.0), T(0.0));
            Eigen::Vector2<T> br_2d((br_3d - ar_3d).dot(b1), T(0.0));
            Eigen::Vector2<T> cr_2d((cr_3d - ar_3d).dot(b1),
                                    (cr_3d - ar_3d).dot(b2));

            // Save 2-by-2 matrix with edge vectors as columns
            Eigen::Matrix<T, 2, 2> fout = col_mat(br_2d - ar_2d, cr_2d - ar_2d);

            rest_shape(fh) = fout;
        });


    // add energy term
    problem.template add_term<Op::FV, true>(
        [=] __device__(const auto& fh, const auto& iter, auto& objective) {
            // fh is a face handle
            // iter is an iterator over fh's vertices
            // objective is the uv coordinates

            assert(iter[0].is_valid() && iter[1].is_valid() &&
                   iter[2].is_valid());

            assert(iter.size() == 3);

            using ActiveT = ACTIVE_TYPE(fh);

            // uv
            Eigen::Vector2<ActiveT> a =
                iter_val<ActiveT, 2>(fh, iter, objective, 0);
            Eigen::Vector2<ActiveT> b =
                iter_val<ActiveT, 2>(fh, iter, objective, 1);
            Eigen::Vector2<ActiveT> c =
                iter_val<ActiveT, 2>(fh, iter, objective, 2);


            // Triangle flipped?
            Eigen::Matrix<ActiveT, 2, 2> M = col_mat(b - a, c - a);

            if (M.determinant() <= 0.0) {
                using PassiveT = PassiveType<ActiveT>;
                return ActiveT(std::numeric_limits<PassiveT>::max());
            }

            // Get constant 2D rest shape and area of triangle t
            const Eigen::Matrix<T, 2, 2> Mr = rest_shape(fh);

            const T A = T(0.5) * Mr.determinant();

            // Compute symmetric Dirichlet energy
            Eigen::Matrix<ActiveT, 2, 2> J = M * Mr.inverse();

            ActiveT res = A * (J.squaredNorm() + J.inverse().squaredNorm());


            return res;
        });


    T convergence_eps = 1e-2;

    int num_iterations = 100;
    int iter;

    GPUTimer timer;
    timer.start();

    for (iter = 0; iter < num_iterations; ++iter) {

        // 2) calc loss function
        problem.eval_terms();

        // 3) get the current value of the loss function
        T f = problem.get_current_loss();
        RXMESH_INFO("Iteration= {}: Energy = {}", iter, f);

        // 3) direction newton
        newton_solver.newton_direction();

        // 4) newton decrement
        if (0.5f * problem.grad.dot(newton_solver.dir) < convergence_eps) {
            break;
        }

        // 5) line search
        newton_solver.line_search();
    }

    timer.stop();


    RXMESH_INFO(
        "Parametrization: iterations ={}, time= {} (ms), "
        "timer/iteration= {} ms/iter, solver_time = {} (ms)",
        iter,
        timer.elapsed_millis(),
        timer.elapsed_millis() / float(num_iterations),
        newton_solver.solve_time);


    problem.objective->move(DEVICE, HOST);

    rx.get_polyscope_mesh()->addVertexParameterizationQuantity(
        "uv_opt", *problem.objective);

    add_mesh_to_polyscope(rx, *problem.objective, "opt_mesh");

    polyscope::show();
}

int main(int argc, char** argv)
{
    Log::init(spdlog::level::trace);

    const uint32_t device_id = 0;
    cuda_query(device_id);

    using T = float;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");

    if (rx.is_closed()) {
        RXMESH_ERROR(
            "The input mesh is closed. The input mesh should have boundaries.");
        exit(EXIT_FAILURE);
    }

    parameterize<T>(rx);
}