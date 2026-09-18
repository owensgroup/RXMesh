#include <CLI/CLI.hpp>

#include <Eigen/Core>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"
#include "rxmesh/geometry_factory.h"
#include "rxmesh/matrix/pcg_solver.h"
#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename T>
void deform(RXMeshStatic&  rx,
            const uint32_t nx,
            const T        penalty,
            const T        displacement)
{
    constexpr int VariableDim     = 3;
    constexpr int newton_max_iter = 100;
    constexpr int cg_max_iter     = 1000;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    ProblemT problem(rx, true);

    // alloc
    auto coordinates = *rx.get_input_vertex_coordinates();
    problem.opt_var->copy_from(coordinates, DEVICE, DEVICE);

    auto rest_inverse =
        *rx.add_tet_attribute<Eigen::Matrix<T, 3, 3>>("rest_inverse", 1);
    auto rest_volume = *rx.add_tet_attribute<T>("rest_volume", 1);
    auto constraints = *rx.add_vertex_attribute<int>("constraints", 1);
    auto targets     = *rx.add_vertex_attribute<T>("constraint_targets", 3);
    constraints.reset(0, HOST);
    targets.reset(T(0), HOST);

    // prep
    rx.for_each<Op::TV, 256>([=] __device__(const TetHandle&      th,
                                            const VertexIterator& tv) mutable {
        const Eigen::Vector3<T> x0 = coordinates.template to_eigen<3>(tv[0]);
        const Eigen::Vector3<T> x1 = coordinates.template to_eigen<3>(tv[1]);
        const Eigen::Vector3<T> x2 = coordinates.template to_eigen<3>(tv[2]);
        const Eigen::Vector3<T> x3 = coordinates.template to_eigen<3>(tv[3]);

        const Eigen::Matrix<T, 3, 3> Dm = col_mat(x1 - x0, x2 - x0, x3 - x0);

        rest_inverse(th) = Dm.inverse();
        rest_volume(th)  = std::abs(Dm.determinant()) / T(6);
    });

    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        for (int i = 0; i < 3; ++i) {
            targets(vh, i) = coordinates(vh, i);
        }

        const uint32_t x = rx.map_to_global(vh) % nx;
        if (x == 0) {
            constraints(vh) = 1;
        } else if (x == nx - 1) {
            constraints(vh) = 2;
            targets(vh, 1) += displacement;
        }
    });
    constraints.move(HOST, DEVICE);
    targets.move(HOST, DEVICE);

    // add term
    problem.template add_term<Op::TV, true>(
        [=] __device__(const auto& th, const auto& tv, auto& opt_var) {
            using ActiveT = ACTIVE_TYPE(th);

            const Eigen::Vector3<ActiveT> x0 =
                opt_var.template active<3>(th, tv, 0);
            const Eigen::Vector3<ActiveT> x1 =
                opt_var.template active<3>(th, tv, 1);
            const Eigen::Vector3<ActiveT> x2 =
                opt_var.template active<3>(th, tv, 2);
            const Eigen::Vector3<ActiveT> x3 =
                opt_var.template active<3>(th, tv, 3);

            const Eigen::Matrix<ActiveT, 3, 3> Ds =
                col_mat(x1 - x0, x2 - x0, x3 - x0);
            const Eigen::Matrix<T, 3, 3>       inv_Dm = rest_inverse(th);
            const Eigen::Matrix<ActiveT, 3, 3> J      = Ds * inv_Dm;

            if (J.determinant() <= 0.0) {
                using PassiveT = PassiveType<ActiveT>;
                return ActiveT(std::numeric_limits<PassiveT>::max());
            }

            return rest_volume(th) *
                   (J.squaredNorm() + J.inverse().squaredNorm());
        });

    // constraints penalty
    problem.template add_term<Op::V>(
        [=] __device__(const auto& vh, auto& opt_var) {
            using ActiveT = ACTIVE_TYPE(vh);

            if (constraints(vh) == 0) {
                return ActiveT(0);
            }

            const Eigen::Vector3<ActiveT> x = opt_var.template active<3>(vh);
            const Eigen::Vector3<T> target  = targets.template to_eigen<3>(vh);
            return penalty * (x - target).squaredNorm();
        });

    constexpr int Order = ProblemT::DenseMatT::OrderT;

    PCGSolver<T, Order> solver(
        *problem.hess, 1, cg_max_iter, std::numeric_limits<T>::min(), T(1e-10));
    NewtonSolver newton(problem, &solver);

    for (int iter = 0; iter < newton_max_iter; ++iter) {
        problem.eval_terms();
        const T energy = problem.get_current_loss();
        newton.compute_direction();

        const T decrement = T(0.5) * problem.grad.dot(newton.dir);
        RXMESH_INFO("Iteration {}: energy = {}, decrement = {}, PCG = {}",
                    iter,
                    energy,
                    decrement,
                    solver.iter_taken());

        if (decrement < T(1e-4)) {
            break;
        }

        if (!newton.line_search(T(1), T(0.5), 64, T(0))) {
            RXMESH_WARN("Line search failed");
            break;
        }
    }

#if USE_POLYSCOPE
    problem.opt_var->move(DEVICE, HOST);
    auto* volume_mesh = rx.get_polyscope_volume_mesh();
    volume_mesh->updateVertexPositions(*problem.opt_var);
    volume_mesh->addVertexScalarQuantity("constraints", constraints)
        ->setEnabled(true);
    polyscope::show();
#endif
}

int main(int argc, char** argv)
{
    using T = rx_coord_t;

    CLI::App app{"Constrained volumetric tet-mesh deformation"};

    uint32_t nx           = 10;
    uint32_t ny           = 10;
    uint32_t nz           = 10;
    uint32_t device_id    = 0;
    T        penalty      = T(1e5);
    T        displacement = T(0.05);

    app.add_option("--nx", nx, "Number of points along x")->default_val(nx);
    app.add_option("--ny", ny, "Number of points along y")->default_val(ny);
    app.add_option("--nz", nz, "Number of points along z")->default_val(nz);
    app.add_option("-d,--device_id", device_id, "GPU device ID")
        ->default_val(device_id);
    app.add_option("-p,--penalty", penalty, "Soft-constraint penalty")
        ->default_val(penalty);
    app.add_option("--displacement",
                   displacement,
                   "Handle displacement as a fraction of mesh extent")
        ->default_val(displacement);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    if (nx < 2 || ny < 2 || nz < 2) {
        RXMESH_ERROR("The number of points along each axis must be at least 2");
        return EXIT_FAILURE;
    }

    rx_init(device_id);

    RXMESH_INFO("nx = {}", nx);
    RXMESH_INFO("ny = {}", ny);
    RXMESH_INFO("nz = {}", nz);
    RXMESH_INFO("device_id = {}", device_id);
    RXMESH_INFO("penalty = {}", penalty);
    RXMESH_INFO("displacement = {}", displacement);

    std::vector<std::vector<T>>        vertices;
    std::vector<std::vector<uint32_t>> tets;
    const T                            dx = T(1) / T(nx - 1);

    create_tet_box(vertices, tets, nx, ny, nz, dx);

    RXMeshStatic rx(vertices, tets);

    deform(rx, nx, penalty, displacement);

    return EXIT_SUCCESS;
}
