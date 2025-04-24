#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/diff/diff_scalar_problem.h"

template <typename ProblemT, typename T>
void add_term(ProblemT& problem, rxmesh::VertexAttribute<T>& x, T mass)
{
    using namespace rxmesh;

    problem.template add_term<Op::V, true>(
        [x, mass] __device__(const auto& vh, auto& obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            Eigen::Vector3<ActiveT> x_tilda = iter_val<ActiveT, 3>(vh, obj);

            Eigen::Vector3<T> xx = x.to_eigen<3>(vh);

            Eigen::Vector3<ActiveT> l = xx - x_tilda;

            ActiveT E = T(0.5) * mass * l.squaredNorm();

            return E;
        });
}
TEST(Diff, Hess)
{
    // Test the hessian of the inertia term of a mass-spring system
    // https://phys-sim-book.github.io/lec4.2-inertia.html

    using namespace rxmesh;

    using T = float;

    std::vector<std::vector<T>>        verts;
    std::vector<std::vector<uint32_t>> fv;

    int n = 16;

    T dx = 1 / T(n - 1);

    create_plane(verts, fv, n, n, 2, dx, true);

    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(verts, "Coords");

    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    ProblemT problem(rx);

    // mass per vertex = rho * volume /num_vertices
    T mass = 0.01;

    auto x = *rx.get_input_vertex_coordinates();

    // add inertial energy term
    add_term(problem, x, mass);


    problem.eval_terms();

    problem.hess.move(DEVICE, HOST);

    problem.hess.for_each([&](int i, int j, T val) {
        if (i == j) {
            EXPECT_NEAR(val, mass, 1e-3);
        } else {
            EXPECT_NEAR(val, 0, 1e-3);
        }
    });
}