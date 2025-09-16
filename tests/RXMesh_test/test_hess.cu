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

void new_entries(int*     d_new_rows,
                 int*     d_new_cols,
                 uint32_t new_size,
                 uint32_t num_v)
{
    using namespace rxmesh;

    for_each_item<<<1, new_size>>>(
        new_size,
        [d_new_rows, d_new_cols, num_v] __device__(int i) mutable {
            int id0 = i * 2 + 0;
            int id1 = i * 2 + 1;

            int v0 = i;

            int v1 = num_v - i - 1;

            d_new_rows[id0] = v0;
            d_new_cols[id0] = v1;

            d_new_rows[id1] = v1;
            d_new_cols[id1] = v0;
        }

    );
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

    problem.hess->move(DEVICE, HOST);

    problem.hess->for_each([&](int i, int j, T val) {
        if (i == j) {
            EXPECT_NEAR(val, mass, 1e-3);
        } else {
            EXPECT_NEAR(val, 0, 1e-3);
        }
    });
}

TEST(Diff, HessUpdate)
{
    using namespace rxmesh;

    using T = float;

    std::vector<std::vector<T>> verts;

    std::vector<std::vector<uint32_t>> fv;

    int n = 12;

    T dx = 1 / T(n - 1);

    create_plane(verts, fv, n, n, 2, dx, true);

    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(verts, "Coords");

    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    ProblemT problem(rx, true, 1.f);

    // mass per vertex = rho * volume /num_vertices
    T mass = 0.01;

    auto x = *rx.get_input_vertex_coordinates();

    // add inertial energy term
    add_term(problem, x, mass);

    problem.eval_terms();

    // new pairs to insert
    int *d_new_rows, *d_new_cols;

    int new_size = n;

    // the 2 is because if v0 adds v1 then v1 should add v0 to make the hessian
    // symmetric
    CUDA_ERROR(cudaMalloc((void**)&d_new_rows, 2 * new_size * sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&d_new_cols, 2 * new_size * sizeof(int)));

    new_entries(d_new_rows, d_new_cols, new_size, rx.get_num_vertices());

    // problem.hess->reset(0, HOST);
    // problem.hess->to_file("old_hess");

    problem.update_hessian(2 * new_size, d_new_rows, d_new_cols);


    // problem.hess->reset(0, HOST);
    // problem.hess->to_file("new_hess");

    GPU_FREE(d_new_rows);
    GPU_FREE(d_new_cols);
}