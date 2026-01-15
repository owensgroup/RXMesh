#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/diff/diff_scalar_problem.h"

using namespace rxmesh;
using PairT = std::pair<VertexHandle, VertexHandle>;

template <typename ProblemT, typename T>
void add_inertia_term(ProblemT& problem, rxmesh::VertexAttribute<T>& x, T mass)
{
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

template <typename ProblemT>
void new_entries(RXMeshStatic& rx,
                 ProblemT&     problem,
                 size_t        num_new_pairs,
                 PairT*        d_pairs)
{
    problem.vv_pairs.reset();

    constexpr uint32_t blockThreads = 256;

    uint32_t blocks = DIVIDE_UP(num_new_pairs, blockThreads);

    for_each_item<<<blocks, blockThreads>>>(
        num_new_pairs,
        [d_pairs, contact_pairs = problem.vv_pairs] __device__(int i) mutable {
            bool inserted =
                contact_pairs.insert(d_pairs[i].first, d_pairs[i].second);
            assert(inserted);
        }

    );
}


std::pair<PairT*, size_t> generate_pairs(RXMeshStatic& rx)
{
    auto x = *rx.get_input_vertex_coordinates();

    std::vector<VertexHandle> bottom;

    rx.for_each_vertex(
        HOST,
        [&](const VertexHandle& vh) {
            if (x(vh, 1) < 0.1) {
                bottom.push_back(vh);
            }
        },
        NULL,
        false);


    std::vector<PairT> pairs;

    rx.for_each_vertex(
        HOST,
        [&](const VertexHandle& vh) {
            if (x(vh, 1) > 0.5) {
                for (int i = 0; i < bottom.size(); ++i) {
                    VertexHandle nh(bottom[i]);

                    if (std::abs(x(vh, 0) - x(nh, 0)) < 0.00001) {
                        PairT p(vh, nh);
                        pairs.push_back(p);
                    }
                }
            }
        },
        NULL,
        false);

    PairT* d_pairs = nullptr;

    CUDA_ERROR(cudaMalloc((void**)&d_pairs, sizeof(PairT) * pairs.size()));

    CUDA_ERROR(cudaMemcpy(d_pairs,
                          pairs.data(),
                          sizeof(PairT) * pairs.size(),
                          cudaMemcpyHostToDevice));

    return {d_pairs, pairs.size()};
}

template <typename ProblemT, typename T>
void verify_inertia_hess(ProblemT& problem, T mass)
{
    problem.hess->move(DEVICE, HOST);

    problem.hess->for_each([&](int i, int j, auto val) {
        if (i == j) {
            EXPECT_NEAR(val, mass, 1e-3);
        } else {
            EXPECT_NEAR(val, 0, 1e-3);
        }
    });
}

TEST(Diff, Hess)
{
    // Test the hessian of the inertia term of a mass-spring system
    // https://phys-sim-book.github.io/lec4.2-inertia.html

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

    ProblemT problem(rx, true);

    // mass per vertex = rho * volume /num_vertices
    T mass = 0.01;

    auto x = *rx.get_input_vertex_coordinates();

    // add inertial energy term
    add_inertia_term(problem, x, mass);
    problem.eval_terms();
    verify_inertia_hess(problem, mass);
}

TEST(Diff, HessUpdate)
{
    using T = float;

    std::vector<std::vector<T>> verts;

    std::vector<std::vector<uint32_t>> fv;

    int n = 5;

    T dx = 1 / T(n - 1);

    create_plane(verts, fv, n, n, 2, dx, true);

    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(verts, "Coords");

    constexpr int VariableDim = 3;


    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    int expected_num_new_pairs = rx.get_num_vertices()*rx.get_num_vertices();

    ProblemT problem(rx, true, expected_num_new_pairs);

    T mass = 0.01;

    auto x = *rx.get_input_vertex_coordinates();

    add_inertia_term(problem, x, mass);
    problem.eval_terms();
    T prv_loss = problem.get_current_loss();
    verify_inertia_hess(problem, mass);

    int prev_nnz = problem.hess->non_zeros();

    problem.hess->reset(0, HOST);
    problem.hess->move(DEVICE, HOST);
    problem.hess->to_file("old_hess");

    auto [d_pairs, num_new_pairs] = generate_pairs(rx);

    EXPECT_EQ(num_new_pairs, expected_num_new_pairs);

    new_entries(rx, problem, num_new_pairs, d_pairs);
    problem.update_hessian();

    problem.eval_terms();
    T   new_loss = problem.get_current_loss();
    int new_nnz  = problem.hess->non_zeros();

    EXPECT_NEAR(prv_loss, new_loss, 0.00001);

    EXPECT_EQ(new_nnz,
              prev_nnz + 2 * num_new_pairs * VariableDim * VariableDim);

    problem.hess->reset(0, HOST);
    problem.hess->move(DEVICE, HOST);
    problem.hess->to_file("new_hess");

    // Note that we did not add any new term for the contact pairs. So Hessian
    // still evaluate to the same Hessian without these new contact pairs--we
    // basically just a few new entries that are zeros
    verify_inertia_hess(problem, mass);

    GPU_FREE(d_pairs);
}

template <typename ProblemT>
void add_vf_term(ProblemT& problem, int face_id, int vert_id)
{
    auto vf_pairs = problem.vf_pairs;
    vf_pairs.reset();

    for_each_item<<<1, 1>>>(1, [=] __device__(int i) mutable {
        FaceHandle   fh(0, face_id);
        VertexHandle vh(0, vert_id);

        vf_pairs.insert(vh, fh);
    });

    problem.template add_interaction_term<Op::VF>([=] __device__(const auto& fh,
                                                                 const auto& vh,
                                                                 auto& iter,
                                                                 auto& obj) {
        using ActiveT = ACTIVE_TYPE(fh);

        assert(fh.local_id() == face_id);
        assert(vh.local_id() == vert_id);

        Eigen::Vector3<ActiveT> x0 = iter_val<ActiveT, 3>(fh, vh, iter, obj, 0);
        Eigen::Vector3<ActiveT> x1 = iter_val<ActiveT, 3>(fh, vh, iter, obj, 1);
        Eigen::Vector3<ActiveT> x2 = iter_val<ActiveT, 3>(fh, vh, iter, obj, 2);
        Eigen::Vector3<ActiveT> x3 = iter_val<ActiveT, 3>(fh, vh, iter, obj, 3);

        ActiveT E;
        return E;
    });
}

TEST(Diff, VFInteraction)
{
    using T = float;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "cube.obj");

    constexpr int VariableDim = 3;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    int expected_vv_candidate_pairs = 0;
    int expected_vf_candidate_pairs = 1;

    ProblemT problem(
        rx, true, expected_vv_candidate_pairs, expected_vf_candidate_pairs);


    int prev_nnz = problem.hess->non_zeros();

    // problem.hess->reset(0, HOST);
    // problem.hess->move(DEVICE, HOST);
    problem.hess->to_file("old_hess");

    int face_id(7), vert_id(4);

    add_vf_term(problem, face_id, vert_id);

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    problem.update_hessian();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    problem.eval_terms();

    problem.eval_terms_passive();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int new_nnz = problem.hess->non_zeros();

    EXPECT_GT(new_nnz, prev_nnz);

    problem.hess->reset(1, DEVICE);
    problem.hess->move(DEVICE, HOST);
    problem.hess->to_file("nnnew_hess");

    // polyscope::show();
}