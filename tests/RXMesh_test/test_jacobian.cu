#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_vector_problem.h"
#include "rxmesh/diff/jacobian_sparse_matrix.h"


using namespace rxmesh;

TEST(Diff, Jacobian)
{
    using T = float;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    std::vector<Op> ops{Op::EF, Op::F};

    std::vector<BlockShape> block_shapes{{5, 1}, {3, 1}};

    JacobianSparseMatrix<T> jac(rx, ops, block_shapes);

    RXMESH_INFO("vertices = {}", rx.get_num_vertices());
    RXMESH_INFO("edges = {}", rx.get_num_edges());
    RXMESH_INFO("face = {}", rx.get_num_faces());
    RXMESH_INFO("nnz = {}", jac.non_zeros());
    RXMESH_INFO("rows = {}", jac.rows());
    RXMESH_INFO("cols = {}", jac.cols());

    EXPECT_EQ(jac.cols(), rx.get_num_faces() * block_shapes[0].y);

    EXPECT_EQ(jac.rows(),
              rx.get_num_edges() * block_shapes[0].x +
                  rx.get_num_faces() * block_shapes[1].x);

    EXPECT_EQ(jac.get_num_terms(), 2);

    EXPECT_EQ(jac.get_term_num_rows(0), rx.get_num_edges() * block_shapes[0].x);

    EXPECT_EQ(jac.get_term_num_rows(1), rx.get_num_faces() * block_shapes[1].x);
}

template <typename ProblemT>
void add_tet_vector_unary_term(ProblemT& problem)
{
    problem.template add_term<Op::T, 2>(
        [] __device__(const auto& th, auto& opt_var) {
            using ActiveT = ACTIVE_TYPE(th);

            ActiveT x = opt_var.template active<1>(th)[0];

            Eigen::Vector<ActiveT, 2> residual;
            residual[0] = x;
            residual[1] = 2 * x;
            return residual;
        });
}

template <Op op, typename ProblemT>
void add_tet_vector_query_term(ProblemT& problem)
{
    problem.template add_term<op, 2>(
        [] __device__(const auto& th, const auto& iter, auto& opt_var) {
            using ActiveT = ACTIVE_TYPE(th);

            ActiveT q = 0;
            for (int i = 0; i < iter.size(); ++i) {
                q += opt_var.template active<1>(th, iter, i)[0];
            }

            Eigen::Vector<ActiveT, 2> residual;
            residual[0] = q;
            residual[1] = 2 * q;
            return residual;
        });
}

template <Op op, typename HandleT, int Valence>
void test_tet_vector_term(RXMeshStatic& rx)
{
    using T        = float;
    using ProblemT = DiffVectorProblem<T, 1, HandleT>;

    ProblemT problem(rx);
    problem.opt_var->reset(1, LOCATION_ALL);

    if constexpr (op == Op::T) {
        add_tet_vector_unary_term(problem);
    } else {
        add_tet_vector_query_term<op>(problem);
    }

    ASSERT_EQ(problem.terms.size(), 1);

    problem.eval_terms_sum_of_squares();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const int num_tets = rx.get_num_tets();
    const int num_vars = rx.get_num_elements<HandleT>();

    EXPECT_EQ(problem.jac->rows(), 2 * num_tets);
    EXPECT_EQ(problem.jac->cols(), num_vars);
    EXPECT_EQ(problem.jac->non_zeros(), 2 * Valence * num_tets);
    EXPECT_EQ(problem.jac->get_num_terms(), 1);
    EXPECT_EQ(problem.jac->get_term_rows_range(0),
              std::make_pair(0, 2 * num_tets));

    problem.residual->move(DEVICE, HOST);
    problem.grad->move(DEVICE, HOST);
    problem.jac->move(DEVICE, HOST);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (int t = 0; t < num_tets; ++t) {
        EXPECT_EQ((*problem.residual)(2 * t, 0), T(Valence));
        EXPECT_EQ((*problem.residual)(2 * t + 1, 0), T(2 * Valence));
    }

    for (int row = 0; row < problem.jac->rows(); ++row) {
        EXPECT_EQ(problem.jac->non_zeros(row), Valence);
    }

    problem.jac->for_each([&](int row, int, T value) {
        EXPECT_EQ(value, row % 2 == 0 ? T(1) : T(2));
    });

    T grad_sum = 0;
    for (int i = 0; i < num_vars; ++i) {
        EXPECT_GT((*problem.grad)(i, 0), 0);
        grad_sum += (*problem.grad)(i, 0);
    }
    EXPECT_EQ(grad_sum, T(5 * Valence * Valence * num_tets));

    for (auto& residual : problem.residual_reshaped) {
        residual.release();
    }
    problem.residual->release();
    problem.grad->release();
    problem.jac->release();
    problem.terms.clear();
    rx.remove_attribute(problem.opt_var.get());
    problem.opt_var.reset();
}

TEST(Diff, TetJacobian)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "car.msh");
    ASSERT_GT(rx.get_num_patches(), 1);

    test_tet_vector_term<Op::T, TetHandle, 1>(rx);
    test_tet_vector_term<Op::TV, VertexHandle, 4>(rx);
    test_tet_vector_term<Op::TE, EdgeHandle, 6>(rx);
    test_tet_vector_term<Op::TF, FaceHandle, 4>(rx);
}
