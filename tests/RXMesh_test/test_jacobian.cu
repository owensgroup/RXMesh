#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/jacobian_sparse_matrix.h"


using namespace rxmesh;

TEST(Diff, Jacobian)
{
    using T = float;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    std::vector<Op> ops{Op::EF, Op::F};

    std::vector<detail::BlockDim> block_dims{{5, 1}, {3, 1}};

    JacobianSparseMatrix<T> jac(rx, ops, block_dims);

    RXMESH_INFO("vertices = {}", rx.get_num_vertices());
    RXMESH_INFO("edges = {}", rx.get_num_edges());
    RXMESH_INFO("face = {}", rx.get_num_faces());
    RXMESH_INFO("nnz = {}", jac.non_zeros());
    RXMESH_INFO("rows = {}", jac.rows());
    RXMESH_INFO("cols = {}", jac.cols());

    EXPECT_EQ(jac.cols(), rx.get_num_faces() * block_dims[0].y);

    EXPECT_EQ(jac.rows(),
              rx.get_num_edges() * block_dims[0].x +
                  rx.get_num_faces() * block_dims[1].x);

    EXPECT_EQ(jac.get_num_terms(), 2);

    EXPECT_EQ(jac.get_term_num_rows(0), rx.get_num_edges() * block_dims[0].x);

    EXPECT_EQ(jac.get_term_num_rows(1), rx.get_num_faces() * block_dims[1].x);
}