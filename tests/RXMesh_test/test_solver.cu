#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"

#include "rxmesh/query.cuh"


#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/matrix/lu_solver.h"
#include "rxmesh/matrix/qr_solver.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

using namespace rxmesh;

template <typename T, uint32_t blockThreads>
__global__ static void setup(const Context      context,
                             VertexAttribute<T> coords,
                             SparseMatrix<T>   A,
                             DenseMatrix<T>     X,
                             DenseMatrix<T>     B)
{
    using namespace rxmesh;
    auto set = [&](VertexHandle& v_id, const VertexIterator& iter) {
        T sum_e_weight(0);

        T v_weight = iter.size();

        B(v_id, 0) = iter.size() * 7.4f;
        B(v_id, 1) = iter.size() * 2.6f;
        B(v_id, 2) = iter.size() * 10.3f;

        X(v_id, 0) = coords(v_id, 0) * v_weight;
        X(v_id, 1) = coords(v_id, 1) * v_weight;
        X(v_id, 2) = coords(v_id, 2) * v_weight;

        for (uint32_t v = 0; v < iter.size(); ++v) {

            A(v_id, iter[v]) = 1;

            sum_e_weight += 1;
        }

        A(v_id, v_id) = v_weight + sum_e_weight + 1000000.f;
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, set);
}

template <typename T>
void setup_matrix(RXMeshStatic&     rx,
                  SparseMatrix<T>& A,
                  DenseMatrix<T>&   B,
                  DenseMatrix<T>&   X)
{
    constexpr uint32_t blockThreads = 256;

    auto coords = rx.get_input_vertex_coordinates();

    rx.run_kernel<blockThreads>(
        {Op::VV}, setup<T, blockThreads>, *coords, A, X, B);
}

template <typename T, typename SolverT>
void test_solver(RXMeshStatic&     rx,
                 SolverT&          solver,
                 SparseMatrix<T>& A,
                 DenseMatrix<T>&   B,
                 DenseMatrix<T>&   X,
                 bool              sol_on_device)
{
    setup_matrix(rx, A, B, X);

    // Needed for LU but does not hurt other solvers
    A.move(DEVICE, HOST);
    B.move(DEVICE, HOST);

    solver.pre_solve(rx);

    solver.solve(B, X);

    if (sol_on_device) {
        X.move(DEVICE, HOST);
    } else {
        X.move(HOST, DEVICE);
    }

    DenseMatrix<T> Ax(rx, A.rows(), X.cols());

    A.multiply(X, Ax);

    Ax.move(DEVICE, HOST);


    for (int i = 0; i < Ax.rows(); ++i) {
        for (int j = 0; j < Ax.cols(); ++j) {
            EXPECT_NEAR(Ax(i, j), B(i, j), 1e-3);
        }
    }

    Ax.release();
}

TEST(Solver, Cholesky)
{

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix<float> A(rx, Op::VV);
    DenseMatrix<float>   X(rx, num_vertices, 3);
    DenseMatrix<float>   B(rx, num_vertices, 3);

    CholeskySolver solver(&A);

    test_solver(rx, solver, A, B, X, true);

    A.release();
    X.release();
    B.release();
}


TEST(Solver, QR)
{

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix<float> A(rx, Op::VV);
    DenseMatrix<float>   X(rx, num_vertices, 3);
    DenseMatrix<float>   B(rx, num_vertices, 3);

    QRSolver solver(&A);

    test_solver(rx, solver, A, B, X, true);

    A.release();
    X.release();
    B.release();
}


TEST(Solver, LU)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix<float> A(rx, Op::VV);
    DenseMatrix<float>   X(rx, num_vertices, 3);
    DenseMatrix<float>   B(rx, num_vertices, 3);

    LUSolver solver(&A);

    test_solver(rx, solver, A, B, X, false);

    A.release();
    X.release();
    B.release();
}


TEST(Solver, CompareEigen)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix<float> A(rx, Op::VV);
    DenseMatrix<float>   X(rx, num_vertices, 3);
    DenseMatrix<float>   B(rx, num_vertices, 3);

    CholeskySolver solver(&A);

    test_solver(rx, solver, A, B, X, true);

    DenseMatrix<float> X_copy(rx, num_vertices, 3);
    X_copy.copy_from(X, DEVICE, HOST);

    auto A_eigen = A.to_eigen();
    auto X_eigen = X.to_eigen();
    auto B_eigen = B.to_eigen();

    // Note: there is a bug with Eigen if we use the default reordering
    // which is Eigen::AMDOrdering<int>
    // (https://gitlab.com/libeigen/eigen/-/issues/2839)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>,
                          Eigen::UpLoType::Lower,
                          Eigen::COLAMDOrdering<int>>
        eigen_solver;

    eigen_solver.compute(A_eigen);
    X_eigen = eigen_solver.solve(B_eigen);

    for (int i = 0; i < X_copy.rows(); ++i) {
        for (int j = 0; j < X_copy.cols(); ++j) {
            EXPECT_NEAR(X_eigen(i, j), X_copy(i, j), 1e-5);
        }
    }

    A.release();
    X.release();
    B.release();
    X_copy.release();
}