#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix2.h"

#include "rxmesh/query.cuh"


#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/matrix/lu_solver.h"
#include "rxmesh/matrix/qr_solver.h"

using namespace rxmesh;

template <typename T, uint32_t blockThreads>
__global__ static void setup(const Context      context,
                             VertexAttribute<T> coords,
                             SparseMatrix2<T>   A,
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
                  SparseMatrix2<T>& A,
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
                 SparseMatrix2<T>& A,
                 DenseMatrix<T>&   B,
                 DenseMatrix<T>&   X,
                 bool              sol_on_device)
{
    setup_matrix(rx, A, B, X);

    //Needed for LU but does not hurt other solvers 
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
}

TEST(RXMeshStatic, CholeskySolve)
{

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix2<float> A(rx, Op::VV);
    DenseMatrix<float>   X(rx, num_vertices, 3);
    DenseMatrix<float>   B(rx, num_vertices, 3);

    CholeskySolver solver(&A);

    test_solver(rx, solver, A, B, X, true);
}


TEST(RXMeshStatic, QRSolve)
{

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix2<float> A(rx, Op::VV);
    DenseMatrix<float>   X(rx, num_vertices, 3);
    DenseMatrix<float>   B(rx, num_vertices, 3);

    QRSolver solver(&A);

    test_solver(rx, solver, A, B, X, true);
}


TEST(RXMeshStatic, LUSolve)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix2<float> A(rx, Op::VV);
    DenseMatrix<float>   X(rx, num_vertices, 3);
    DenseMatrix<float>   B(rx, num_vertices, 3);

    LUSolver solver(&A);

    test_solver(rx, solver, A, B, X, false);
}