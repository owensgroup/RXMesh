#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"

#include "rxmesh/query.cuh"


#include "rxmesh/matrix/cg_mat_free_solver.h"
#include "rxmesh/matrix/cg_solver.h"
#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/matrix/lu_solver.h"
#include "rxmesh/matrix/pcg_solver.h"
#include "rxmesh/matrix/qr_solver.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

using namespace rxmesh;

template <typename T, uint32_t blockThreads>
__global__ static void matvec(const Context            context,
                              const VertexAttribute<T> coords,
                              const SparseMatrix<T>    A,
                              const DenseMatrix<T>     in,
                              DenseMatrix<T>           out,
                              const T                  factor1,
                              const T                  factor2,
                              const T                  factor3,
                              const T                  factor4)
{
    using namespace rxmesh;

    int num_cols = out.cols();

    auto set = [&](VertexHandle& v_row, const VertexIterator& iter) {
        T sum_e_weight(0);

        int row_id = context.linear_id(v_row);

        T v_weight = iter.size();


        for (int i = 0; i < num_cols; ++i) {
            out(row_id, i) = 0;         
        }


        for (uint32_t v = 0; v < iter.size(); ++v) {

            VertexHandle v_col = iter[v];

            int col_id = context.linear_id(v_col);

            T val = A(v_row, v_col);
            for (int i = 0; i < num_cols; ++i) {
                out(row_id, i) += in(col_id, i) * val;
            }
            sum_e_weight += 1;
        }

        T diag = v_weight + sum_e_weight + factor4;

        for (int i = 0; i < num_cols; ++i) {
            out(row_id, i) += diag * in(row_id, i);
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, set);
}

template <typename T, uint32_t blockThreads>
__global__ static void setup(const Context      context,
                             VertexAttribute<T> coords,
                             SparseMatrix<T>    A,
                             DenseMatrix<T>     X,
                             DenseMatrix<T>     B,
                             T                  factor1,
                             T                  factor2,
                             T                  factor3,
                             T                  factor4)
{
    using namespace rxmesh;
    auto set = [&](VertexHandle& v_id, const VertexIterator& iter) {
        T sum_e_weight(0);

        T v_weight = iter.size();

        B(v_id, 0) = iter.size() * factor1;
        B(v_id, 1) = iter.size() * factor2;
        B(v_id, 2) = iter.size() * factor3;

        X(v_id, 0) = coords(v_id, 0) * v_weight;
        X(v_id, 1) = coords(v_id, 1) * v_weight;
        X(v_id, 2) = coords(v_id, 2) * v_weight;

        for (uint32_t v = 0; v < iter.size(); ++v) {

            A(v_id, iter[v]) = 1;

            sum_e_weight += 1;
        }

        A(v_id, v_id) = v_weight + sum_e_weight + factor4;
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, set);
}

template <typename T, typename SolverT>
void test_direct_solver(RXMeshStatic&    rx,
                        SolverT&         solver,
                        SparseMatrix<T>& A,
                        DenseMatrix<T>&  B,
                        DenseMatrix<T>&  X,
                        bool             sol_on_device,
                        T                factor1 = 7.4f,
                        T                factor2 = 2.6f,
                        T                factor3 = 10.3f,
                        T                factor4 = 100.f)
{
    rx.run_kernel<256>({Op::VV},
                       setup<T, 256>,
                       *rx.get_input_vertex_coordinates(),
                       A,
                       X,
                       B,
                       factor1,
                       factor2,
                       factor3,
                       factor4);

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
    DenseMatrix<float>  X(rx, num_vertices, 3);
    DenseMatrix<float>  B(rx, num_vertices, 3);

    CholeskySolver solver(&A);

    test_direct_solver(rx, solver, A, B, X, true);

    // testing resolving
    test_direct_solver(
        rx, solver, A, B, X, true, 2.888f, 55.109f, 3.464f, 70.f);


    A.release();
    X.release();
    B.release();
}


TEST(Solver, QR)
{

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix<float> A(rx, Op::VV);
    DenseMatrix<float>  X(rx, num_vertices, 3);
    DenseMatrix<float>  B(rx, num_vertices, 3);

    QRSolver solver(&A);

    test_direct_solver(rx, solver, A, B, X, true);

    // testing resolving
    // test_direct_solver(rx, solver, A, B, X,
    // true, 2.888f, 55.109f, 3.464f, 70.f);

    A.release();
    X.release();
    B.release();
}

TEST(Solver, LU)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix<float> A(rx, Op::VV);
    DenseMatrix<float>  X(rx, num_vertices, 3);
    DenseMatrix<float>  B(rx, num_vertices, 3);

    LUSolver solver(&A);

    test_direct_solver(rx, solver, A, B, X, false);

    A.release();
    X.release();
    B.release();
}


TEST(Solver, CompareEigen)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix<float> A(rx, Op::VV);
    DenseMatrix<float>  X(rx, num_vertices, 3);
    DenseMatrix<float>  B(rx, num_vertices, 3);

    CholeskySolver solver(&A);

    test_direct_solver(rx, solver, A, B, X, true);

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

template <typename T, typename SolverT>
void test_iterative_solver(RXMeshStatic&    rx,
                           SolverT&         solver,
                           SparseMatrix<T>& A,
                           DenseMatrix<T>&  B,
                           DenseMatrix<T>&  X)
{
    rx.run_kernel<256>({Op::VV},
                       setup<T, 256>,
                       *rx.get_input_vertex_coordinates(),
                       A,
                       X,
                       B,
                       7.4f,
                       2.6f,
                       10.3f,
                       100.f);

    solver.pre_solve(B, X);

    solver.solve(B, X);

    RXMESH_INFO(" iter taken = {}, final_res = {}",
                solver.iter_taken(),
                solver.final_residual());

    X.move(DEVICE, HOST);

    DenseMatrix<T> Ax(rx, A.rows(), X.cols());

    A.multiply(X, Ax);

    Ax.move(DEVICE, HOST);
    B.move(DEVICE, HOST);

    for (int i = 0; i < Ax.rows(); ++i) {
        for (int j = 0; j < Ax.cols(); ++j) {
            EXPECT_NEAR(Ax(i, j), B(i, j), 1e-3);
        }
    }

    Ax.release();
}

TEST(Solver, CG)
{

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    using T = float;

    SparseMatrix<T> A(rx, Op::VV);
    DenseMatrix<T>  X(rx, num_vertices, 3);
    DenseMatrix<T>  B(rx, num_vertices, 3);

    CGSolver solver(A, 3, 5000, T(1e-7));

    test_iterative_solver(rx, solver, A, B, X);

    A.release();
    X.release();
    B.release();
}


TEST(Solver, PCG)
{

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    using T = float;

    SparseMatrix<T> A(rx, Op::VV);
    DenseMatrix<T>  X(rx, num_vertices, 3);
    DenseMatrix<T>  B(rx, num_vertices, 3);

    PCGSolver solver(A, 3, 5000, T(1e-7));

    test_iterative_solver(rx, solver, A, B, X);

    A.release();
    X.release();
    B.release();
}

TEST(Solver, CGMatFree)
{

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    using T = float;

    SparseMatrix<T> A(rx, Op::VV);
    DenseMatrix<T>  X(rx, num_vertices, 3);
    DenseMatrix<T>  B(rx, num_vertices, 3);

    CGMatFreeSolver solver(num_vertices, 3, 5000, T(1e-7));

    solver.m_mat_vec = [&](const DenseMatrix<T>& in,
                           DenseMatrix<T>&       out,
                           cudaStream_t          stream) {
        rx.run_kernel<256>({Op::VV},
                           matvec<T, 256>,
                           *rx.get_input_vertex_coordinates(),
                           A,
                           in,
                           out,
                           7.4f,
                           2.6f,
                           10.3f,
                           100.f);        
    };    

    test_iterative_solver(rx, solver, A, B, X);

    A.release();
    X.release();
    B.release();
}