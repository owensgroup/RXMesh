#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.cuh"

TEST(RXMeshStatic, DenseMatrixASum)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<float> mat(rx, 10, 10);

    mat.fill_random();

    float a_sum = mat.abs_sum();

    float res = 0;

    for (uint32_t i = 0; i < mat.rows(); ++i) {
        for (uint32_t j = 0; j < mat.cols(); ++j) {
            res += std::abs(mat(i, j));
        }
    }

    EXPECT_NEAR(res, a_sum, 0.001);

    mat.release();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}


TEST(RXMeshStatic, DenseMatrixAXPY)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<float> Y(rx, 10, 10);
    DenseMatrix<float> X(rx, 10, 10);

    DenseMatrix<float> Y_copy(rx, 10, 10);
    DenseMatrix<float> X_copy(rx, 10, 10);

    Y.fill_random();
    X.fill_random();

    Y_copy.copy_from(Y, HOST, HOST);
    X_copy.copy_from(X, HOST, HOST);

    Y.axpy(X, 0.5f);

    Y.move(DEVICE, HOST);

    for (uint32_t i = 0; i < Y.rows(); ++i) {
        for (uint32_t j = 0; j < Y.cols(); ++j) {
            EXPECT_NEAR(Y_copy(i, j) + 0.5 * X_copy(i, j), Y(i, j), 0.001);
        }
    }


    X.release();
    Y.release();
    Y_copy.release();
    X_copy.release();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}