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

TEST(RXMeshStatic, DenseMatrixDot)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<cuComplex> y(rx, 10, 10);
    y.fill_random();

    DenseMatrix<cuComplex> x(rx, 10, 10);
    x.fill_random();

    cuComplex dot_res = y.dot(x);

    cuComplex res = make_cuComplex(0.f, 0.f);


    for (uint32_t i = 0; i < y.rows(); ++i) {
        for (uint32_t j = 0; j < y.cols(); ++j) {
            // for complex number (rx, ix) and (ry+iy), the result of the
            // multiplication is (rx.ry-ix.iy) + i(rx.iy + ix.ry)

            cuComplex x_val = x(i, j);
            cuComplex y_val = y(i, j);

            res.x += x_val.x * y_val.x - x_val.y * y_val.y;
            res.y += x_val.x * y_val.y + x_val.y * y_val.x;
        }
    }

    EXPECT_NEAR(res.x, dot_res.x, 0.001);
    EXPECT_NEAR(res.y, dot_res.y, 0.001);

    y.release();
    x.release();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}


TEST(RXMeshStatic, DenseMatrixNorm2)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<cuComplex> x(rx, 10, 10);
    x.fill_random();

    float norm2_res = x.norm2();

    float res = 0.f;

    for (uint32_t i = 0; i < x.rows(); ++i) {
        for (uint32_t j = 0; j < x.cols(); ++j) {

            cuComplex x_val = x(i, j);

            res += x_val.x * x_val.x + x_val.y * x_val.y;
        }
    }

    EXPECT_NEAR(norm2_res, std::sqrt(res), 0.001);

    x.release();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}


TEST(RXMeshStatic, DenseMatrixMulitply)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<cuComplex> x(rx, 10, 10);
    DenseMatrix<cuComplex> copy(rx, 10, 10);

    x.fill_random();

    copy.copy_from(x, HOST, HOST);

    float scalar = 5.0f;

    x.multiply(scalar);

    x.move(DEVICE, HOST);

    for (uint32_t i = 0; i < x.rows(); ++i) {
        for (uint32_t j = 0; j < x.cols(); ++j) {

            cuComplex x_val = x(i, j);

            cuComplex res = copy(i, j);
            res.x *= scalar;
            res.y *= scalar;

            EXPECT_NEAR(res.x, x_val.x, 0.001);
            EXPECT_NEAR(res.y, x_val.y, 0.001);
        }
    }


    x.release();
    copy.release();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}