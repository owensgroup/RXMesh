#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.h"

#include <vector>

template <typename MatT>
void write_dense_device_view(rxmesh::RXMeshStatic& rx, MatT view)
{
    using namespace rxmesh;
    rx.for_each_vertex(DEVICE,
                       [view] __device__(const VertexHandle vh) mutable {
                           view(vh, 0) = 4.0f;
                           view(vh, 1) = 5.0f;
                           view(vh, 2) = 6.0f;
                       });
}

TEST(RXMeshStatic, DenseMatrixToEigen)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<float> rx_mat(rx, 10, 10, LOCATION_ALL);
    DenseMatrix<float> rx_mat_copy(rx, 10, 10, LOCATION_ALL);

    rx_mat.fill_random();
    rx_mat_copy.copy_from(rx_mat, HOST, HOST);

    auto eigen_mat = rx_mat.to_eigen();

    // ensure that the content of Eigen matrix is the same as the RXMesh
    // DenseMatrix
    for (int i = 0; i < rx_mat.rows(); ++i) {
        for (int j = 0; j < rx_mat.cols(); ++j) {
            EXPECT_NEAR(rx_mat(i, j), eigen_mat(i, j), 0.0000001);
        }
    }

    // ensure operations done on the Eigen matrix is reflected on RXMesh
    // DenseMatrix
    const float scalar = 5.f;
    eigen_mat *= scalar;

    for (int i = 0; i < rx_mat.rows(); ++i) {
        for (int j = 0; j < rx_mat.cols(); ++j) {
            EXPECT_NEAR(rx_mat_copy(i, j), rx_mat(i, j) / scalar, 0.0000001);
        }
    }

    rx_mat.release();
    rx_mat_copy.release();
}

TEST(RXMeshStatic, DenseMatrixASum)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<float> mat(rx, 10, 10, LOCATION_ALL);

    mat.fill_random();

    float a_sum = mat.abs_sum();

    float res = 0;

    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
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

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<float> Y(rx, 10, 10, LOCATION_ALL);
    DenseMatrix<float> X(rx, 10, 10, LOCATION_ALL);

    DenseMatrix<float> Y_copy(rx, 10, 10, LOCATION_ALL);
    DenseMatrix<float> X_copy(rx, 10, 10, LOCATION_ALL);

    Y.fill_random();
    X.fill_random();

    Y_copy.copy_from(Y, HOST, HOST);
    X_copy.copy_from(X, HOST, HOST);

    Y.axpy(X, 0.5f);

    Y.move(DEVICE, HOST);

    for (int i = 0; i < Y.rows(); ++i) {
        for (int j = 0; j < Y.cols(); ++j) {
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

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<cuComplex> y(rx, 10, 10, LOCATION_ALL);
    y.fill_random();

    DenseMatrix<cuComplex> x(rx, 10, 10, LOCATION_ALL);
    x.fill_random();

    cuComplex dot_res = y.dot(x);

    cuComplex res = make_cuComplex(0.f, 0.f);


    for (int i = 0; i < y.rows(); ++i) {
        for (int j = 0; j < y.cols(); ++j) {
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

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<cuComplex> x(rx, 10, 10, LOCATION_ALL);
    x.fill_random();

    float norm2_res = x.norm2();

    float res = 0.f;

    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {

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

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<cuComplex> x(rx, 10, 10, LOCATION_ALL);
    DenseMatrix<cuComplex> copy(rx, 10, 10, LOCATION_ALL);

    x.fill_random();

    copy.copy_from(x, HOST, HOST);

    float scalar = 5.0f;

    x.multiply(scalar);

    x.move(DEVICE, HOST);

    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {

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

TEST(RXMeshStatic, DenseMatrixSwap)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<float> x(rx, 10, 10, LOCATION_ALL);
    DenseMatrix<float> copy(rx, 10, 10, LOCATION_ALL);
    x.fill_random();

    copy.copy_from(x, HOST, HOST);

    DenseMatrix<float> y(rx, 10, 10, LOCATION_ALL);
    y.fill_random();

    x.swap(y);

    x.move(DEVICE, HOST);
    y.move(DEVICE, HOST);

    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {

            EXPECT_NEAR(y(i, j), copy(i, j), 0.001);
        }
    }


    x.release();
    y.release();
    copy.release();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}


TEST(RXMeshStatic, DenseMatrixUserManaged)
{
    using namespace rxmesh;
    using T = float;

    int rows = 10;
    int cols = 3;

    std::vector<T> h_ptr(rows * cols);

    fill_with_random_numbers(h_ptr.data(), rows * cols);

    T* d_ptr(nullptr);
    CUDA_ERROR(cudaMalloc((void**)&d_ptr, sizeof(T) * rows * cols));

    DenseMatrix<T> mat(rows, cols, d_ptr, h_ptr.data());

    mat.move(HOST, DEVICE);

    T norm2_res = mat.norm2();

    T res = 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            res += h_ptr[i * cols + j] * h_ptr[i * cols + j];
        }
    }

    EXPECT_NEAR(norm2_res, std::sqrt(res), 1e-5);

    mat.release();
    GPU_FREE(d_ptr);

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(RXMeshStatic, DenseMatrixSlice)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    DenseMatrix<float> mat(rx, 10, 3, LOCATION_ALL);

    mat.fill_random();

    for (int j = 0; j < mat.cols(); ++j) {

        auto col = mat.col(j);

        for (int i = 0; i < mat.rows(); ++i) {

            EXPECT_NEAR(mat(i, j), col(i), 0.0000001);
        }
        col.release();
    }

    mat.release();
}

TEST(RXMeshStatic, DenseMatrixDeviceView)
{
    using namespace rxmesh;
    using MatT = DenseMatrix<float, Eigen::RowMajor>;

    RXMeshStatic  rx(STRINGIFY(INPUT_DIR) "sphere3.obj");
    const int     rows  = static_cast<int>(rx.get_num_vertices());
    constexpr int cols  = 3;
    const size_t  bytes = size_t(rows) * cols * sizeof(float);

    float* device_ptr = nullptr;
    CUDA_ERROR(cudaMalloc(&device_ptr, bytes));

    auto view = MatT::device_view(rx, rows, cols, device_ptr);
    EXPECT_EQ(view.data(DEVICE), device_ptr);
    EXPECT_EQ(view.get_allocated() & DEVICE, DEVICE);
    EXPECT_EQ(view.get_allocated() & HOST, LOCATION_NONE);
    EXPECT_TRUE(view.has_compatible_context<VertexHandle>(rx));
    EXPECT_FALSE(view.has_library_resources());

    write_dense_device_view(rx, view);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> values(size_t(rows) * cols);
    ASSERT_EQ(
        cudaMemcpy(values.data(), device_ptr, bytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int row = 0; row < rows; ++row) {
        EXPECT_FLOAT_EQ(values[size_t(row) * cols], 4.0f);
        EXPECT_FLOAT_EQ(values[size_t(row) * cols + 1], 5.0f);
        EXPECT_FLOAT_EQ(values[size_t(row) * cols + 2], 6.0f);
    }

    view.release();
    view.release();
    EXPECT_FALSE(view.has_library_resources());
    EXPECT_EQ(cudaMemset(device_ptr, 0, bytes), cudaSuccess);
    GPU_FREE(device_ptr);
}

TEST(RXMeshStatic, DenseMatrixUserManagedAllocationFlags)
{
    using namespace rxmesh;

    float* device_ptr = nullptr;
    CUDA_ERROR(cudaMalloc(&device_ptr, 12 * sizeof(float)));

    DenseMatrix<float> device_only(4, 3, device_ptr, nullptr);
    EXPECT_EQ(device_only.get_allocated() & DEVICE, DEVICE);
    EXPECT_EQ(device_only.get_allocated() & HOST, LOCATION_NONE);
    EXPECT_TRUE(device_only.has_library_resources());
    device_only.release();
    device_only.release();
    EXPECT_FALSE(device_only.has_library_resources());

    EXPECT_EQ(cudaMemset(device_ptr, 0, 12 * sizeof(float)), cudaSuccess);
    GPU_FREE(device_ptr);

    std::vector<float> host_values(12, 2.0f);
    DenseMatrix<float> host_only(4, 3, nullptr, host_values.data());
    EXPECT_EQ(host_only.get_allocated() & DEVICE, LOCATION_NONE);
    EXPECT_EQ(host_only.get_allocated() & HOST, HOST);
    EXPECT_FALSE(host_only.has_library_resources());
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    host_only.release();
    host_only.release();
    EXPECT_FALSE(host_only.has_library_resources());
    EXPECT_FLOAT_EQ(host_values.front(), 2.0f);
}
