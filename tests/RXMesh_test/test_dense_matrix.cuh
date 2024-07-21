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