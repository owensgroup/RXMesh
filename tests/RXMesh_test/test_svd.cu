#include "gtest/gtest.h"

#include <cstdlib>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.h"

#include "rxmesh/util/svd3_cuda.h"

#include <Eigen/Dense>

template <uint32_t blockThreads>
__global__ static void test_svd(const rxmesh::Context                context,
                                const rxmesh::VertexAttribute<float> in_mat,
                                rxmesh::VertexAttribute<float>       out_mat)
{
    using namespace rxmesh;

    auto compute_svd = [&](VertexHandle& vh) {
        // input matrix
        Eigen::Matrix3f mat;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                mat(i, j) = in_mat(vh, i * 3 + j);
            }
        }
        Eigen::Matrix3f U;  // left singular vectors
        Eigen::Matrix3f V;  // right singular vectors
        Eigen::Vector3f S;  // singular values

        svd(mat, U, S, V);


        // reconstructed matrix from SVD
        Eigen::Matrix3f recon = U * S.asDiagonal() * V.transpose();

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                out_mat(vh, i * 3 + j) = recon(i, j);
            }
        }
    };

    for_each<Op::V, blockThreads>(context, compute_svd);
}


TEST(Util, SVD)
{
    using namespace rxmesh;

    std::string obj_path = STRINGIFY(INPUT_DIR) "sphere3.obj";

    RXMeshStatic rx(obj_path);

    // every vertex is assigned to a 3x3 matrix i.e., 9 attributes
    auto in_mat = *rx.add_vertex_attribute<float>("vAttrIn", 9);

    auto out_mat = *rx.add_vertex_attribute<float>("vAttrOut", 9);

    rx.for_each_vertex(HOST, [&](VertexHandle vh) {
        for (int i = 0; i < 9; ++i) {
            in_mat(vh, i) = (float(std::rand()) / float(RAND_MAX));
        }
    });

    in_mat.move(HOST, DEVICE);

    out_mat.reset(0.f, DEVICE);

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> launch_box;

    rx.prepare_launch_box({}, launch_box, (void*)test_svd<blockThreads>);

    test_svd<blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), in_mat, out_mat);

    CUDA_ERROR(cudaDeviceSynchronize());

    out_mat.move(DEVICE, HOST);

    rx.for_each_vertex(
        HOST,
        [&](VertexHandle vh) {
            for (int i = 0; i < 9; ++i) {
                EXPECT_LT(std::abs(in_mat(vh, i) - out_mat(vh, i)), 0.01);
            }
        },
        NULL,
        false);
}
