#include "gtest/gtest.h"

#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/kernels/for_each.cuh"

#include <Eigen/Dense>

template <uint32_t blockThreads>
__global__ static void eigen_norm(const rxmesh::Context                 context,
                                  const rxmesh::VertexAttribute<double> in_attr,
                                  rxmesh::VertexAttribute<double> out_attr)
{
    using namespace rxmesh;

    auto normalize = [&](VertexHandle& vh) {
        Eigen::Vector3d attr;

        attr << in_attr(vh, 0), in_attr(vh, 1), in_attr(vh, 2);

        out_attr(vh) = attr.norm();
    };

    for_each<Op::V, blockThreads>(context, normalize);
}


TEST(Attribute, Eigen)
{
    using namespace rxmesh;

    cuda_query(0);

    std::string obj_path = STRINGIFY(INPUT_DIR) "dragon.obj";

    RXMeshStatic rx(obj_path);

    auto in_attr = *rx.add_vertex_attribute<double>("vAttrIn", 3);

    auto out_attr = *rx.add_vertex_attribute<double>("vAttrOut", 1);

    rx.for_each_vertex(HOST, [&](VertexHandle vh) {
        for (int i = 0; i < 3; ++i) {
            in_attr(vh, i) = rand() % rx.get_num_vertices();
        }
    });

    in_attr.move(HOST, DEVICE);

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> launch_box;

    rx.prepare_launch_box({}, launch_box, (void*)eigen_norm<blockThreads>);

    eigen_norm<blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), in_attr, out_attr);

    cudaDeviceSynchronize();

    out_attr.move(DEVICE, HOST);

    rx.for_each_vertex(HOST, [&](VertexHandle vh) {
        double n = in_attr(vh, 0) * in_attr(vh, 0) +
                   in_attr(vh, 1) * in_attr(vh, 1) +
                   in_attr(vh, 2) * in_attr(vh, 2);
        n = std::sqrt(n);

        EXPECT_NEAR(n, out_attr(vh), 0.0001);
    });
}
