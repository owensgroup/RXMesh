#include <cmath>
#include "gtest/gtest.h"

#include "rxmesh/geometry_util.cuh"
#include "rxmesh/rxmesh_static.h"

#include "query_kernel.cuh"

TEST(RXMeshStatic, EVDiamond)
{
    using namespace rxmesh;
        
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "plane_5.obj");

    // input/output container
    auto input  = *rx.add_edge_attribute<EdgeHandle>("input", 1);
    auto output = *rx.add_edge_attribute<VertexHandle>("output", 4);

    output.reset(VertexHandle(), DEVICE);

    // launch box
    constexpr uint32_t      blockThreads = 320;
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::EVDiamond},
                          launch_box,
                          (void*)query_kernel<blockThreads,
                                              Op::EVDiamond,
                                              EdgeHandle,
                                              VertexHandle,
                                              EdgeAttribute<EdgeHandle>,
                                              EdgeAttribute<VertexHandle>>);

    // query
    query_kernel<blockThreads, Op::EVDiamond, EdgeHandle, VertexHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), input, output);

    CUDA_ERROR(cudaDeviceSynchronize());

    // move containers to the CPU for testing
    output.move(DEVICE, HOST);
    input.move(DEVICE, HOST);

    auto coords = *rx.get_input_vertex_coordinates();

    rx.for_each_edge(HOST, [&](const EdgeHandle& eh) {
        /* 
              v3
            /   \
           /     \
          /       \
         v0--->---v2
          \       /
           \     /
            \   /
             v1
        */
        auto v0 = output(eh, 0);
        auto v1 = output(eh, 1);
        auto v2 = output(eh, 2);
        auto v3 = output(eh, 3);

        if (v0.is_valid() && v1.is_valid() && v2.is_valid() && v3.is_valid()) {

            glm::vec3 x0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
            glm::vec3 x1(coords(v1, 0), coords(v1, 1), coords(v1, 2));
            glm::vec3 x2(coords(v2, 0), coords(v2, 1), coords(v2, 2));
            glm::vec3 x3(coords(v3, 0), coords(v3, 1), coords(v3, 2));

            float t0 = tri_area(x0, x1, x2);
            float t1 = tri_area(x0, x2, x3);

            EXPECT_NEAR(t0 + t1, 1, 0.00001);
        }
    });
}