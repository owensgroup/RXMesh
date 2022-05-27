#include <functional>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh_test.h"

#include "query.cuh"

TEST(RXMeshStatic, Oriented_VV_Open)
{
    using namespace rxmesh;

    // Select device
    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(
        import_obj(STRINGIFY(INPUT_DIR) "plane.obj", Verts, Faces, true));

    // RXMesh
    RXMeshStatic rxmesh(Faces, rxmesh_args.quite);

    EXPECT_FALSE(rxmesh.is_closed())
        << " Expected input to be open with boundaries";

    auto coordinates = rxmesh.add_vertex_attribute<dataT>(Verts, "coordinates");

    // input/output container
    auto input  = rxmesh.add_vertex_attribute<VertexHandle>("input", 1);
    auto output = rxmesh.add_vertex_attribute<VertexHandle>(
        "output", rxmesh.get_input_max_valence());

    input->reset(VertexHandle(), rxmesh::DEVICE);
    output->reset(VertexHandle(), rxmesh::DEVICE);

    // launch box
    constexpr uint32_t      blockThreads = 320;
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box(
        {Op::VV},
        launch_box,
        (void*)query_kernel<blockThreads,
                            Op::VV,
                            VertexHandle,
                            VertexHandle,
                            VertexAttribute<VertexHandle>,
                            VertexAttribute<VertexHandle>>,
        true);

    // query
    query_kernel<blockThreads, Op::VV, VertexHandle, VertexHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh.get_context(), *input, *output, true);

    CUDA_ERROR(cudaDeviceSynchronize());

    // move containers to the CPU for testing
    output->move(rxmesh::DEVICE, rxmesh::HOST);
    input->move(rxmesh::DEVICE, rxmesh::HOST);

    RXMeshTest tester(rxmesh, Faces, rxmesh_args.quite);
    EXPECT_TRUE(tester.run_test(rxmesh, Faces, *input, *output));

    auto vector_length = [](const dataT x, const dataT y, const dataT z) {
        return std::sqrt(x * x + y * y + z * z);
    };

    auto dot = [](const std::vector<dataT>& u, const std::vector<dataT>& v) {
        return std::inner_product(
            std::begin(u), std::end(u), std::begin(v), 0.0);
    };

    rxmesh.for_each_vertex(HOST, [&](const VertexHandle& vertex) {
        // the vertex 4 is the center vertex that is not boundary vertex
        // and the sum angle around it is 360
        if (rxmesh.map_to_global(vertex) != 4) {

            dataT sum_angles = 0;

            // 2 since every vertex is connected to three vertices (except
            // vertex 4 which is connected to 8 other vertices but we don't
            // check on it)
            for (uint32_t i = 0; i < 2; ++i) {

                uint32_t j = i + 1;

                auto v_0 = (*output)(vertex, i);
                auto v_1 = (*output)(vertex, j);

                if (v_1.is_valid() && v_0.is_valid()) {

                    std::vector<dataT> p1{
                        (*coordinates)(vertex, 0) - (*coordinates)(v_0, 0),
                        (*coordinates)(vertex, 1) - (*coordinates)(v_0, 1),
                        (*coordinates)(vertex, 2) - (*coordinates)(v_0, 2)};

                    std::vector<dataT> p2{
                        (*coordinates)(vertex, 0) - (*coordinates)(v_1, 0),
                        (*coordinates)(vertex, 1) - (*coordinates)(v_1, 1),
                        (*coordinates)(vertex, 2) - (*coordinates)(v_1, 2)};

                    dataT dot_pro = dot(p1, p2);
                    dataT angle   = std::acos(
                        dot_pro / (vector_length(p1[0], p1[1], p1[2]) *
                                   vector_length(p2[0], p2[1], p2[2])));
                    sum_angles += (angle * 180) / 3.14159265;
                }
            }

            EXPECT_TRUE(std::abs(sum_angles - 90) < 0.0001 ||
                        std::abs(sum_angles - 180) < 0.0001);
        }
    });
}

TEST(RXMeshStatic, Oriented_VV_Closed)
{
    using namespace rxmesh;

    // Select device
    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(
        import_obj(STRINGIFY(INPUT_DIR) "cube.obj", Verts, Faces, true));

    // RXMesh
    RXMeshStatic rxmesh(Faces, rxmesh_args.quite);

    EXPECT_TRUE(rxmesh.is_closed())
        << " Expecting input to be closed without boundaries";

    auto coordinates = rxmesh.add_vertex_attribute<dataT>(Verts, "coordinates");

    // input/output container
    auto input  = rxmesh.add_vertex_attribute<VertexHandle>("input", 1);
    auto output = rxmesh.add_vertex_attribute<VertexHandle>(
        "output", rxmesh.get_input_max_valence());

    input->reset(VertexHandle(), rxmesh::DEVICE);
    output->reset(VertexHandle(), rxmesh::DEVICE);

    // launch box
    constexpr uint32_t      blockThreads = 320;
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box(
        {Op::VV},
        launch_box,
        (void*)query_kernel<blockThreads,
                            Op::VV,
                            VertexHandle,
                            VertexHandle,
                            VertexAttribute<VertexHandle>,
                            VertexAttribute<VertexHandle>>,
        true);

    // query
    query_kernel<blockThreads, Op::VV, VertexHandle, VertexHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh.get_context(), *input, *output, true);

    CUDA_ERROR(cudaDeviceSynchronize());

    // move containers to the CPU for testing
    output->move(rxmesh::DEVICE, rxmesh::HOST);
    input->move(rxmesh::DEVICE, rxmesh::HOST);

    RXMeshTest tester(rxmesh, Faces, rxmesh_args.quite);
    EXPECT_TRUE(tester.run_test(rxmesh, Faces, *input, *output));


    // Make sure orientation is accurate
    // for the cube, all angle are either 45 or 90

    auto vector_length = [](const dataT x, const dataT y, const dataT z) {
        return std::sqrt(x * x + y * y + z * z);
    };

    auto dot = [](const std::vector<dataT>& u, const std::vector<dataT>& v) {
        return std::inner_product(
            std::begin(u), std::end(u), std::begin(v), 0.0);
    };

    rxmesh.for_each_vertex(HOST, [&](const VertexHandle& vertex) {
        for (uint32_t i = 0; i < (*output).get_num_attributes(); ++i) {

            uint32_t j = (i + 1) % output->get_num_attributes();

            auto v_0 = (*output)(vertex, i);
            auto v_1 = (*output)(vertex, j);

            if (v_1.is_valid() && v_0.is_valid()) {

                std::vector<dataT> p1{
                    (*coordinates)(vertex, 0) - (*coordinates)(v_0, 0),
                    (*coordinates)(vertex, 1) - (*coordinates)(v_0, 1),
                    (*coordinates)(vertex, 2) - (*coordinates)(v_0, 2)};

                std::vector<dataT> p2{
                    (*coordinates)(vertex, 0) - (*coordinates)(v_1, 0),
                    (*coordinates)(vertex, 1) - (*coordinates)(v_1, 1),
                    (*coordinates)(vertex, 2) - (*coordinates)(v_1, 2)};

                dataT dot_pro = dot(p1, p2);
                dataT theta =
                    std::acos(dot_pro / (vector_length(p1[0], p1[1], p1[2]) *
                                         vector_length(p2[0], p2[1], p2[2])));
                theta = (theta * 180) / 3.14159265;
                EXPECT_TRUE(std::abs(theta - 90) < 0.0001 ||
                            std::abs(theta - 45) < 0.0001);
            }
        }
    });
}