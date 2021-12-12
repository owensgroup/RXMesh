#include "gtest/gtest.h"
#include "higher_query.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh_test.h"

TEST(RXMeshStatic, DISABLED_HigherQueries)
{
    using namespace rxmesh;


    // Select device
    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;
    ASSERT_TRUE(import_obj(
        STRINGIFY(INPUT_DIR) "sphere3.obj", Verts, Faces, rxmesh_args.quite));

    // RXMesh
    RXMeshStatic rxmesh(Faces, rxmesh_args.quite);

    uint32_t input_size = rxmesh.get_num_vertices();

    // input/output container
    auto input = rxmesh.add_vertex_attribute<VertexHandle>("input", 1);

    // we assume that every vertex could store up to num_vertices as its
    // neighbor vertices which is a bit excessive
    auto output =
        rxmesh.add_vertex_attribute<VertexHandle>("output", input_size);

    rxmesh.for_each_vertex(HOST, [&](const VertexHandle& handle) {
        (*input)(handle) = VertexHandle();
        for (uint32_t j = 0; j < (*output).get_num_attributes(); ++j) {
            (*output)(handle, j) = VertexHandle();
        }
    });

    output->move(rxmesh::HOST, rxmesh::DEVICE);
    input->move(rxmesh::HOST, rxmesh::DEVICE);

    // launch box
    constexpr uint32_t      blockThreads = 512;
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box(Op::VV, launch_box, true, false);


    RXMeshTest tester(rxmesh, Faces, true);

    // launch
    higher_query<blockThreads, Op::VV>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh.get_context(), *input, *output);

    CUDA_ERROR(cudaDeviceSynchronize());

    // move containers to the CPU for testing
    output->move(rxmesh::DEVICE, rxmesh::HOST);
    input->move(rxmesh::DEVICE, rxmesh::HOST);

    // verify
    EXPECT_TRUE(tester.run_test(rxmesh, Faces, *input, *output, true));
}