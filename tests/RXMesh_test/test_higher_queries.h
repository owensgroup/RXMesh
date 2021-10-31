#include "gtest/gtest.h"
#include "higher_query.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh_test.h"

using namespace rxmesh;

TEST(RXMeshStatic, HigherQueries)
{
    // Select device
    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;
    ASSERT_TRUE(
        import_obj(STRINGIFY(INPUT_DIR) "sphere3.obj", Verts, Faces, true));

    // RXMesh
    RXMeshStatic rxmesh_static(Faces, rxmesh_args.quite);

    uint32_t input_size = rxmesh_static.get_num_vertices();

    // input/output container
    auto input_container = rxmesh_static.add_vertex_attribute<uint32_t>(
        "input", 1u, rxmesh::LOCATION_ALL, rxmesh::AoS, false, false);

    auto output_container = rxmesh_static.add_vertex_attribute<uint32_t>(
        "output", input_size, rxmesh::LOCATION_ALL, rxmesh::SoA, false, false);
    //           ^^that is a bit excessive

    // launch box
    constexpr uint32_t      blockThreads = 512;
    LaunchBox<blockThreads> launch_box;
    rxmesh_static.prepare_launch_box(Op::VV, launch_box, true, false);

    output_container->reset(INVALID32, rxmesh::DEVICE);
    input_container->reset(INVALID32, rxmesh::DEVICE);

    ::RXMeshTest tester(true);

    // launch
    higher_query<Op::VV, blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh_static.get_context(), *input_container, *output_container);

    // move containers to the CPU for testing
    output_container->move(rxmesh::DEVICE, rxmesh::HOST);
    input_container->move(rxmesh::DEVICE, rxmesh::HOST);

    // verify
    EXPECT_TRUE(
        tester.test_VVV(rxmesh_static, *input_container, *output_container));
}