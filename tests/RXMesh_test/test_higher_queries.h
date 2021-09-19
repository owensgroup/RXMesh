#include "gtest/gtest.h"
#include "higher_query.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh_test.h"

using namespace RXMESH;

TEST(RXMesh, HigherQueries)
{
    // Select device
    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<uint32_t>> Faces;
    if (!import_obj(rxmesh_args.obj_file_name, Verts, Faces,
                    rxmesh_args.quite)) {
        exit(EXIT_FAILURE);
    }

    // RXMesh
    RXMeshStatic<PATCH_SIZE> rxmesh_static(Faces, Verts, false,
                                           rxmesh_args.quite);

    uint32_t input_size = rxmesh_static.get_num_vertices();

    // input/output container
    RXMeshAttribute<uint32_t> input_container;
    input_container.init(input_size, 1u, RXMESH::DEVICE, RXMESH::AoS, false,
                         false);

    RXMeshAttribute<uint32_t> output_container;
    output_container.init(input_size,
                          input_size,  // that is a bit excessive
                          RXMESH::DEVICE, RXMESH::SoA, false, false);

    // launch box
    constexpr uint32_t      blockThreads = 512;
    LaunchBox<blockThreads> launch_box;
    rxmesh_static.prepare_launch_box(Op::VV, launch_box, true, false);

    output_container.reset(INVALID32, RXMESH::DEVICE);
    input_container.reset(INVALID32, RXMESH::DEVICE);

    ::RXMeshTest tester(true);

    // launch
    higher_query<Op::VV, blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh_static.get_context(), input_container, output_container);

    // move containers to the CPU for testing
    output_container.move(RXMESH::DEVICE, RXMESH::HOST);
    input_container.move(RXMESH::DEVICE, RXMESH::HOST);

    // verify
    EXPECT_TRUE(tester.run_higher_query_verifier(rxmesh_static, input_container,
                                                 output_container));


    input_container.release();
    output_container.release();
}