#include "gtest/gtest.h"
#include "higher_query.cuh"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh_test.h"

TEST(RXMeshStatic, DISABLED_HigherQueries)
{
    using namespace rxmesh;


    std::vector<std::vector<float>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;
    ASSERT_TRUE(import_obj(STRINGIFY(INPUT_DIR) "sphere3.obj", Verts, Faces));

    RXMeshStatic rx(Faces);


    // input/output container
    auto input = rx.add_vertex_attribute<VertexHandle>("input", 1);
    input->reset(VertexHandle(), rxmesh::DEVICE);

    // we assume that every vertex could store up to num_vertices as its
    // neighbor vertices which is a bit excessive
    auto output =
        rx.add_vertex_attribute<VertexHandle>("output", rx.get_num_vertices());
    output->reset(VertexHandle(), rxmesh::DEVICE);

    // launch box
    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box(
        {Op::VV}, launch_box, (void*)higher_query<blockThreads, Op::VV>, false);


    RXMeshTest tester(rx, Faces);

    // launch
    higher_query<blockThreads, Op::VV>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *input, *output);

    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaDeviceSynchronize());

    // move containers to the CPU for testing
    output->move(rxmesh::DEVICE, rxmesh::HOST);
    input->move(rxmesh::DEVICE, rxmesh::HOST);

    // verify
    EXPECT_TRUE(tester.run_test(rx, Faces, *input, *output, true));
}