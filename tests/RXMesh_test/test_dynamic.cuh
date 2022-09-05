#include "gtest/gtest.h"

#include "rxmesh/cavity.cuh"
#include "rxmesh/rxmesh_dynamic.h"

template <uint32_t blockThreads>
__global__ static void dynamic_kernel(rxmesh::Context                context,
                                      rxmesh::VertexAttribute<float> v_attr,
                                      rxmesh::EdgeAttribute<float>   e_attr,
                                      rxmesh::FaceAttribute<float>   f_attr)
{
    if (blockIdx.x != 0) {
        return;
    }
    using namespace rxmesh;
    namespace cg           = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    ShmemAllocator   shrd_alloc;
    PatchInfo        patch_info = context.get_patches_info()[blockIdx.x];
    Cavity<blockThreads, CavityOp::E> cavity(block, shrd_alloc, patch_info);

    for_each_dispatcher<Op::E, blockThreads>(context, [&](const EdgeHandle eh) {
        // TODO user-defined condition
        if (eh.unpack().second == 10) {
            cavity.add(eh);
        }
    });
    block.sync();

    cavity.process(block, shrd_alloc, patch_info);

    cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
        assert(size == 4);

        auto new_edge =
            cavity.add_edge(patch_info,
                            cavity.get_cavity_vertex(patch_info, c, 1),
                            cavity.get_cavity_vertex(patch_info, c, 3));
        cavity.add_face(patch_info,
                        cavity.get_cavity_edge(patch_info, c, 0),
                        new_edge,
                        cavity.get_cavity_edge(patch_info, c, 3));

        cavity.add_face(patch_info,
                        cavity.get_cavity_edge(patch_info, c, 1),
                        cavity.get_cavity_edge(patch_info, c, 2),
                        new_edge.get_flip_dedge());
    });

    //cavity.cleanup();
}

TEST(RXMeshDynamic, Cavity)
{
    using namespace rxmesh;
    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    // RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "sphere3.obj", rxmesh_args.quite);
    // rx.save(STRINGIFY(OUTPUT_DIR) "sphere3_patcher");

    RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "sphere3.obj",
                     rxmesh_args.quite,
                     STRINGIFY(OUTPUT_DIR) "sphere3_patcher");

    auto coords = rx.get_input_vertex_coordinates();

    auto v_attr = rx.add_vertex_attribute<float>("vAttr", 1);
    auto e_attr = rx.add_edge_attribute<float>("eAttr", 1);
    auto f_attr = rx.add_face_attribute<float>("fAttr", 1);

    v_attr->reset(0, DEVICE);
    e_attr->reset(0, DEVICE);
    f_attr->reset(0, DEVICE);

    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;

    rx.prepare_launch_box({}, launch_box, (void*)dynamic_kernel<blockThreads>);

    dynamic_kernel<blockThreads><<<launch_box.blocks,
                                   launch_box.num_threads,
                                   launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *v_attr, *e_attr, *f_attr);

    CUDA_ERROR(cudaDeviceSynchronize());

    v_attr->move(DEVICE, HOST);
    e_attr->move(DEVICE, HOST);
    f_attr->move(DEVICE, HOST);

#if USE_POLYSCOPE
    polyscope::init();
    auto polyscope_mesh = rx.get_polyscope_mesh();
    polyscope_mesh->setEdgeWidth(1.0);
    rx.polyscope_render_vertex_patch();
    rx.polyscope_render_edge_patch();
    rx.polyscope_render_face_patch();
    polyscope_mesh->addVertexScalarQuantity("vAttr", *v_attr);
    polyscope_mesh->addEdgeScalarQuantity("eAttr", *e_attr);
    polyscope_mesh->addFaceScalarQuantity("fAttr", *f_attr);
    polyscope::show();
#endif

    // TODO
    // rx.update_host();
    // EXPECT_TRUE(rx.validate());
}