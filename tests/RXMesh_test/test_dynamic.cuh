#include <assert.h>
#include "gtest/gtest.h"

#include "rxmesh/cavity.cuh"
#include "rxmesh/rxmesh_dynamic.h"

template <uint32_t blockThreads>
__global__ static void edge_flip_kernel(
    rxmesh::Context                      context,
    const rxmesh::VertexAttribute<float> coords,
    rxmesh::VertexAttribute<float>       v_attr,
    rxmesh::EdgeAttribute<float>         e_attr,
    rxmesh::FaceAttribute<float>         f_attr,
    bool                                 conflicting,
    bool                                 on_ribbon)
{
    if (blockIdx.x != 0) {
        return;
    }
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    Cavity<blockThreads, CavityOp::E> cavity(block, context, shrd_alloc);

    const uint16_t before_num_vertices = cavity.m_patch_info.num_vertices[0];
    const uint16_t before_num_edges    = cavity.m_patch_info.num_edges[0];
    const uint16_t before_num_faces    = cavity.m_patch_info.num_faces[0];


    for (uint16_t v = 0; v < cavity.m_patch_info.num_vertices[0]; ++v) {
        if (!detail::is_owned(v, cavity.m_patch_info.owned_mask_v)) {
            LPPair lp = cavity.m_patch_info.lp_v.find(v);
            v_attr({cavity.m_patch_info.patch_stash.get_patch(lp),
                    lp.local_id_in_owner_patch()}) = v;
        }
    }

    for_each_dispatcher<Op::E, blockThreads>(context, [&](const EdgeHandle eh) {
        if (on_ribbon) {
            if (eh.unpack().second == 11) {
                e_attr(eh) = 100;
                cavity.add(eh);
            }
        } else {
            if (!conflicting) {
                if (eh.unpack().second == 26 || eh.unpack().second == 174 ||
                    eh.unpack().second == 184 || eh.unpack().second == 94 ||
                    eh.unpack().second == 58 || eh.unpack().second == 362 ||
                    eh.unpack().second == 70 || eh.unpack().second == 420) {
                    e_attr(eh) = 100;
                    cavity.add(eh);
                }
            } else {
                if (eh.unpack().second == 26 || eh.unpack().second == 22 ||
                    eh.unpack().second == 29 || eh.unpack().second == 156 ||
                    eh.unpack().second == 23 || eh.unpack().second == 389 ||
                    eh.unpack().second == 39 || eh.unpack().second == 40 ||
                    eh.unpack().second == 41 || eh.unpack().second == 16) {
                    e_attr(eh) = 100;
                    cavity.add(eh);
                }
            }
        }
    });

    block.sync();


    cavity.process(block, shrd_alloc);

    /*cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
        assert(size == 4);

        auto new_edge = cavity.add_edge(
            c, cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));

        cavity.add_face(c,
                        cavity.get_cavity_edge(c, 0),
                        new_edge,
                        cavity.get_cavity_edge(c, 3));

        cavity.add_face(c,
                        cavity.get_cavity_edge(c, 1),
                        cavity.get_cavity_edge(c, 2),
                        new_edge.get_flip_dedge());
    });

    cavity.cleanup(block);

    assert(before_num_vertices == cavity.m_patch_info.num_vertices[0]);
    assert(before_num_edges == cavity.m_patch_info.num_edges[0]);
    assert(before_num_faces == cavity.m_patch_info.num_faces[0]);
    */
}

TEST(RXMeshDynamic, Cavity)
{
    using namespace rxmesh;
    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    // RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "sphere3.obj", rxmesh_args.quite);
    // rx.save(STRINGIFY(OUTPUT_DIR) "sphere3_patches");

    RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "sphere3.obj",
                     rxmesh_args.quite,
                     STRINGIFY(INPUT_DIR) "sphere3_patches");

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();
    const uint32_t num_faces    = rx.get_num_faces();

    auto coords = rx.get_input_vertex_coordinates();

    auto v_attr = rx.add_vertex_attribute<float>("vAttr", 1);
    auto e_attr = rx.add_edge_attribute<float>("eAttr", 1);
    auto f_attr = rx.add_face_attribute<float>("fAttr", 1);

    v_attr->reset(0, DEVICE);
    e_attr->reset(0, DEVICE);
    f_attr->reset(0, DEVICE);

    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;

    rx.prepare_launch_box(
        {}, launch_box, (void*)edge_flip_kernel<blockThreads>);

    edge_flip_kernel<blockThreads><<<launch_box.blocks,
                                     launch_box.num_threads,
                                     launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *coords, *v_attr, *e_attr, *f_attr, false, true);

    CUDA_ERROR(cudaDeviceSynchronize());

    v_attr->move(DEVICE, HOST);
    e_attr->move(DEVICE, HOST);
    f_attr->move(DEVICE, HOST);

    rx.update_host();

    EXPECT_EQ(num_vertices, rx.get_num_vertices());
    EXPECT_EQ(num_edges, rx.get_num_edges());
    EXPECT_EQ(num_faces, rx.get_num_faces());

    EXPECT_TRUE(rx.validate());

#if USE_POLYSCOPE
    polyscope::init();
    auto polyscope_mesh = rx.get_polyscope_mesh();
    rx.polyscope_render_vertex_patch();
    rx.polyscope_render_edge_patch();
    rx.polyscope_render_face_patch();
    polyscope_mesh->addVertexScalarQuantity("vAttr", *v_attr);
    polyscope_mesh->addEdgeScalarQuantity("eAttr", *e_attr);
    polyscope_mesh->addFaceScalarQuantity("fAttr", *f_attr);
    polyscope::show();
#endif
}