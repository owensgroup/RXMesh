#include <assert.h>
#include "gtest/gtest.h"

#include "rxmesh/cavity.cuh"
#include "rxmesh/kernels/for_each_dispatcher.cuh"
#include "rxmesh/rxmesh_dynamic.h"

using Config = uint32_t;
enum : Config
{
    OnRibbonConflicting    = 0x01,
    InteriorConflicting    = 0x02,
    OnRibbonNotConflicting = 0x04,
    InteriorNotConflicting = 0x08,
};

template <uint32_t blockThreads>
__global__ static void edge_flip_kernel(rxmesh::Context                context,
                                        rxmesh::VertexAttribute<float> coords,
                                        rxmesh::VertexAttribute<float> v_attr,
                                        rxmesh::EdgeAttribute<float>   e_attr,
                                        rxmesh::FaceAttribute<float>   f_attr,
                                        Config                         config)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    Cavity<blockThreads, CavityOp::E> cavity(block, context, shrd_alloc);

    const uint32_t pid = cavity.m_patch_info.patch_id;

    if (pid != 1) {
        return;
    }


    // if (cavity.m_patch_info.patch_id == INVALID32) {
    //    return;
    //}

    for_each_dispatcher<Op::E, blockThreads>(context, [&](const EdgeHandle eh) {
        if (pid == 0) {

            if ((config & OnRibbonNotConflicting) == OnRibbonNotConflicting) {
                if (eh.unpack().second == 11 || eh.unpack().second == 51 ||
                    eh.unpack().second == 2 || eh.unpack().second == 315) {
                    e_attr(eh) = 100;
                    cavity.add(eh);
                }
            }


            if ((config & OnRibbonConflicting) == OnRibbonConflicting) {
                if (eh.unpack().second == 11 || eh.unpack().second == 10 ||
                    eh.unpack().second == 358 || eh.unpack().second == 359 ||
                    eh.unpack().second == 354 || eh.unpack().second == 356) {
                    e_attr(eh) = 100;
                    cavity.add(eh);
                }
            }


            if ((config & InteriorNotConflicting) == InteriorNotConflicting) {
                if (eh.unpack().second == 26 || eh.unpack().second == 174 ||
                    eh.unpack().second == 184 || eh.unpack().second == 94 ||
                    eh.unpack().second == 58 || eh.unpack().second == 362 ||
                    eh.unpack().second == 70 || eh.unpack().second == 420) {
                    e_attr(eh) = 100;
                    cavity.add(eh);
                }
            }

            if ((config & InteriorConflicting) == InteriorConflicting) {
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


        if (pid == 1) {
            if ((config & OnRibbonNotConflicting) == OnRibbonNotConflicting) {
                if (eh.unpack().second == 383 || eh.unpack().second == 324 ||
                    eh.unpack().second == 355 || eh.unpack().second == 340 ||
                    eh.unpack().second == 726 || eh.unpack().second == 667 ||
                    eh.unpack().second == 706) {
                    e_attr(eh) = 100;
                    cavity.add(eh);
                }
            }


            if ((config & OnRibbonConflicting) == OnRibbonConflicting) {
                if (eh.unpack().second == 399 || eh.unpack().second == 398 ||
                    eh.unpack().second == 402 || eh.unpack().second == 418 ||
                    eh.unpack().second == 419 || eh.unpack().second == 401 ||
                    eh.unpack().second == 413 || eh.unpack().second == 388 ||
                    eh.unpack().second == 396 || eh.unpack().second == 395) {
                    e_attr(eh) = 100;
                    cavity.add(eh);
                }
            }


            if ((config & InteriorNotConflicting) == InteriorNotConflicting) {
                if (eh.unpack().second == 528 || eh.unpack().second == 532 ||
                    eh.unpack().second == 103 || eh.unpack().second == 140 ||
                    eh.unpack().second == 206 || eh.unpack().second == 285 ||
                    eh.unpack().second == 162 || eh.unpack().second == 385) {
                    e_attr(eh) = 100;
                    cavity.add(eh);
                }
            }

            if ((config & InteriorConflicting) == InteriorConflicting) {
                if (eh.unpack().second == 527 || eh.unpack().second == 209 ||
                    eh.unpack().second == 44 || eh.unpack().second == 525 ||
                    eh.unpack().second == 212 || eh.unpack().second == 47 ||
                    eh.unpack().second == 46 || eh.unpack().second == 58 ||
                    eh.unpack().second == 59 || eh.unpack().second == 57 ||
                    eh.unpack().second == 232 || eh.unpack().second == 214 ||
                    eh.unpack().second == 233) {
                    e_attr(eh) = 100;
                    cavity.add(eh);
                }
            }
        }
    });

    block.sync();


    if (cavity.process(block, shrd_alloc)) {

        cavity.update_attributes(block, coords);
        cavity.update_attributes(block, v_attr);
        cavity.update_attributes(block, e_attr);
        cavity.update_attributes(block, f_attr);

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            DEdgeHandle new_edge =
                cavity.add_edge(c,
                                cavity.get_cavity_vertex(c, 1),
                                cavity.get_cavity_vertex(c, 3));

            cavity.add_face(c,
                            cavity.get_cavity_edge(c, 0),
                            new_edge,
                            cavity.get_cavity_edge(c, 3));

            cavity.add_face(c,
                            cavity.get_cavity_edge(c, 1),
                            cavity.get_cavity_edge(c, 2),
                            new_edge.get_flip_dedge());
        });
        block.sync();

        cavity.cleanup(block);
    }
}

TEST(RXMeshDynamic, Cavity)
{
    using namespace rxmesh;
    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    // RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "sphere3.obj",
    // rxmesh_args.quite); rx.save(STRINGIFY(OUTPUT_DIR) "sphere3_patches");

    RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "sphere3.obj",
                     rxmesh_args.quite,
                     STRINGIFY(INPUT_DIR) "sphere3_patches");

#if USE_POLYSCOPE
    std::pair<double, double> ps_range(-2, 2);
    rx.polyscope_render_vertex_patch()->setMapRange(ps_range);
    rx.polyscope_render_edge_patch()->setMapRange(ps_range);
    rx.polyscope_render_face_patch()->setMapRange(ps_range);
#endif

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

    Config congif = InteriorNotConflicting | InteriorConflicting |
                    OnRibbonNotConflicting | OnRibbonConflicting;

    edge_flip_kernel<blockThreads><<<launch_box.blocks,
                                     launch_box.num_threads,
                                     launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *coords, *v_attr, *e_attr, *f_attr, congif);

    CUDA_ERROR(cudaDeviceSynchronize());

    rx.update_host();

    coords->move(DEVICE, HOST);
    v_attr->move(DEVICE, HOST);
    e_attr->move(DEVICE, HOST);
    f_attr->move(DEVICE, HOST);

    EXPECT_EQ(num_vertices, rx.get_num_vertices());
    EXPECT_EQ(num_edges, rx.get_num_edges());
    EXPECT_EQ(num_faces, rx.get_num_faces());

    EXPECT_TRUE(rx.validate());

#if USE_POLYSCOPE
    rx.update_polyscope();
    rx.polyscope_render_vertex_patch()->setMapRange(ps_range);
    rx.polyscope_render_edge_patch()->setMapRange(ps_range);
    rx.polyscope_render_face_patch()->setMapRange(ps_range);

    {
        uint32_t pid      = 0;
        auto     ps_patch = rx.render_patch(pid);
        rx.polyscope_render_vertex_patch(pid, ps_patch)->setMapRange(ps_range);
        rx.polyscope_render_face_patch(pid, ps_patch)->setMapRange(ps_range);
        rx.polyscope_render_edge_patch(pid, ps_patch)->setMapRange(ps_range);
    }
    {
        uint32_t pid      = 1;
        auto     ps_patch = rx.render_patch(pid);
        rx.polyscope_render_vertex_patch(pid, ps_patch)->setMapRange(ps_range);
        rx.polyscope_render_face_patch(pid, ps_patch)->setMapRange(ps_range);
        rx.polyscope_render_edge_patch(pid, ps_patch)->setMapRange(ps_range);
    }

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->addVertexScalarQuantity("vAttr", *v_attr);
    ps_mesh->addEdgeScalarQuantity("eAttr", *e_attr);
    ps_mesh->addFaceScalarQuantity("fAttr", *f_attr);
    polyscope::show();
#endif
}