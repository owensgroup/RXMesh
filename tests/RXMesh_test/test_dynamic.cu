#include <assert.h>
#include "gtest/gtest.h"

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/kernels/for_each.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"

using Config = uint32_t;
enum : Config
{
    None                   = 0x00,
    OnRibbonConflicting    = 0x01,
    InteriorConflicting    = 0x02,
    OnRibbonNotConflicting = 0x04,
    InteriorNotConflicting = 0x08,
};

__global__ static void set_should_slice(rxmesh::Context context)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < context.m_num_patches[0]) {
        context.m_patches_info[tid].should_slice = true;
    }
}

template <uint32_t blockThreads>
__global__ static void random_flips(rxmesh::Context                context,
                                    rxmesh::VertexAttribute<float> coords,
                                    rxmesh::EdgeAttribute<int>     to_flip)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, false);


    if (cavity.patch_id() == INVALID32) {
        return;
    }

    Query<blockThreads> query(context, cavity.patch_id());
    query.compute_vertex_valence(block, shrd_alloc);

    const LocalVertexT* ev = cavity.patch_info().ev;

    for_each_edge(cavity.patch_info(), [&](const EdgeHandle eh) {
        const uint16_t v0 = ev[2 * eh.local_id() + 0].id;
        const uint16_t v1 = ev[2 * eh.local_id() + 1].id;

        if (to_flip(eh) == 1) {
            if (query.vertex_valence(v0) > 3 && query.vertex_valence(v1) > 3) {
                cavity.create(eh);
            }
        }
    });

    block.sync();

    if (cavity.prologue(block, shrd_alloc, coords, to_flip)) {

        // so that we don't flip them again
        for_each_edge(cavity.patch_info(), [&](const EdgeHandle eh) {
            if (to_flip(eh) == 1) {
                to_flip(eh) = 2;
            }
        });

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            DEdgeHandle new_edge = cavity.add_edge(
                cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));

            if (new_edge.is_valid()) {
                cavity.add_face(cavity.get_cavity_edge(c, 0),
                                new_edge,
                                cavity.get_cavity_edge(c, 3));


                cavity.add_face(cavity.get_cavity_edge(c, 1),
                                cavity.get_cavity_edge(c, 2),
                                new_edge.get_flip_dedge());
            }
        });
    }

    cavity.epilogue(block);
}


template <uint32_t blockThreads>
__global__ static void random_collapses(rxmesh::Context                context,
                                        rxmesh::VertexAttribute<float> coords,
                                        rxmesh::EdgeAttribute<int> to_collapse,
                                        rxmesh::VertexAttribute<int> v_attr,
                                        rxmesh::EdgeAttribute<int>   e_attr,
                                        rxmesh::FaceAttribute<int>   f_attr)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::EV> cavity(
        block, context, shrd_alloc, true);


    if (cavity.patch_id() == INVALID32) {
        return;
    }

    for_each_edge(cavity.patch_info(), [&](const EdgeHandle eh) {
        if (to_collapse(eh) == 1) {
            cavity.create(eh);
        }
    });

    block.sync();

    if (cavity.prologue(
            block, shrd_alloc, coords, to_collapse, v_attr, e_attr, f_attr)) {

        for_each_edge(cavity.patch_info(), [&](const EdgeHandle eh) {
            if (cavity.is_successful(eh)) {
                to_collapse(eh) = 0;
            }
        });

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);

            VertexHandle v0, v1;

            cavity.get_vertices(src, v0, v1);

            const VertexHandle new_v = cavity.add_vertex();

            coords(new_v, 0) = (coords(v0, 0) + coords(v1, 0)) / 2.0f;
            coords(new_v, 1) = (coords(v0, 1) + coords(v1, 1)) / 2.0f;
            coords(new_v, 2) = (coords(v0, 2) + coords(v1, 2)) / 2.0f;

            DEdgeHandle e0 =
                cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));
            const DEdgeHandle e_init = e0;

            if (e0.is_valid()) {
                for (uint16_t i = 0; i < size; ++i) {
                    const DEdgeHandle e = cavity.get_cavity_edge(c, i);
                    const DEdgeHandle e1 =
                        (i == size - 1) ?
                            e_init.get_flip_dedge() :
                            cavity.add_edge(cavity.get_cavity_vertex(c, i + 1),
                                            new_v);
                    if (!e1.is_valid()) {
                        break;
                    }
                    const FaceHandle f = cavity.add_face(e0, e, e1);
                    if (!f.is_valid()) {
                        break;
                    }
                    e0 = e1.get_flip_dedge();
                }
            }
        });
    }

    cavity.epilogue(block);
}

inline void set_edge_tag(rxmesh::RXMeshDynamic&      rx,
                         rxmesh::EdgeAttribute<int>& edge_tag,
                         const Config                config)
{
    using namespace rxmesh;

    rx.for_each_edge(HOST, [edge_tag, config](const EdgeHandle eh) {
        if (eh.patch_id() == 0) {

            if ((config & OnRibbonNotConflicting) == OnRibbonNotConflicting) {
                if (eh.local_id() == 42 || eh.local_id() == 2 ||
                    eh.local_id() == 53 || eh.local_id() == 315) {
                    edge_tag(eh) = 1;
                }
            }


            if ((config & OnRibbonConflicting) == OnRibbonConflicting) {
                if (eh.local_id() == 11 || eh.local_id() == 10 ||
                    eh.local_id() == 358 || eh.local_id() == 359 ||
                    eh.local_id() == 354 || eh.local_id() == 356 ||
                    eh.local_id() == 51) {
                    edge_tag(eh) = 1;
                }
            }


            if ((config & InteriorNotConflicting) == InteriorNotConflicting) {
                if (eh.local_id() == 174 || eh.local_id() == 184 ||
                    eh.local_id() == 94 || eh.local_id() == 58 ||
                    eh.local_id() == 362 || eh.local_id() == 70 ||
                    eh.local_id() == 420) {
                    edge_tag(eh) = 1;
                }
            }

            if ((config & InteriorConflicting) == InteriorConflicting) {
                if (eh.local_id() == 22 || eh.local_id() == 29 ||
                    eh.local_id() == 23 || eh.local_id() == 389 ||
                    eh.local_id() == 39 || eh.local_id() == 40 ||
                    eh.local_id() == 41 || eh.local_id() == 16) {
                    edge_tag(eh) = 1;
                }
            }
        }


        if (eh.patch_id() == 1) {
            if ((config & OnRibbonNotConflicting) == OnRibbonNotConflicting) {
                if (eh.local_id() == 383 || eh.local_id() == 324 ||
                    eh.local_id() == 355 || eh.local_id() == 340 ||
                    eh.local_id() == 726 || eh.local_id() == 667 ||
                    eh.local_id() == 706) {
                    edge_tag(eh) = 1;
                }
            }


            if ((config & OnRibbonConflicting) == OnRibbonConflicting) {
                if (eh.local_id() == 399 || eh.local_id() == 398 ||
                    eh.local_id() == 402 || eh.local_id() == 418 ||
                    eh.local_id() == 419 || eh.local_id() == 401 ||
                    eh.local_id() == 413 || eh.local_id() == 388 ||
                    eh.local_id() == 396 || eh.local_id() == 395) {
                    edge_tag(eh) = 1;
                }
            }


            if ((config & InteriorNotConflicting) == InteriorNotConflicting) {
                if (eh.local_id() == 528 || eh.local_id() == 532 ||
                    eh.local_id() == 103 || eh.local_id() == 140 ||
                    eh.local_id() == 206 || eh.local_id() == 285 ||
                    eh.local_id() == 162 || eh.local_id() == 385) {
                    edge_tag(eh) = 1;
                }
            }

            if ((config & InteriorConflicting) == InteriorConflicting) {
                if (eh.local_id() == 630 || eh.local_id() == 525 ||
                    eh.local_id() == 212 || eh.local_id() == 47 ||
                    eh.local_id() == 46 || eh.local_id() == 58 ||
                    eh.local_id() == 59 || eh.local_id() == 57 ||
                    eh.local_id() == 232 || eh.local_id() == 214 ||
                    eh.local_id() == 233) {
                    edge_tag(eh) = 1;
                }
            }
        }
    });
}

TEST(RXMeshDynamic, RandomFlips)
{
    using namespace rxmesh;

    RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "sphere3.obj",
                     STRINGIFY(INPUT_DIR) "sphere3_patches",
                     256,
                     1.8);

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();
    const uint32_t num_faces    = rx.get_num_faces();

    auto coords = rx.get_input_vertex_coordinates();

    auto to_flip = rx.add_edge_attribute<int>("to_flip", 1);
    to_flip->reset(0, HOST);


    const Config config = InteriorNotConflicting | InteriorConflicting |
                          OnRibbonNotConflicting | OnRibbonConflicting;

    // adding couple more edges
    if ((config & InteriorConflicting) == InteriorConflicting) {
        (*to_flip)({0, 156}) = 1;
        (*to_flip)({0, 26})  = 1;
        (*to_flip)({1, 527}) = 1;
        (*to_flip)({1, 44})  = 1;
        (*to_flip)({1, 209}) = 1;
    }
    if ((config & InteriorNotConflicting) == InteriorNotConflicting) {
        (*to_flip)({0, 26}) = 1;
    }

    set_edge_tag(rx, *to_flip, config);

    to_flip->move(HOST, DEVICE);


    set_should_slice<<<rx.get_num_patches(), 1>>>(rx.get_context());
    rx.slice_patches(*coords, *to_flip);
    rx.cleanup();
    CUDA_ERROR(cudaDeviceSynchronize());
    rx.update_host();

    EXPECT_TRUE(rx.validate());
    coords->move(DEVICE, HOST);
    to_flip->move(DEVICE, HOST);

#if USE_POLYSCOPE
    rx.update_polyscope();
    rx.get_polyscope_mesh()->updateVertexPositions(*coords);
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    rx.get_polyscope_mesh()
        ->addEdgeScalarQuantity("toFlip", *to_flip)
        ->setMapRange({0, 2});
    for (uint32_t p = 0; p < rx.get_num_patches(); ++p) {
        rx.render_patch(p)->setEnabled(false);
    }
    // polyscope::show();
#endif


    constexpr uint32_t blockThreads = 256;

    int iter = 0;
    while (!rx.is_queue_empty()) {
        RXMESH_INFO("iter = {}", ++iter);
        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box({},
                              launch_box,
                              (void*)random_flips<blockThreads>,
                              true,
                              false,
                              true);
        random_flips<blockThreads><<<launch_box.blocks,
                                     launch_box.num_threads,
                                     launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *to_flip);

        rx.slice_patches(*coords, *to_flip);
        rx.cleanup();
    }

    CUDA_ERROR(cudaDeviceSynchronize());

    rx.update_host();

    coords->move(DEVICE, HOST);
    to_flip->move(DEVICE, HOST);


    EXPECT_EQ(num_vertices, rx.get_num_vertices());
    EXPECT_EQ(num_edges, rx.get_num_edges());
    EXPECT_EQ(num_faces, rx.get_num_faces());
    EXPECT_TRUE(rx.validate());

#if USE_POLYSCOPE
    rx.update_polyscope();
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();

    for (uint32_t p = 0; p < rx.get_num_patches(); ++p) {
        rx.render_patch(p)->setEnabled(false);
    }

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->addEdgeScalarQuantity("toFlip", *to_flip)->setMapRange({0, 2});
    ps_mesh->setEnabled(false);
    // polyscope::show();
#endif

    // polyscope::removeAllStructures();
}


TEST(RXMeshDynamic, RandomCollapse)
{
    using namespace rxmesh;

    RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "sphere3.obj",
                     STRINGIFY(INPUT_DIR) "sphere3_patches",
                     256,
                     1.8);

    auto coords = rx.get_input_vertex_coordinates();

    auto to_collapse = rx.add_edge_attribute<int>("to_collapse", 1);
    to_collapse->reset(0, HOST);

    auto v_attr = rx.add_vertex_attribute<int>("vAttr", 1);
    v_attr->reset(0, HOST);

    auto e_attr = rx.add_edge_attribute<int>("eAttr", 1);
    e_attr->reset(0, HOST);

    auto f_attr = rx.add_face_attribute<int>("fAttr", 1);
    f_attr->reset(0, HOST);

    // these config names makes better sense for edge flips but it is possible
    // that some of the non-conflicting cavities becomes conflicting for edge
    // collapse
    const Config config = InteriorNotConflicting | InteriorConflicting |
                          OnRibbonNotConflicting | OnRibbonConflicting;


    set_edge_tag(rx, *to_collapse, config);

    to_collapse->move(HOST, DEVICE);

    EXPECT_TRUE(rx.validate());

#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    rx.get_polyscope_mesh()
        ->addEdgeScalarQuantity("toCollapse", *to_collapse)
        ->setMapRange({0, 2});
    for (uint32_t p = 0; p < rx.get_num_patches(); ++p) {
        rx.render_patch(p)->setEnabled(false);
    }
    // polyscope::show();
#endif


    constexpr uint32_t blockThreads = 256;

    int iter = 0;
    while (!rx.is_queue_empty()) {
        RXMESH_INFO("iter = {}", ++iter);
        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box(
            {}, launch_box, (void*)random_flips<blockThreads>);
        random_collapses<blockThreads><<<launch_box.blocks,
                                         launch_box.num_threads,
                                         launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *to_collapse, *v_attr, *e_attr, *f_attr);

        rx.slice_patches(*coords, *to_collapse);
        rx.cleanup();
    }

    CUDA_ERROR(cudaDeviceSynchronize());

    rx.update_host();

    coords->move(DEVICE, HOST);

    v_attr->move(DEVICE, HOST);
    e_attr->move(DEVICE, HOST);
    f_attr->move(DEVICE, HOST);


    EXPECT_TRUE(rx.validate());

    // rx.export_obj(STRINGIFY(OUTPUT_DIR) "sphere3_33.obj", *coords);

#if USE_POLYSCOPE
    rx.update_polyscope();
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();

    for (uint32_t p = 0; p < rx.get_num_patches(); ++p) {
        rx.render_patch(p)->setEnabled(false);
    }

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->setEnabled(false);
    ps_mesh->addVertexScalarQuantity("vAttr", *v_attr);
    ps_mesh->addEdgeScalarQuantity("eAttr", *e_attr);
    ps_mesh->addFaceScalarQuantity("fAttr", *f_attr);
    // polyscope::show();
#endif

    // polyscope::removeAllStructures();
}