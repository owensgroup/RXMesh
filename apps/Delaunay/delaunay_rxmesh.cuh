#include "rxmesh/cavity.cuh"
#include "rxmesh/rxmesh_dynamic.h"

template <typename T, uint32_t blockThreads>
__global__ static void delaunay_edge_flip(rxmesh::Context            context,
                                          rxmesh::VertexAttribute<T> coords)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    Cavity<blockThreads, CavityOp::E> cavity(block, context, shrd_alloc);

    const uint32_t pid = cavity.m_patch_info.patch_id;

    if (pid == INVALID32) {
        return;
    }

    // TODO edge diamond query to and calc delaunay condition

    block.sync();

    if (cavity.process(block, shrd_alloc)) {

        cavity.update_attributes(block, coords);

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

inline bool delaunay_rxmesh(rxmesh::RXMeshDynamic& rx)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();
    const uint32_t num_faces    = rx.get_num_faces();

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::EVDiamond},
                          launch_box,
                          (void*)delaunay_edge_flip<float, blockThreads>);

    auto coords = rx.get_input_vertex_coordinates();

    GPUTimer timer;
    timer.start();

    delaunay_edge_flip<float, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords);

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    RXMESH_TRACE("delaunay_rxmesh() RXMesh Delaunay Edge Flip took {} (ms)",
                 timer.elapsed_millis());


    rx.update_host();
    coords->move(DEVICE, HOST);

    EXPECT_EQ(num_vertices, rx.get_num_vertices());
    EXPECT_EQ(num_edges, rx.get_num_edges());
    EXPECT_EQ(num_faces, rx.get_num_faces());

    EXPECT_TRUE(rx.validate());

#if USE_POLYSCOPE
    rx.update_polyscope();
    rx.get_polyscope_mesh()->updateVertexPositions(*coords);
    polyscope::show();
#endif

    return true;
}