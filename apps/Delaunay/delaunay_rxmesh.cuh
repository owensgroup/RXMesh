#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"

template <typename T, uint32_t blockThreads>
__global__ static void delaunay_edge_flip(rxmesh::Context            context,
                                          rxmesh::VertexAttribute<T> coords)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager<blockThreads, CavityOp::E> cavity(block, context, shrd_alloc);

    const uint32_t pid = cavity.patch_id();

    if (pid == INVALID32) {
        return;
    }

    // edge diamond query to and calc delaunay condition
    auto is_delaunay = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices


        auto v0 = iter[0];
        auto v1 = iter[2];

        auto v2 = iter[1];
        auto v3 = iter[3];

        // if not a boundary edge
        if (v2.is_valid() && v3.is_valid()) {

            constexpr double PII = 3.14159265358979323f;

            const Vector<3, T> V0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
            const Vector<3, T> V1(coords(v1, 0), coords(v1, 1), coords(v1, 2));
            const Vector<3, T> V2(coords(v2, 0), coords(v2, 1), coords(v2, 2));
            const Vector<3, T> V3(coords(v3, 0), coords(v3, 1), coords(v3, 2));

            // find the angle between S, M, Q vertices (i.e., angle at M)
            auto angle_between_three_vertices = [](const Vector<3, T>& S,
                                                   const Vector<3, T>& M,
                                                   const Vector<3, T>& Q) {
                Vector<3, T> p1      = S - M;
                Vector<3, T> p2      = Q - M;
                T            dot_pro = dot(p1, p2);
                if constexpr (std::is_same_v<T, float>) {
                    return acosf(dot_pro / (p1.norm() * p2.norm()));
                } else {
                    return acos(dot_pro / (p1.norm() * p2.norm()));
                }
            };

            // first check if the edge formed by v0-v1 is a delaunay edge
            // where v2 and v3 are the opposite vertices to the edge
            //    0
            //  / | \
            // 3  |  2
            // \  |  /
            //    1
            // if not delaunay, then we check if flipping it won't create a
            // foldover The case below would create a fold over
            //      0
            //    / | \
            //   /  1  \
            //  / /  \  \
            //  2       3


            T lambda = angle_between_three_vertices(V0, V2, V1);
            T gamma  = angle_between_three_vertices(V0, V3, V1);

            if (lambda + gamma > PII + std::numeric_limits<T>::epsilon()) {
                // check if flipping won't create foldover

                const T alpha0 = angle_between_three_vertices(V3, V0, V1);
                const T beta0  = angle_between_three_vertices(V2, V0, V1);

                const T alpha1 = angle_between_three_vertices(V3, V1, V0);
                const T beta1  = angle_between_three_vertices(V2, V1, V0);

                if (alpha0 + beta0 < PII - std::numeric_limits<T>::epsilon() &&
                    alpha1 + beta1 < PII - std::numeric_limits<T>::epsilon()) {
                    cavity.create(eh);
                }
            }
        }
    };

    Query<blockThreads> query(context, pid);
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, is_delaunay);
    block.sync();

    // create the cavity
    if (cavity.prologue(block, shrd_alloc)) {

        // update the cavity
        cavity.update_attributes(block, coords);

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            DEdgeHandle new_edge = cavity.add_edge(
                cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));

            cavity.add_face(cavity.get_cavity_edge(c, 0),
                            new_edge,
                            cavity.get_cavity_edge(c, 3));

            cavity.add_face(cavity.get_cavity_edge(c, 1),
                            cavity.get_cavity_edge(c, 2),
                            new_edge.get_flip_dedge());
        });

        if (threadIdx.x == 0) {
            printf("\n p= %u", cavity.patch_id());
        }
    }
    cavity.epilogue(block);
}

inline bool delaunay_rxmesh(rxmesh::RXMeshDynamic& rx)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();
    const uint32_t num_faces    = rx.get_num_faces();

#if USE_POLYSCOPE
    rx.polyscope_render_vertex_patch();
    rx.polyscope_render_edge_patch();
    rx.polyscope_render_face_patch();
#endif

    auto coords = rx.get_input_vertex_coordinates();

    EXPECT_TRUE(rx.validate());

    GPUTimer timer;
    timer.start();
    int iter = 0;
    while (!rx.is_queue_empty()) {
        RXMESH_INFO("\niter = {}", iter++);
        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box({Op::EVDiamond},
                              launch_box,
                              (void*)delaunay_edge_flip<float, blockThreads>);
        delaunay_edge_flip<float, blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords);
        rx.slice_patches(*coords);
        rx.cleanup();
    }

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

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->setEnabled(false);

    rx.polyscope_render_vertex_patch();
    rx.polyscope_render_edge_patch();
    rx.polyscope_render_face_patch();

    polyscope::show();
#endif

    return true;
}