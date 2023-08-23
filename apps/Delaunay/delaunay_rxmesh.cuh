#include <cuda_profiler_api.h>
#include <glm/glm.hpp>
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"


#include "../common/openmesh_trimesh.h"

#include <cmath>

template <typename T, uint32_t blockThreads>
__global__ static void delaunay_edge_flip(rxmesh::Context            context,
                                          rxmesh::VertexAttribute<T> coords,
                                          int*                       d_flipped,
                                          uint32_t* num_successful,
                                          uint32_t* num_sliced,
                                          uint32_t  current_p)
{
    using namespace rxmesh;
    using VecT           = glm::vec<3, T, glm::defaultp>;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, current_p);

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

            constexpr T PII = 3.14159265358979323f;

            const VecT V0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
            const VecT V1(coords(v1, 0), coords(v1, 1), coords(v1, 2));
            const VecT V2(coords(v2, 0), coords(v2, 1), coords(v2, 2));
            const VecT V3(coords(v3, 0), coords(v3, 1), coords(v3, 2));

            // find the angle between S, M, Q vertices (i.e., angle at M)
            auto angle_between_three_vertices = [](const VecT& S,
                                                   const VecT& M,
                                                   const VecT& Q) {
                VecT p1      = S - M;
                VecT p2      = Q - M;
                T    dot_pro = glm::dot(p1, p2);
                if constexpr (std::is_same_v<T, float>) {
                    return acosf(dot_pro / (glm::length(p1) * glm::length(p2)));
                } else {
                    return acos(dot_pro / (glm::length(p1) * glm::length(p2)));
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

        if (threadIdx.x == 0) {
            ::atomicAdd(num_successful, 1);
        }

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
            d_flipped[0] = 1;
        });
    }

    if (threadIdx.x == 0) {
        if (cavity.should_slice()) {
            ::atomicAdd(num_sliced, 1);
        }
    }

    cavity.epilogue(block);
}


inline uint32_t count_non_delaunay_edges(TriMesh& mesh)
{
    // verify that mesh is a delaunay mesh
    uint32_t num_non_delaunay = 0;
    for (TriMesh::VertexIter v0_it = mesh.vertices_begin();
         v0_it != mesh.vertices_end();
         ++v0_it) {
        //    0
        //  / | \
        // 3  |  2
        // \  |  /
        //    1
        TriMesh::VertexHandle v0 = *v0_it;

        if (mesh.is_boundary(v0)) {
            continue;
        }

        TriMesh::Point p0 = mesh.point(v0);

        for (auto v1_it = mesh.vv_iter(*v0_it); v1_it.is_valid(); ++v1_it) {
            TriMesh::VertexHandle v1 = *v1_it;

            TriMesh::VertexHandle v2 = *(++v1_it);
            --v1_it;  // reset

            TriMesh::VertexHandle v3 = *(--v1_it);
            ++v1_it;  // reset


            TriMesh::Point p1 = mesh.point(v1);
            TriMesh::Point p2 = mesh.point(v2);
            TriMesh::Point p3 = mesh.point(v3);

            auto angle_between_three_vertices =
                [](TriMesh::Point a, TriMesh::Point b, TriMesh::Point c) {
                    auto la_sq = (b - c).sqrnorm();
                    auto lb_sq = (a - c).sqrnorm();
                    auto lc_sq = (a - b).sqrnorm();

                    return std::acos((la_sq + lc_sq - lb_sq) /
                                     (2.0 * std::sqrt(la_sq * lc_sq)));
                };

            float lambda = angle_between_three_vertices(p0, p2, p1);
            float gamma  = angle_between_three_vertices(p1, p3, p0);
            if (lambda + gamma - 0.00001 > M_PI) {
                num_non_delaunay++;
            }
        }
    }

    return num_non_delaunay;
}

inline void delaunay_rxmesh(rxmesh::RXMeshDynamic& rx, bool with_verify = true)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();
    const uint32_t num_faces    = rx.get_num_faces();

#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    // polyscope::show();
#endif

    auto coords = rx.get_input_vertex_coordinates();

    EXPECT_TRUE(rx.validate());

    int*      d_flipped        = nullptr;
    uint32_t* d_num_successful = nullptr;
    uint32_t* d_num_sliced     = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_flipped, sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&d_num_successful, sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_num_sliced, sizeof(uint32_t)));

    int h_flipped = 1;

    int outer_iter = 0;

    float total_time = 0;

    float app_time     = 0;
    float slice_time   = 0;
    float cleanup_time = 0;

    CUDA_ERROR(cudaProfilerStart());

    bool validate = false;
    while (h_flipped != 0) {
        CUDA_ERROR(cudaMemset(d_flipped, 0, sizeof(int)));
        CUDA_ERROR(cudaMemset(d_num_successful, 0, sizeof(uint32_t)));
        CUDA_ERROR(cudaMemset(d_num_sliced, 0, sizeof(uint32_t)));

        h_flipped = 0;
        rx.reset_scheduler();
        int inner_iter = 0;
        while (!rx.is_queue_empty()) {
            RXMESH_INFO("\n outer_iter= {}, inner_iter = {}, queue size= {}",
                        outer_iter,
                        inner_iter++,
                        rx.get_context().m_patch_scheduler.size());


            LaunchBox<blockThreads> launch_box;
            rx.prepare_launch_box(
                {Op::EVDiamond},
                launch_box,
                (void*)delaunay_edge_flip<float, blockThreads>);

            GPUTimer timer;
            timer.start();

            GPUTimer app_timer;
            app_timer.start();
            delaunay_edge_flip<float, blockThreads>
                <<<launch_box.blocks,
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *coords,
                                                d_flipped,
                                                d_num_successful,
                                                d_num_sliced,
                                                0);
            app_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords);
            slice_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();


            timer.stop();
            CUDA_ERROR(cudaDeviceSynchronize());
            CUDA_ERROR(cudaGetLastError());

            total_time += timer.elapsed_millis();
            app_time += app_timer.elapsed_millis();
            slice_time += slice_timer.elapsed_millis();
            cleanup_time += cleanup_timer.elapsed_millis();

            if (validate) {
                rx.update_host();
                EXPECT_TRUE(rx.validate());

                EXPECT_EQ(num_vertices, rx.get_num_vertices());
                EXPECT_EQ(num_edges, rx.get_num_edges());
                EXPECT_EQ(num_faces, rx.get_num_faces());
            }

            uint32_t h_num_successful, h_num_sliced;
            CUDA_ERROR(cudaMemcpy(&h_num_successful,
                                  d_num_successful,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            CUDA_ERROR(cudaMemcpy(&h_num_sliced,
                                  d_num_sliced,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            RXMESH_INFO("num_patches = {}, num_successful= {}, num_sliced = {}",
                        rx.get_num_patches(),
                        h_num_successful,
                        h_num_sliced);
            // break;
        }
        CUDA_ERROR(cudaMemcpy(
            &h_flipped, d_flipped, sizeof(int), cudaMemcpyDeviceToHost));
        // break;
        outer_iter++;
    }

    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("delaunay_rxmesh() RXMesh Delaunay Edge Flip took {} (ms)",
                total_time);
    RXMESH_INFO("delaunay_rxmesh() App time {} (ms)", app_time);
    RXMESH_INFO("delaunay_rxmesh() Slice timer {} (ms)", slice_time);
    RXMESH_INFO("delaunay_rxmesh() Cleanup timer {} (ms)", cleanup_time);

    if (!validate) {
        rx.update_host();
    }
    coords->move(DEVICE, HOST);

    EXPECT_EQ(num_vertices, rx.get_num_vertices());
    EXPECT_EQ(num_edges, rx.get_num_edges());
    EXPECT_EQ(num_faces, rx.get_num_faces());

    if (with_verify) {
        rx.export_obj(STRINGIFY(OUTPUT_DIR) "temp.obj", *coords);
        TriMesh tri_mesh;
        ASSERT_TRUE(OpenMesh::IO::read_mesh(tri_mesh,
                                            STRINGIFY(OUTPUT_DIR) "temp.obj"));
        EXPECT_EQ(count_non_delaunay_edges(tri_mesh), 0);
    }


#if USE_POLYSCOPE
    rx.update_polyscope();

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->setEnabled(false);

    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    polyscope::show();
#endif
    CUDA_ERROR(cudaFree(d_flipped));
    CUDA_ERROR(cudaFree(d_num_successful));
    CUDA_ERROR(cudaFree(d_num_sliced));
}