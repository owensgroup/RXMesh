#include <cuda_profiler_api.h>
#include <glm/glm.hpp>
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/report.h"

#include "../common/openmesh_trimesh.h"

#include <cmath>

#include "mcf_rxmesh.h"

template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    delaunay_edge_flip(rxmesh::Context            context,
                       rxmesh::VertexAttribute<T> coords,
                       int*                       d_flipped,
                       uint32_t*                  num_successful,
                       uint32_t*                  num_sliced)
{
    using namespace rxmesh;
    using vec3           = glm::vec<3, T, glm::defaultp>;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, false);

    const uint32_t pid = cavity.patch_id();

    if (pid == INVALID32) {
        return;
    }

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // for each edge we want to flip, we its id in one of its opposite vertices
    // along with the other opposite vertex
    uint16_t* v_info =
        shrd_alloc.alloc<uint16_t>(2 * cavity.patch_info().num_vertices[0]);
    fill_n<blockThreads>(
        v_info, 2 * cavity.patch_info().num_vertices[0], uint16_t(INVALID16));

    // a bitmask that indicates which edge we want to flip
    Bitmask e_flip(cavity.patch_info().num_edges[0], shrd_alloc);
    e_flip.reset(block);

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

            if (v0 == v1 || v0 == v2 || v0 == v3 ||v1 == v2 || v1 == v3 ||
                v2 == v3) {
                return;
            }

            constexpr T PII = 3.14159265358979323f;

            const vec3 V0 = coords.to_glm<3>(v0);
            const vec3 V1 = coords.to_glm<3>(v1);
            const vec3 V2 = coords.to_glm<3>(v2);
            const vec3 V3 = coords.to_glm<3>(v3);            

            // find the angle between S, M, Q vertices (i.e., angle at M)
            auto angle_between_three_vertices = [](const vec3& S,
                                                   const vec3& M,
                                                   const vec3& Q) {
                vec3 p1      = S - M;
                vec3 p2      = Q - M;
                T    dot_pro = glm::dot(p1, p2);
                if constexpr (std::is_same_v<T, float>) {
                    return acosf(dot_pro / (glm::length(p1) * glm::length(p2)));
                } else {
                    return acos(dot_pro / (glm::length(p1) * glm::length(p2)));
                }
            };

            // first check if the edge formed by v0-v1 is a delaunay edge
            // where v2 and v3 are the opposite vertices to the edge
            /*
                0
              / | \
             3  |  2
             \  |  /
                1
            */
            // if not delaunay, then we check if flipping it won't create a
            // foldover The case below would create a fold over
            /*
                  0
                / | \
               /  1  \
              / /  \  \
              2       3
            */


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
                    // cavity.create(eh);
                    e_flip.set(eh.local_id(), true);
                }
            }
        }
    };

    // 1. mark edge that we want to flip
    Query<blockThreads> query(context, pid);
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, is_delaunay);
    block.sync();

    // 2. make sure that the two vertices opposite to a flipped edge are not
    // connected
    auto check_edges = [&](const VertexHandle& vh, const VertexIterator& iter) {
        uint16_t opposite_v = v_info[2 * vh.local_id()];
        if (opposite_v != INVALID16) {
            bool is_valid = true;
            for (uint16_t v = 0; v < iter.size(); ++v) {
                if (iter.local(v) == opposite_v) {
                    is_valid = false;
                    break;
                }
            }
            if (!is_valid) {
                e_flip.reset(v_info[2 * vh.local_id() + 1], true);
            }
        }
    };
    query.dispatch<Op::VV>(block, shrd_alloc, check_edges);
    block.sync();

    // 3. create cavity for the surviving edges
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (e_flip(eh.local_id())) {
            cavity.create(eh);
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    // create cavities
    if (cavity.prologue(block, shrd_alloc, coords)) {

        if (threadIdx.x == 0) {
            ::atomicAdd(num_successful, 1);
        }

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

            d_flipped[0] = 1;
        });
    }

    // if (threadIdx.x == 0) {
    //     if (cavity.should_slice()) {
    //         ::atomicAdd(num_sliced, 1);
    //     }
    // }

    cavity.epilogue(block);
}


inline uint32_t count_non_delaunay_edges(TriMesh& mesh)
{
    // verify that mesh is a delaunay mesh
    uint32_t num_non_delaunay = 0;
    for (TriMesh::VertexIter v0_it = mesh.vertices_begin();
         v0_it != mesh.vertices_end();
         ++v0_it) {
        /*
            0
          / | \
         3  |  2
         \  |  /
            1
        */
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

    // Report
    Report report("Delaunay_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name + "_before", rx, "model_before");
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();
    const uint32_t num_faces    = rx.get_num_faces();


    MCFData mcf_data_before = mcf_rxmesh_cg<float>(rx, false);
    report.add_member("mcf_before_time", mcf_data_before.total_time);
    report.add_member("mcf_before_num_iter", mcf_data_before.num_iter);
    report.add_member("mcf_before_matvec_time", mcf_data_before.matvec_time);
    report.add_member(
        "mcf_before_time_per_iter",
        mcf_data_before.total_time / float(mcf_data_before.num_iter));

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

    RXMESH_INFO("Input mesh #Vertices {}", rx.get_num_vertices());
    RXMESH_INFO("Input mesh #Edges {}", rx.get_num_edges());
    RXMESH_INFO("Input mesh #Faces {}", rx.get_num_faces());
    RXMESH_INFO("Input mesh #Patches {}", rx.get_num_patches());

    CUDA_ERROR(cudaProfilerStart());

    size_t   max_smem_bytes_dyn           = 0;
    size_t   max_smem_bytes_static        = 0;
    uint32_t max_num_registers_per_thread = 0;
    uint32_t max_num_blocks               = 0;

    GPUTimer timer;
    timer.start();

    while (h_flipped != 0) {
        CUDA_ERROR(cudaMemset(d_flipped, 0, sizeof(int)));

        h_flipped = 0;
        rx.reset_scheduler();
        int inner_iter = 0;
        while (!rx.is_queue_empty()) {
            LaunchBox<blockThreads> launch_box;
            rx.update_launch_box(
                {Op::EVDiamond, Op::VV},
                launch_box,
                (void*)delaunay_edge_flip<float, blockThreads>,
                true,
                false,
                false,
                false,
                [&](uint32_t v, uint32_t e, uint32_t f) {
                    return detail::mask_num_bytes(e) +
                           2 * v * sizeof(uint16_t) +
                           2 * ShmemAllocator::default_alignment;
                });

            max_smem_bytes_dyn =
                std::max(max_smem_bytes_dyn, launch_box.smem_bytes_dyn);
            max_smem_bytes_static =
                std::max(max_smem_bytes_static, launch_box.smem_bytes_static);
            max_num_registers_per_thread =
                std::max(max_num_registers_per_thread,
                         launch_box.num_registers_per_thread);
            max_num_blocks =
                std::max(max_num_blocks, DIVIDE_UP(launch_box.blocks, 8));

            GPUTimer app_timer;
            app_timer.start();
            delaunay_edge_flip<float, blockThreads>
                <<<DIVIDE_UP(launch_box.blocks, 8),
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *coords,
                                                d_flipped,
                                                d_num_successful,
                                                d_num_sliced);
            app_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords);
            slice_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();

            app_time += app_timer.elapsed_millis();
            slice_time += slice_timer.elapsed_millis();
            cleanup_time += cleanup_timer.elapsed_millis();

            // uint32_t h_num_successful, h_num_sliced;
            // CUDA_ERROR(cudaMemcpy(&h_num_successful,
            //                       d_num_successful,
            //                       sizeof(uint32_t),
            //                       cudaMemcpyDeviceToHost));
            //
            // CUDA_ERROR(cudaMemcpy(&h_num_sliced,
            //                       d_num_sliced,
            //                       sizeof(uint32_t),
            //                       cudaMemcpyDeviceToHost));
            //
            // RXMESH_INFO("num_patches = {}, num_successful= {}, num_sliced =
            // {}",
            //             rx.get_num_patches(),
            //             h_num_successful,
            //             h_num_sliced);
            //  break;
        }
        CUDA_ERROR(cudaMemcpy(
            &h_flipped, d_flipped, sizeof(int), cudaMemcpyDeviceToHost));
        // break;
        outer_iter++;
    }
    timer.stop();
    total_time = timer.elapsed_millis();

    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("delaunay_rxmesh() RXMesh Delaunay Edge Flip took {} (ms)",
                total_time);
    RXMESH_INFO("delaunay_rxmesh() App time {} (ms)", app_time);
    RXMESH_INFO("delaunay_rxmesh() Slice timer {} (ms)", slice_time);
    RXMESH_INFO("delaunay_rxmesh() Cleanup timer {} (ms)", cleanup_time);


    rx.update_host();

    report.add_member("delaunay_edge_flip_time", total_time);
    report.add_member("delaunay_edge_flip_app_time", app_time);
    report.add_member("delaunay_edge_flip_slice_time", slice_time);
    report.add_member("delaunay_edge_flip_cleanup_time", cleanup_time);

    report.add_member("max_smem_bytes_dyn", max_smem_bytes_dyn);
    report.add_member("max_smem_bytes_static", max_smem_bytes_static);
    report.add_member("max_num_registers_per_thread",
                      max_num_registers_per_thread);
    report.add_member("max_num_blocks", max_num_blocks);


    report.model_data(Arg.obj_file_name + "_after", rx, "model_after");

    coords->move(DEVICE, HOST);

    EXPECT_EQ(num_vertices, rx.get_num_vertices());
    EXPECT_EQ(num_edges, rx.get_num_edges());
    EXPECT_EQ(num_faces, rx.get_num_faces());

    RXMESH_INFO("Output mesh #Vertices {}", rx.get_num_vertices());
    RXMESH_INFO("Output mesh #Edges {}", rx.get_num_edges());
    RXMESH_INFO("Output mesh #Faces {}", rx.get_num_faces());
    RXMESH_INFO("Output mesh #Patches {}", rx.get_num_patches());

    report.add_member("attributes_memory_mg", coords->get_memory_mg());

    if (with_verify) {
        rx.export_obj(STRINGIFY(OUTPUT_DIR) "temp.obj", *coords);
        TriMesh tri_mesh;
        ASSERT_TRUE(OpenMesh::IO::read_mesh(tri_mesh,
                                            STRINGIFY(OUTPUT_DIR) "temp.obj"));
        int num_non_del = count_non_delaunay_edges(tri_mesh);
        EXPECT_EQ(num_non_del, 0);
        report.add_member("after_num_non_delaunay_edges", num_non_del);
    }

#if USE_POLYSCOPE
    rx.update_polyscope();
    rx.get_polyscope_mesh()->updateVertexPositions(*coords);
    rx.get_polyscope_mesh()->setEnabled(false);
#endif

    MCFData mcf_data_after = mcf_rxmesh_cg<float>(rx, true);
    report.add_member("mcf_after_time", mcf_data_after.total_time);
    report.add_member("mcf_after_num_iter", mcf_data_after.num_iter);
    report.add_member("mcf_after_matvec_time", mcf_data_after.matvec_time);
    report.add_member(
        "mcf_after_time_per_iter",
        mcf_data_after.total_time / float(mcf_data_after.num_iter));


#if USE_POLYSCOPE
    rx.update_polyscope();
    rx.get_polyscope_mesh()->updateVertexPositions(*coords);
    rx.get_polyscope_mesh()->setEnabled(false);

    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    polyscope::show();
#endif
    CUDA_ERROR(cudaFree(d_flipped));
    CUDA_ERROR(cudaFree(d_num_successful));
    CUDA_ERROR(cudaFree(d_num_sliced));

    report.write(Arg.output_folder + "/rxmesh_delaunay",
                 "Delaunay_RXMesh_" + extract_file_name(Arg.obj_file_name));
}