#pragma once

#include <cuda_profiler_api.h>

#include "filtering_rxmesh_kernel.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

/**
 * filtering_rxmesh()
 */
template <typename T>
void filtering_rxmesh(std::vector<std::vector<uint32_t>>& Faces,
                      const std::vector<std::vector<T>>&  Verts,
                      const std::vector<std::vector<T>>&  ground_truth,
                      const size_t                        max_neighbour_size)
{
    using namespace rxmesh;

    constexpr uint32_t maxVVSize = 20 * 4;

    ASSERT_GE(maxVVSize, max_neighbour_size)
        << " Max #neighbour per vertex (as per OpenMesh calculation) is "
           "greater than maxVVSize. Should increase maxVVSize to "
        << max_neighbour_size << " to avoid illegal memory access";

    RXMeshStatic rxmesh(Faces, false);

    // Report
    Report report("Filtering_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rxmesh);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("num_filter_iter", Arg.num_filter_iter);


    // input coords
    auto coords = rxmesh.add_vertex_attribute(Verts, "coords");

    // Vertex normals (only on device)
    auto vertex_normal = rxmesh.add_vertex_attribute<T>("vn", 3, DEVICE);
    vertex_normal->reset_v1(0, DEVICE);


    // Filtered coordinates
    auto filtered_coord = rxmesh.add_vertex_attribute<T>("vn", 3, LOCATION_ALL);
    filtered_coord->reset_v1(0, LOCATION_ALL);

    // vertex normal launch box
    constexpr uint32_t          vn_block_threads = 256;
    LaunchBox<vn_block_threads> vn_launch_box;
    rxmesh.prepare_launch_box(rxmesh::Op::FV, vn_launch_box);

    // filter launch box
    constexpr uint32_t              filter_block_threads = 512;
    LaunchBox<filter_block_threads> filter_launch_box;
    rxmesh.prepare_launch_box(rxmesh::Op::VV, filter_launch_box, true);

    // double buffer
    RXMeshAttribute<T>* double_buffer[2] = {coords.get(), filtered_coord.get()};

    CUDA_ERROR(cudaProfilerStart());
    GPUTimer timer;
    timer.start();
    uint32_t d = 0;

    for (uint32_t itr = 0; itr < Arg.num_filter_iter; ++itr) {
        vertex_normal->reset_v1(0, rxmesh::DEVICE);

        // update vertex normal before filtering
        compute_vertex_normal<T, vn_block_threads>
            <<<vn_launch_box.blocks,
               vn_block_threads,
               vn_launch_box.smem_bytes_dyn>>>(
                rxmesh.get_context(), *double_buffer[d], vertex_normal);

        bilateral_filtering<T, filter_block_threads, maxVVSize>
            <<<filter_launch_box.blocks,
               filter_block_threads,
               filter_launch_box.smem_bytes_dyn>>>(rxmesh.get_context(),
                                                   *double_buffer[d],
                                                   *double_buffer[!d],
                                                   vertex_normal);

        d = !d;
        CUDA_ERROR(cudaDeviceSynchronize());
    }

    timer.stop();
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaProfilerStop());
    RXMESH_TRACE("filtering_rxmesh() took {} (ms) (i.e., {} ms/iter) ",
                 timer.elapsed_millis(),
                 timer.elapsed_millis() / float(Arg.num_filter_iter));

    // move output to host
    coords->copy_from(*double_buffer[d], rxmesh::DEVICE, rxmesh::HOST);

    // output to obj
    // rxmesh.export_obj(
    //   "output_rxmesh" + std::to_string(Arg.num_filter_iter) + ".obj",
    //   [&](uint32_t i, uint32_t j) { return coords(i, j); });


    // Verify
    const T tol = 0.01;
    rxmesh.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t           v_id = rxmesh.map_to_global(vh);
        const Vector<3, T> gt(ground_truth[v_id][0],
                              ground_truth[v_id][1],
                              ground_truth[v_id][2]);
        const Vector<3, T> co(
            (*coords)(vh, 0), (*coords)(vh, 1), (*coords)(vh, 2));

        EXPECT_LT(std::fabs((*coords)(vh, 0) - ground_truth[v_id][0]), tol);
        EXPECT_LT(std::fabs((*coords)(vh, 1) - ground_truth[v_id][1]), tol);
        EXPECT_LT(std::fabs((*coords)(vh, 2) - ground_truth[v_id][2]), tol);
    });

    // Finalize report
    TestData td;
    td.test_name = "Filtering";
    td.time_ms.push_back(timer.elapsed_millis());
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "Filtering_RXMesh_" + extract_file_name(Arg.obj_file_name));
}