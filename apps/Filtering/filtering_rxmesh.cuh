#pragma once

#include <cuda_profiler_api.h>

#include "filtering_rxmesh_kernel.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

/**
 * filtering_rxmesh()
 */
template <typename T, uint32_t patchSize>
void filtering_rxmesh(RXMESH::RXMeshStatic<patchSize>&  rxmesh_static,
                      std::vector<std::vector<T>>&      Verts,
                      const RXMESH::RXMeshAttribute<T>& ground_truth,
                      const size_t                      max_neighbour_size)
{
    using namespace RXMESH;

    constexpr uint32_t maxVVSize = 20 * 4;

    ASSERT_GE(maxVVSize, max_neighbour_size)
        << " Max #neighbour per vertex (as per OpenMesh calculation) is "
           "greater than maxVVSize. Should increase maxVVSize to "
        << max_neighbour_size << " to avoid illegal memory access";

    // Report
    Report report("Filtering_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rxmesh_static);
    report.add_member("method", std::string("RXMesh"));
    std::string order = "default";
    if (Arg.shuffle) {
        order = "shuffle";
    } else if (Arg.sort) {
        order = "sorted";
    }
    report.add_member("input_order", order);
    report.add_member("num_filter_iter", Arg.num_filter_iter);


    // input coords
    RXMeshAttribute<T> coords;
    coords.set_name("coords");
    coords.init(rxmesh_static.get_num_vertices(), 3u, RXMESH::LOCATION_ALL);
    for (uint32_t i = 0; i < Verts.size(); ++i) {
        for (uint32_t j = 0; j < Verts[i].size(); ++j) {
            coords(i, j) = Verts[i][j];
        }
    }
    coords.move(RXMESH::HOST, RXMESH::DEVICE);

    // Vertex normals (only on device)
    RXMeshAttribute<T> vertex_normal;
    vertex_normal.set_name("vertex_normal");
    vertex_normal.init(rxmesh_static.get_num_vertices(), 3u, RXMESH::DEVICE);
    vertex_normal.reset(0.0, RXMESH::DEVICE);


    // Filtered coordinates
    RXMeshAttribute<T> filtered_coord;
    filtered_coord.set_name("filtered_coord");
    filtered_coord.init(
        rxmesh_static.get_num_vertices(), 3u, RXMESH::LOCATION_ALL);
    filtered_coord.reset(0.0, RXMESH::LOCATION_ALL);
    filtered_coord.move(RXMESH::HOST, RXMESH::DEVICE);

    // vertex normal launch box
    constexpr uint32_t          vn_block_threads = 256;
    LaunchBox<vn_block_threads> vn_launch_box;
    rxmesh_static.prepare_launch_box(RXMESH::Op::FV, vn_launch_box);

    // filter launch box
    constexpr uint32_t              filter_block_threads = 512;
    LaunchBox<filter_block_threads> filter_launch_box;
    rxmesh_static.prepare_launch_box(RXMESH::Op::VV, filter_launch_box, true);

    // double buffer
    RXMeshAttribute<T>* double_buffer[2] = {&coords, &filtered_coord};

    cudaStream_t stream;
    CUDA_ERROR(cudaStreamCreate(&stream));
    CUDA_ERROR(cudaProfilerStart());
    GPUTimer timer;
    timer.start();
    uint32_t d = 0;

    for (uint32_t itr = 0; itr < Arg.num_filter_iter; ++itr) {
        vertex_normal.reset(0, RXMESH::DEVICE, stream);

        // update vertex normal before filtering
        compute_vertex_normal<T, vn_block_threads>
            <<<vn_launch_box.blocks,
               vn_block_threads,
               vn_launch_box.smem_bytes_dyn,
               stream>>>(
                rxmesh_static.get_context(), *double_buffer[d], vertex_normal);

        bilateral_filtering<T, filter_block_threads, maxVVSize>
            <<<filter_launch_box.blocks,
               filter_block_threads,
               filter_launch_box.smem_bytes_dyn,
               stream>>>(rxmesh_static.get_context(),
                         *double_buffer[d],
                         *double_buffer[!d],
                         vertex_normal);

        d = !d;
        CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaProfilerStop());
    CUDA_ERROR(cudaStreamDestroy(stream));
    RXMESH_TRACE("filtering_rxmesh() took {} (ms) (i.e., {} ms/iter) ",
                 timer.elapsed_millis(),
                 timer.elapsed_millis() / float(Arg.num_filter_iter));

    // move output to host
    coords.copy(*double_buffer[d], RXMESH::DEVICE, RXMESH::HOST);

    // output to obj
    // rxmesh_static.exportOBJ(
    //   "output_rxmesh" + std::to_string(Arg.num_filter_iter) + ".obj",
    //   [&](uint32_t i, uint32_t j) { return coords(i, j); });


    // Verify
    bool    passed = true;
    const T tol    = 0.01;
    for (uint32_t v = 0; v < coords.get_num_mesh_elements(); ++v) {
        const Vector<3, T> gt(
            ground_truth(v, 0), ground_truth(v, 1), ground_truth(v, 2));
        const Vector<3, T> co(coords(v, 0), coords(v, 1), coords(v, 2));

        if (std::fabs(co[0] - gt[0]) > tol || std::fabs(co[1] - gt[1]) > tol ||
            std::fabs(co[2] - gt[2]) > tol) {
            passed = false;
            break;
        }
    }

    EXPECT_TRUE(passed);

    // Release allocation
    filtered_coord.release();
    coords.release();
    vertex_normal.release();

    // Finalize report
    TestData td;
    td.test_name = "Filtering";
    td.passed.push_back(passed);
    td.time_ms.push_back(timer.elapsed_millis());
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "Filtering_RXMesh_" + extract_file_name(Arg.obj_file_name));
}