#pragma once
#include "geodesic_kernel.cuh"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

constexpr float EPS = 10e-6;

template <typename T, uint32_t patchSize>
inline bool geodesic_rxmesh(RXMESH::RXMeshStatic<patchSize>&    rxmesh_static,
                            std::vector<std::vector<uint32_t>>& Faces,
                            std::vector<std::vector<T>>&        Verts,
                            const std::vector<uint32_t>&        h_seeds,
                            const RXMESH::RXMeshAttribute<T>&   ground_truth,
                            const std::vector<uint32_t>&        h_sorted_index,
                            const std::vector<uint32_t>&        h_limits,
                            const RXMESH::RXMeshAttribute<uint32_t>& toplesets)
{
    using namespace RXMESH;
    constexpr uint32_t blockThreads = 256;

    // Report
    Report report("Geodesic_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rxmesh_static);
    report.add_member("seeds", h_seeds);
    report.add_member("method", std::string("RXMesh"));
    std::string order = "default";
    if (Arg.shuffle) {
        order = "shuffle";
    } else if (Arg.sort) {
        order = "sorted";
    }
    report.add_member("input_order", order);


    // input coords
    RXMESH::RXMeshAttribute<T> input_coord;
    input_coord.set_name("coord");
    input_coord.init(Verts.size(), 3u, RXMESH::LOCATION_ALL);
    for (uint32_t i = 0; i < Verts.size(); ++i) {
        for (uint32_t j = 0; j < Verts[i].size(); ++j) {
            input_coord(i, j) = Verts[i][j];
        }
    }
    input_coord.change_layout(RXMESH::HOST);
    input_coord.move(RXMESH::HOST, RXMESH::DEVICE);

    // RXMesh launch box
    LaunchBox<blockThreads> launch_box;
    rxmesh_static.prepare_launch_box(RXMESH::Op::VV, launch_box, true);


    // Geodesic distance attribute for all vertices (seeds set to zero
    // and infinity otherwise)
    RXMeshAttribute<T> rxmesh_geo;
    rxmesh_geo.init(rxmesh_static.get_num_vertices(), 1u, RXMESH::LOCATION_ALL);
    rxmesh_geo.reset(std::numeric_limits<T>::infinity(), RXMESH::HOST);
    for (uint32_t v : h_seeds) {
        rxmesh_geo(v) = 0;
    }
    rxmesh_geo.move(RXMESH::HOST, RXMESH::DEVICE);

    // second buffer for geodesic distance for double buffering
    RXMeshAttribute<T> rxmesh_geo_2;
    rxmesh_geo_2.init(rxmesh_static.get_num_vertices(), 1u, RXMESH::DEVICE);
    rxmesh_geo_2.copy(rxmesh_geo, RXMESH::DEVICE, RXMESH::DEVICE);


    // Error
    uint32_t *d_error(nullptr), h_error(0);
    CUDA_ERROR(cudaMalloc((void**)&d_error, sizeof(uint32_t)));

    // double buffer
    RXMeshAttribute<T>* double_buffer[2] = {&rxmesh_geo, &rxmesh_geo_2};

    // start time
    GPUTimer timer;
    timer.start();

    // actual computation
    uint32_t d = 0;
    uint32_t i(1), j(2);
    uint32_t iter = 0;
    uint32_t max_iter = 2 * h_limits.size();
    while (i < j && iter < max_iter) {
        iter++;
        if (i < (j / 2)) {
            i = j / 2;
        }

        // compute new geodesic
        relax_ptp_rxmesh<T, blockThreads>
            <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
                rxmesh_static.get_context(), input_coord, *double_buffer[!d],
                *double_buffer[d], toplesets, i, j, d_error,
                std::numeric_limits<T>::infinity(), T(1e-3));

        CUDA_ERROR(cudaMemcpy(&h_error, d_error, sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemset(d_error, 0, sizeof(uint32_t)));


        const uint32_t n_cond = h_limits[i + 1] - h_limits[i];

        if (n_cond == h_error) {
            i++;
        }
        if (j < h_limits.size() - 1) {
            j++;
        }

        d = !d;
    }

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaProfilerStop());

    // verify
    rxmesh_geo.copy(*double_buffer[d], RXMESH::DEVICE, RXMESH::HOST);
    T err = 0;
    for (uint32_t i = 0; i < ground_truth.get_num_mesh_elements(); ++i) {
        if (ground_truth(i) > EPS) {
            err += std::abs(rxmesh_geo(i) - ground_truth(i)) / ground_truth(i);
        }
    }
    err /= T(ground_truth.get_num_mesh_elements());
    bool is_passed = (err < 10E-2);

    RXMESH_TRACE("Geodesic_RXMesh took {} (ms) -- err= {} -- #iter= {}",
                 timer.elapsed_millis(), err, iter);

    // export_attribute_VTK("geo_rxmesh.vtk", Faces, Verts, false,
    //                     rxmesh_geo.operator->(), rxmesh_geo.operator->());

    // Release allocation
    rxmesh_geo.release();
    rxmesh_geo_2.release();
    input_coord.release();
    GPU_FREE(d_error);

    // Finalize report
    report.add_member("num_iter_taken", iter);
    TestData td;
    td.test_name = "Geodesic";
    td.time_ms.push_back(timer.elapsed_millis());
    td.passed.push_back(is_passed);
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "Geodesic_RXMesh_" + extract_file_name(Arg.obj_file_name));

    return is_passed;
}