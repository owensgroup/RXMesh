#pragma once
#include "geodesic_kernel.cuh"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

constexpr float EPS = 10e-6;

template <typename T>
inline void geodesic_rxmesh(rxmesh::RXMeshStatic&               rxmesh,
                            std::vector<std::vector<uint32_t>>& Faces,
                            std::vector<std::vector<T>>&        Verts,
                            const std::vector<uint32_t>&        h_seeds,
                            const std::vector<uint32_t>&        h_sorted_index,
                            const std::vector<uint32_t>&        h_limits,
                            const std::vector<uint32_t>&        toplesets)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    // Report
    Report report("Geodesic_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rxmesh);
    report.add_member("seeds", h_seeds);
    report.add_member("method", std::string("RXMesh"));

    // input coords
    auto input_coord = rxmesh.add_vertex_attribute(Verts, "coord");

    //TODO copy toplesets
    auto d_toplesets = rxmesh.add_vertex_attribute<uint32_t>("topleset", 1);


    // RXMesh launch box
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box(rxmesh::Op::VV, launch_box, false, true);


    // Geodesic distance attribute for all vertices (seeds set to zero
    // and infinity otherwise)
    RXMeshAttribute<T> rxmesh_geo;
    rxmesh_geo.init(rxmesh.get_num_vertices(), 1u, rxmesh::LOCATION_ALL);
    rxmesh_geo.reset(std::numeric_limits<T>::infinity(), rxmesh::HOST);
    for (uint32_t v : h_seeds) {
        rxmesh_geo(v) = 0;
    }
    rxmesh_geo.move(rxmesh::HOST, rxmesh::DEVICE);

    // second buffer for geodesic distance for double buffering
    RXMeshAttribute<T> rxmesh_geo_2;
    rxmesh_geo_2.init(rxmesh.get_num_vertices(), 1u, rxmesh::DEVICE);
    rxmesh_geo_2.copy(rxmesh_geo, rxmesh::DEVICE, rxmesh::DEVICE);


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
    uint32_t iter     = 0;
    uint32_t max_iter = 2 * h_limits.size();
    while (i < j && iter < max_iter) {
        iter++;
        if (i < (j / 2)) {
            i = j / 2;
        }

        // compute new geodesic
        relax_ptp_rxmesh<T, blockThreads>
            <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
                rxmesh.get_context(),
                *input_coord,
                *double_buffer[!d],
                *double_buffer[d],
                *d_toplesets,
                i,
                j,
                d_error,
                std::numeric_limits<T>::infinity(),
                T(1e-3));

        CUDA_ERROR(cudaMemcpy(
            &h_error, d_error, sizeof(uint32_t), cudaMemcpyDeviceToHost));
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
    rxmesh_geo.copy(*double_buffer[d], rxmesh::DEVICE, rxmesh::HOST);
    T err = 0;


    RXMESH_TRACE("Geodesic_RXMesh took {} (ms) -- #iter= {}",
                 timer.elapsed_millis(),
                 iter);

    // export_attribute_VTK("geo_rxmesh.vtk", Faces, Verts, false,
    //                     rxmesh_geo.operator->(), rxmesh_geo.operator->());

    // Release allocation
    rxmesh_geo.release();
    rxmesh_geo_2.release();
    GPU_FREE(d_error);

    // Finalize report
    report.add_member("num_iter_taken", iter);
    TestData td;
    td.test_name = "Geodesic";
    td.time_ms.push_back(timer.elapsed_millis());
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "Geodesic_RXMesh_" + extract_file_name(Arg.obj_file_name));
}