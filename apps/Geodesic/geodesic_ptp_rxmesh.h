#pragma once
#include "geodesic_kernel.cuh"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

constexpr float EPS = 10e-6;

template <typename T>
inline void geodesic_rxmesh(rxmesh::RXMeshStatic&        rxmesh,
                            const std::vector<uint32_t>& h_seeds,
                            const std::vector<uint32_t>& h_sorted_index,
                            const std::vector<uint32_t>& h_limits,
                            const std::vector<uint32_t>& toplesets)
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
    auto input_coord = rxmesh.get_input_vertex_coordinates();

    // toplesets
    auto d_toplesets = rxmesh.add_vertex_attribute(toplesets, "topleset");


    // RXMesh launch box
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box({rxmesh::Op::VV},
                              launch_box,
                              (void*)relax_ptp_rxmesh<T, blockThreads>,
                              true);


    // Geodesic distance attribute for all vertices (seeds set to zero
    // and infinity otherwise)
    auto rxmesh_geo = rxmesh.add_vertex_attribute<T>("geo", 1u);
    rxmesh_geo->reset(std::numeric_limits<T>::infinity(), rxmesh::HOST);
    rxmesh.for_each_vertex(rxmesh::HOST, [&](const VertexHandle vh) {
        uint32_t v_id = rxmesh.map_to_global(vh);
        for (uint32_t s : h_seeds) {
            if (s == v_id) {
                (*rxmesh_geo)(vh) = 0;
                break;
            }
        }
    });
    rxmesh_geo->move(rxmesh::HOST, rxmesh::DEVICE);

    // second buffer for geodesic distance for double buffering
    auto rxmesh_geo_2 =
        rxmesh.add_vertex_attribute<T>("geo2", 1u, rxmesh::DEVICE);

    rxmesh_geo_2->copy_from(*rxmesh_geo, rxmesh::DEVICE, rxmesh::DEVICE);


    // Error
    uint32_t *d_error(nullptr), h_error(0);
    CUDA_ERROR(cudaMalloc((void**)&d_error, sizeof(uint32_t)));

    // double buffer
    VertexAttribute<T>* double_buffer[2] = {rxmesh_geo.get(),
                                            rxmesh_geo_2.get()};

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

    rxmesh_geo->copy_from(*double_buffer[d], rxmesh::DEVICE, rxmesh::HOST);

    RXMESH_TRACE("Geodesic_RXMesh took {} (ms) -- #iter= {}",
                 timer.elapsed_millis(),
                 iter);

    // std::vector<T> geo(rxmesh.get_num_vertices());
    // rxmesh.for_each_vertex(rxmesh::HOST, [&](const VertexHandle vh) {
    //    uint32_t v_id = rxmesh.map_to_global(vh);
    //    geo[v_id]     = (*rxmesh_geo)(vh);
    //});


    GPU_FREE(d_error);

    // Finalize report
    report.add_member("num_iter_taken", iter);
    TestData td;
    td.test_name   = "Geodesic";
    td.num_threads = launch_box.num_threads;
    td.num_blocks  = launch_box.blocks;
    td.dyn_smem    = launch_box.smem_bytes_dyn;
    td.static_smem = launch_box.smem_bytes_static;
    td.num_reg     = launch_box.num_registers_per_thread;
    td.time_ms.push_back(timer.elapsed_millis());
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "Geodesic_RXMesh_" + extract_file_name(Arg.obj_file_name));
}