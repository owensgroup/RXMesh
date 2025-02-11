#pragma once

#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"

#include "histogram.cuh"
#include "rxmesh/util/report.h"
#include "sec_kernels.cuh"

inline void sec_rxmesh(rxmesh::RXMeshDynamic& rx,
                       const uint32_t         final_num_vertices)
{
    EXPECT_TRUE(rx.validate());

    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    rxmesh::Report report("ShortestEdgeCollapse_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name + "_before", rx, "model_before");
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);

    auto coords = rx.get_input_vertex_coordinates();

    LaunchBox<blockThreads> launch_box;

    Timers<GPUTimer> timers;
    timers.add("Total");
    timers.add("App");
    timers.add("Slice");
    timers.add("Cleanup");
    timers.add("Histo");


    float histo_time = 0;

    const int num_bins = 256;

    CostHistogram<float> histo(num_bins);

    RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
    RXMESH_INFO("#Edges {}", rx.get_num_edges());
    RXMESH_INFO("#Faces {}", rx.get_num_faces());
    RXMESH_INFO("#Patches {}", rx.get_num_patches());

    size_t   max_smem_bytes_dyn           = 0;
    size_t   max_smem_bytes_static        = 0;
    uint32_t max_num_registers_per_thread = 0;
    uint32_t max_num_blocks               = 0;

    bool validate = false;

    int num_passes = 0;

    CUDA_ERROR(cudaProfilerStart());

    timers.start("Total");
    while (rx.get_num_vertices(true) > final_num_vertices) {
        ++num_passes;

        timers.start("Histo");

        // compute max-min histogram
        histo.init();

        rx.update_launch_box({Op::EV},
                             launch_box,
                             (void*)compute_min_max_cost<float, blockThreads>,
                             false);
        compute_min_max_cost<float, blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, histo);

        // compute histogram bins
        rx.update_launch_box({Op::EV},
                             launch_box,
                             (void*)populate_histogram<float, blockThreads>,
                             false);
        populate_histogram<float, blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, histo);

        histo.scan();

        // how much we can reduce the number of edge at each iterations

        // loop over the mesh, and try to collapse
        const int num_edges_before = int(rx.get_num_edges(true));

        const int reduce_threshold =
            std::max(1, int(Arg.reduce_ratio * float(num_edges_before)));

        timers.stop("Histo");


        rx.reset_scheduler();
        while (!rx.is_queue_empty() &&
               rx.get_num_vertices(true) > final_num_vertices) {
            // RXMESH_INFO(" Queue size = {}, reduce_threshold = {}, #V= {}",
            //             rx.get_context().m_patch_scheduler.size(),
            //             reduce_threshold,
            //             rx.get_num_vertices(true));
            //
            rx.update_launch_box(
                {Op::EV},
                launch_box,
                (void*)sec<float, blockThreads>,
                true,
                false,
                false,
                false,
                [&](uint32_t v, uint32_t e, uint32_t f) {
                    return detail::mask_num_bytes(e) +
                           2 * detail::mask_num_bytes(v) +
                           3 * ShmemAllocator::default_alignment;
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

            timers.start("App");
            sec<float, blockThreads><<<launch_box.blocks,
                                       launch_box.num_threads,
                                       launch_box.smem_bytes_dyn>>>(
                rx.get_context(), *coords, histo, reduce_threshold);

            timers.stop("App");

            timers.start("Cleanup");
            rx.cleanup();
            timers.stop("Cleanup");

            // timers.start("Slice");
            // rx.slice_patches(*coords);
            // timers.stop("Slice");
            //
            // timers.start("Cleanup");
            // rx.cleanup();
            // timers.stop("Cleanup");
        }
    }
    timers.stop("Total");
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("sec_rxmesh() RXMesh SEC took {} (ms), num_passes= {}",
                timers.elapsed_millis("Total"),
                num_passes);
    RXMESH_INFO("sec_rxmesh() Histo time {} (ms)",
                timers.elapsed_millis("Histo"));
    RXMESH_INFO("sec_rxmesh() App time {} (ms)", timers.elapsed_millis("App"));
    RXMESH_INFO("sec_rxmesh() Slice timer {} (ms)",
                timers.elapsed_millis("Slice"));
    RXMESH_INFO("sec_rxmesh() Cleanup timer {} (ms)",
                timers.elapsed_millis("Cleanup"));

    RXMESH_INFO("#Vertices {}", rx.get_num_vertices(true));
    RXMESH_INFO("#Edges {}", rx.get_num_edges(true));
    RXMESH_INFO("#Faces {}", rx.get_num_faces(true));
    RXMESH_INFO("#Patches {}", rx.get_num_patches(true));


    rx.update_host();

    coords->move(DEVICE, HOST);

    report.add_member("num_passes", num_passes);
    report.add_member("max_smem_bytes_dyn", max_smem_bytes_dyn);
    report.add_member("max_smem_bytes_static", max_smem_bytes_static);
    report.add_member("max_num_registers_per_thread",
                      max_num_registers_per_thread);
    report.add_member("max_num_blocks", max_num_blocks);
    report.add_member("secs_remesh_time", timers.elapsed_millis("Total"));
    report.add_member("histogram_time", timers.elapsed_millis("Histo"));
    report.add_member("app_time", timers.elapsed_millis("App"));
    report.add_member("slice_time", timers.elapsed_millis("Slice"));
    report.add_member("cleanup_time", timers.elapsed_millis("Cleanup"));
    report.add_member("attributes_memory_mg", coords->get_memory_mg());
    report.model_data(Arg.obj_file_name + "_after", rx, "model_after");

#if USE_POLYSCOPE
    rx.update_polyscope();

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->setEnabled(false);

    // rx.render_vertex_patch();
    // rx.render_edge_patch();
    // rx.render_face_patch();
    polyscope::show();
#endif

    histo.free();

    report.write(Arg.output_folder + "/rxmesh_sec",
                 "SEC_RXMesh_" + extract_file_name(Arg.obj_file_name));
}