#pragma once

#include <cuda_profiler_api.h>

#include "cub/device/device_scan.cuh"

#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/histogram.cuh"
#include "rxmesh/util/report.h"

#include "qslim_kernels.cuh"

template <typename T>
inline void populate_bins(const rxmesh::RXMeshDynamic&    rx,
                          const rxmesh::EdgeAttribute<T>& edge_cost,
                          rxmesh::Histogram<T>            histo)
{
    using namespace rxmesh;

    rx.for_each_edge(DEVICE, [=] __device__(EdgeHandle eh) {
        T cost = edge_cost(eh);

        histo.insert(cost);
    });
}

template <typename T>
void calc_edge_cost(const rxmesh::RXMeshDynamic&      rx,
                    const rxmesh::VertexAttribute<T>* coords,
                    rxmesh::VertexAttribute<T>*       vertex_quadrics,
                    rxmesh::EdgeAttribute<T>*         edge_cost,
                    rxmesh::EdgeAttribute<T>*         edge_col_coord,
                    rxmesh::Histogram<T>&             histo)
{
    // compute vertex quadrics, edge cost, and min.max histogram

    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box({Op::FV},
                         launch_box,
                         (void*)compute_vertex_quadric_fv<T, blockThreads>,
                         false);
    compute_vertex_quadric_fv<T, blockThreads><<<launch_box.blocks,
                                                 launch_box.num_threads,
                                                 launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *coords, *vertex_quadrics);


    rx.update_launch_box(
        {Op::EV}, launch_box, (void*)compute_edge_cost<T, blockThreads>, false);
    compute_edge_cost<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                        *coords,
                                        *vertex_quadrics,
                                        *edge_cost,
                                        *edge_col_coord,
                                        histo);
}

inline void qslim_rxmesh(rxmesh::RXMeshDynamic& rx,
                         const uint32_t         final_num_vertices)
{
    EXPECT_TRUE(rx.validate());

    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    rxmesh::Report report("QSlim_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name + "_before", rx, "model_before");
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);


    auto coords = rx.get_input_vertex_coordinates();


    Timers<GPUTimer> timers;
    timers.add("Total");
    timers.add("App");
    timers.add("Slice");
    timers.add("Cleanup");
    timers.add("Histo");

    const int num_bins = 256;

    Histogram<float> histo(num_bins);

    auto vertex_quadrics = rx.add_vertex_attribute<float>("quadrics", 16);
    vertex_quadrics->reset(0, DEVICE);

    auto edge_cost = rx.add_edge_attribute<float>("cost", 1);
    edge_cost->reset(0, DEVICE);

    auto edge_col_coord = rx.add_edge_attribute<float>("eCoord", 3);
    edge_col_coord->reset(0, DEVICE);


    bool validate = false;

    CUDA_ERROR(cudaProfilerStart());

    timers.start("Total");
    while (rx.get_num_vertices(true) > final_num_vertices) {

        timers.start("Histo");
        histo.init();

        calc_edge_cost(rx,
                       coords.get(),
                       vertex_quadrics.get(),
                       edge_cost.get(),
                       edge_col_coord.get(),
                       histo);

        // populate bins
        rx.for_each_edge(
            DEVICE,
            [edge_cost = *edge_cost, histo = histo] __device__(
                EdgeHandle eh) mutable { histo.insert(edge_cost(eh)); });

        histo.scan();

        // loop over the mesh, and try to collapse
        const int num_edges_before = int(rx.get_num_edges(true));

        const int reduce_threshold =
            std::max(1, int(Arg.reduce_ratio * float(num_edges_before)));

        timers.stop("Histo");

        rx.reset_scheduler();

        while (!rx.is_queue_empty() &&
               rx.get_num_vertices(true) > final_num_vertices) {

            LaunchBox<blockThreads> lb;
            rx.update_launch_box(
                {Op::EV},
                lb,
                (void*)simplify_ev<float, blockThreads>,
                true,
                false,
                false,
                false,
                [](uint32_t v, uint32_t e, uint32_t f) {
                    return detail::mask_num_bytes(e) +
                           2 * detail::mask_num_bytes(v) +
                           3 * ShmemAllocator::default_alignment;
                });


            timers.start("App");
            simplify_ev<float, blockThreads>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(),
                    *coords,
                    histo,
                    reduce_threshold,
                    *vertex_quadrics,
                    *edge_cost,
                    *edge_col_coord);


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

            if (validate) {
                rx.update_host();
                EXPECT_TRUE(rx.validate());
            }
        }
    }

    timers.stop("Total");
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("qslim_rxmesh() RXMesh QSlim took {} (ms)",
                timers.elapsed_millis("Total"));
    RXMESH_INFO("qslim_rxmesh() Histo time {} (ms)",
                timers.elapsed_millis("Histo"));
    RXMESH_INFO("qslim_rxmesh() App time {} (ms)",
                timers.elapsed_millis("App"));
    RXMESH_INFO("qslim_rxmesh() Slice timer {} (ms)",
                timers.elapsed_millis("Slice"));
    RXMESH_INFO("qslim_rxmesh() Cleanup timer {} (ms)",
                timers.elapsed_millis("Cleanup"));


    RXMESH_INFO("#Vertices {}", rx.get_num_vertices(true));
    RXMESH_INFO("#Edges {}", rx.get_num_edges(true));
    RXMESH_INFO("#Faces {}", rx.get_num_faces(true));
    RXMESH_INFO("#Patches {}", rx.get_num_patches(true));

    rx.update_host();

    coords->move(DEVICE, HOST);


    report.add_member("qslim_rxmesh_time", timers.elapsed_millis("Total"));
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

    report.write(Arg.output_folder + "/rxmesh_qslim",
                 "QSlim_RXMesh_" + extract_file_name(Arg.obj_file_name));
}