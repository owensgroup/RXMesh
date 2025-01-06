#pragma once
#include <cuda_profiler_api.h>

#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/util.h"

#include "smoothing.cuh"

#include "collapse.cuh"
#include "flip.cuh"

#include "split.cuh"

#include "util.cuh"

int ps_iddd = 0;

struct Stats
{
    float avg_edge_len, max_edge_len, min_edge_len, avg_vertex_valence;
    int   max_vertex_valence, min_vertex_valence;
};

inline void patch_stats(rxmesh::RXMeshDynamic& rx)
{
    rx.update_host();
    rx.validate();

    std::vector<double> loads_v;
    std::vector<double> loads_e;
    std::vector<double> loads_f;

    uint32_t max_v_p, max_e_p, max_f_p;

    double avg_v(0.0), max_v(std::numeric_limits<double>::min()),
        min_v(std::numeric_limits<double>::max()), avg_e(0.0),
        max_e(std::numeric_limits<double>::min()),
        min_e(std::numeric_limits<double>::max()), avg_f(0.0),
        max_f(std::numeric_limits<double>::min()),
        min_f(std::numeric_limits<double>::max());

    for (uint32_t p = 0; p < rx.get_num_patches(); ++p) {
        // RXMESH_INFO(
        //    "P {} #V {}, #VOwned {}, Ratio {}, #E {}, #EOwned {}, Ratio {}, #F
        //    "
        //    "{}, #FOwned {}, Ratio {}",
        //    p,
        //    rx.get_num_vertices(p),
        //    rx.get_num_owned_vertices(p),
        //    double(rx.get_num_owned_vertices(p)) /
        //        double(rx.get_num_vertices(p)),
        //    rx.get_num_edges(p),
        //    rx.get_num_owned_edges(p),
        //    double(rx.get_num_owned_edges(p)) / double(rx.get_num_edges(p)),
        //    rx.get_num_faces(p),
        //    rx.get_num_owned_faces(p),
        //    double(rx.get_num_owned_faces(p)) / double(rx.get_num_faces(p)));
        //
        const auto patch = rx.get_patch(p);
        loads_v.push_back(
            patch.get_lp<rxmesh::VertexHandle>().compute_load_factor());
        loads_e.push_back(
            patch.get_lp<rxmesh::EdgeHandle>().compute_load_factor());
        loads_f.push_back(
            patch.get_lp<rxmesh::FaceHandle>().compute_load_factor());

        avg_v += loads_v.back();
        min_v = std::min(min_v, loads_v.back());
        if (loads_v.back() > max_v) {
            max_v   = loads_v.back();
            max_v_p = p;
        }

        avg_e += loads_e.back();
        min_e = std::min(min_e, loads_e.back());
        if (loads_e.back() > max_e) {
            max_e   = loads_e.back();
            max_e_p = p;
        }


        avg_f += loads_f.back();
        min_f = std::min(min_f, loads_f.back());
        if (loads_f.back() > max_f) {
            max_f   = loads_f.back();
            max_f_p = p;
        }
    }

    avg_v /= double(rx.get_num_patches());
    avg_e /= double(rx.get_num_patches());
    avg_f /= double(rx.get_num_patches());

    RXMESH_INFO(
        " Load factors:\n"
        "  avg_v= {}, max_v= {}, {}, min_v= {}\n"
        "  avg_e= {}, max_e= {}, {}, min_e= {}\n"
        "  avg_f= {}, max_f= {}, {}, max_f= {}\n",
        avg_v,
        max_v,
        max_v_p,
        min_v,
        avg_e,
        max_e,
        max_e_p,
        min_e,
        avg_f,
        max_f,
        max_f_p,
        min_f);
}

inline void compute_stats(rxmesh::RXMeshDynamic&                rx,
                          const rxmesh::VertexAttribute<float>* coords,
                          rxmesh::EdgeAttribute<float>*         edge_len,
                          rxmesh::VertexAttribute<int>*         vertex_valence,
                          Stats&                                stats)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    edge_len->reset(DEVICE, 0);
    vertex_valence->reset(DEVICE, 0);

    stats.avg_edge_len       = 0;
    stats.max_edge_len       = std::numeric_limits<float>::min();
    stats.min_edge_len       = std::numeric_limits<float>::max();
    stats.avg_vertex_valence = 0;
    stats.max_vertex_valence = std::numeric_limits<int>::min();
    stats.min_vertex_valence = std::numeric_limits<int>::max();

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box({Op::EV},
                         launch_box,
                         (void*)stats_kernel<float, blockThreads>,
                         false,
                         false,
                         true);

    stats_kernel<float, blockThreads><<<launch_box.blocks,
                                        launch_box.num_threads,
                                        launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *coords, *edge_len, *vertex_valence);
    CUDA_ERROR(cudaDeviceSynchronize());

    // valence
    vertex_valence->move(DEVICE, HOST);
    rx.for_each_vertex(
        HOST,
        [&](const VertexHandle vh) {
            int val = (*vertex_valence)(vh);
            stats.avg_vertex_valence += val;
            stats.max_vertex_valence = std::max(stats.max_vertex_valence, val);
            stats.min_vertex_valence = std::min(stats.min_vertex_valence, val);
        },
        NULL,
        false);
    stats.avg_vertex_valence /= rx.get_num_vertices();

    // edge len
    edge_len->move(DEVICE, HOST);
    rx.for_each_edge(
        HOST,
        [&](const EdgeHandle eh) {
            float len = (*edge_len)(eh);
            stats.avg_edge_len += len;
            stats.max_edge_len = std::max(stats.max_edge_len, len);
            if (len > std::numeric_limits<float>::epsilon()) {
                stats.min_edge_len = std::min(stats.min_edge_len, len);
            }
        },
        NULL,
        false);
    stats.avg_edge_len /= rx.get_num_edges();
}

template <typename T>
void screen_shot(rxmesh::RXMeshDynamic&      rx,
                 rxmesh::VertexAttribute<T>* coords,
                 std::string                 app)
{
#if USE_POLYSCOPE
    using namespace rxmesh;

    rx.update_host();
    coords->move(DEVICE, HOST);
    rx.update_polyscope();

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->setEdgeWidth(1.0);
    ps_mesh->setEnabled(true);
    rx.render_face_patch()->setEnabled(true);

    polyscope::screenshot(app + "_" + std::to_string(ps_iddd) + ".png");
    ps_iddd++;

    // polyscope::show();

    ps_mesh->setEnabled(false);
#endif
}


inline void remesh_rxmesh(rxmesh::RXMeshDynamic& rx)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    rxmesh::Report report("Remesh_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name + "_before", rx, "model_before");
    report.add_member("method", std::string("RXMesh"));


    auto coords     = rx.get_input_vertex_coordinates();
    auto new_coords = rx.add_vertex_attribute<float>("newCoords", 3);
    new_coords->reset(LOCATION_ALL, 0);
    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);

    auto v_valence = rx.add_vertex_attribute<uint8_t>("Valence", 1);

    auto v_boundary = rx.add_vertex_attribute<bool>("BoundaryV", 1);

    auto edge_len       = rx.add_edge_attribute<float>("edgeLen", 1);
    auto vertex_valence = rx.add_vertex_attribute<int>("vertexValence", 1);

    RXMESH_INFO("Input mesh #Vertices {}", rx.get_num_vertices());
    RXMESH_INFO("Input mesh #Edges {}", rx.get_num_edges());
    RXMESH_INFO("Input mesh #Faces {}", rx.get_num_faces());
    RXMESH_INFO("Input mesh #Patches {}", rx.get_num_patches());

    int* d_buffer;
    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));

    auto edge_link = rx.add_edge_attribute<int8_t>("edgeLink", 1);

    rx.get_boundary_vertices(*v_boundary);

    // edge_link->move(DEVICE, HOST);
    // rx.get_polyscope_mesh()->addEdgeScalarQuantity("edgeLink", *edge_link);

    // compute stats

    Timers<GPUTimer> timers;

    timers.add("Total");

    timers.add("SplitTotal");
    timers.add("Split");
    timers.add("SplitCleanup");
    timers.add("SplitSlice");

    timers.add("CollapseTotal");
    timers.add("Collapse");
    timers.add("CollapseCleanup");
    timers.add("CollapseSlice");

    timers.add("FlipTotal");
    timers.add("Flip");
    timers.add("FlipCleanup");
    timers.add("FlipSlice");

    timers.add("SmoothTotal");


    Stats stats;
    compute_stats(
        rx, coords.get(), edge_len.get(), vertex_valence.get(), stats);

    RXMESH_INFO(
        "Input Stats: Avg Edge Length= {}, Max Edge Length= {}, Min Edge "
        "Length= {}, Avg Vertex Valence= {}, Max Vertex Valence= {}, Min "
        "Vertex Valence= {}",
        stats.avg_edge_len,
        stats.max_edge_len,
        stats.min_edge_len,
        stats.avg_vertex_valence,
        stats.max_vertex_valence,
        stats.min_vertex_valence);

    RXMESH_INFO("Target edge length = {}",
                Arg.relative_len * stats.avg_edge_len);

    // 4.0/5.0 * targe_edge_len
    const float low_edge_len =
        (4.f / 5.f) * Arg.relative_len * stats.avg_edge_len;
    const float low_edge_len_sq = low_edge_len * low_edge_len;

    // 4.0/3.0 * targe_edge_len
    const float high_edge_len =
        (4.f / 3.f) * Arg.relative_len * stats.avg_edge_len;
    const float high_edge_len_sq = high_edge_len * high_edge_len;

    // stats(rx);

    timers.start("Total");
    for (uint32_t iter = 0; iter < Arg.num_iter; ++iter) {
        RXMESH_INFO(" Edge Split -- iter {}", iter);
        split_long_edges(rx,
                         coords.get(),
                         edge_status.get(),
                         v_boundary.get(),
                         high_edge_len_sq,
                         low_edge_len_sq,
                         timers,
                         d_buffer);

        RXMESH_INFO(" Edge Collapse -- iter {}", iter);
        collapse_short_edges(rx,
                             coords.get(),
                             edge_status.get(),
                             edge_link.get(),
                             v_boundary.get(),
                             low_edge_len_sq,
                             high_edge_len_sq,
                             timers,
                             d_buffer);


        RXMESH_INFO(" Edge Flip -- iter {}", iter);
        equalize_valences(rx,
                          coords.get(),
                          v_valence.get(),
                          edge_status.get(),
                          edge_link.get(),
                          v_boundary.get(),
                          timers,
                          d_buffer);

        RXMESH_INFO(" Vertex Smoothing -- iter {}", iter);
        tangential_relaxation(rx,
                              coords.get(),
                              new_coords.get(),
                              v_boundary.get(),
                              Arg.num_smooth_iters,
                              timers);
        std::swap(new_coords, coords);
    }

    timers.stop("Total");
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());

    RXMESH_INFO("remesh_rxmesh() took {} (ms)", timers.elapsed_millis("Total"));

    rx.update_host();
    coords->move(DEVICE, HOST);
    new_coords->move(DEVICE, HOST);

    RXMESH_INFO("Output mesh #Vertices {}", rx.get_num_vertices());
    RXMESH_INFO("Output mesh #Edges {}", rx.get_num_edges());
    RXMESH_INFO("Output mesh #Faces {}", rx.get_num_faces());
    RXMESH_INFO("Output mesh #Patches {}", rx.get_num_patches());

    report.add_member("total_remesh_time", timers.elapsed_millis("Total"));
    report.model_data(Arg.obj_file_name + "_after", rx, "model_after");

    compute_stats(
        rx, coords.get(), edge_len.get(), vertex_valence.get(), stats);

    RXMESH_INFO(
        "Output Stats: Avg Edge Length= {}, Max Edge Length= {}, Min Edge "
        "Length= {}, Avg Vertex Valence= {}, Max Vertex Valence= {}, Min "
        "Vertex Valence= {}",
        stats.avg_edge_len,
        stats.max_edge_len,
        stats.min_edge_len,
        stats.avg_vertex_valence,
        stats.max_vertex_valence,
        stats.min_vertex_valence);

    report.add_member("avg_edge_len", stats.avg_edge_len);
    report.add_member("max_edge_len", stats.max_edge_len);
    report.add_member("min_edge_len", stats.min_edge_len);
    report.add_member("avg_vertex_valence", stats.avg_vertex_valence);
    report.add_member("max_vertex_valence", stats.max_vertex_valence);
    report.add_member("min_vertex_valence", stats.min_vertex_valence);

    report.add_member("attributes_memory_mg",
                      coords->get_memory_mg() + new_coords->get_memory_mg() +
                          edge_status->get_memory_mg() +
                          vertex_valence->get_memory_mg());

    RXMESH_INFO("Split Total Time {} (ms)",
                timers.elapsed_millis("SplitTotal"));
    RXMESH_INFO("Collapse Total Time {} (ms)",
                timers.elapsed_millis("CollapseTotal"));
    RXMESH_INFO("Flip Total Time {} (ms)",
                timers.elapsed_millis("FlipTotal"));
    RXMESH_INFO("Smooth Total Time {} (ms)",
                timers.elapsed_millis("SmoothTotal"));   

    report.add_member("split_time_ms", timers.elapsed_millis("SplitTotal"));
    report.add_member("collapse_time_ms",
                      timers.elapsed_millis("CollapseTotal"));
    report.add_member("flip_time_ms", timers.elapsed_millis("FlipTotal"));
    report.add_member("smoothing_time_ms",
                      timers.elapsed_millis("SmoothTotal"));

    EXPECT_TRUE(rx.validate());
    // rx.export_obj("remesh.obj", *coords);

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

    report.write(Arg.output_folder + "/rxmesh_remesh",
                 "Remesh_RXMesh_" + extract_file_name(Arg.obj_file_name));
}