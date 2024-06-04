#include <cuda_profiler_api.h>

#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/util.h"

int ps_iddd = 0;

using EdgeStatus = int8_t;
enum : EdgeStatus
{
    UNSEEN = 0,  // means we have not tested it before for e.g., split/flip/col
    OKAY   = 1,  // means we have tested it and it is okay to skip
    UPDATE = 2,  // means we should update it i.e., we have tested it before
    ADDED  = 3,  // means it has been added to during the split/flip/collapse
};


#include "remesh_kernels.cuh"

float split_time_ms, collapse_time_ms, flip_time_ms, smoothing_time_ms;

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
            stats.min_edge_len = std::min(stats.min_edge_len, len);
        },
        NULL,
        false);
    stats.avg_edge_len /= rx.get_num_edges();
}

int is_done(const rxmesh::RXMeshDynamic&             rx,
            const rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
            int*                                     d_buffer)
{
    using namespace rxmesh;

    // if there is at least one edge that is UNSEEN, then we are not done yet
    CUDA_ERROR(cudaMemset(d_buffer, 0, sizeof(int)));

    rx.for_each_edge(
        DEVICE,
        [edge_status = *edge_status, d_buffer] __device__(const EdgeHandle eh) {
            if (edge_status(eh) == UNSEEN || edge_status(eh) == UPDATE) {
                ::atomicAdd(d_buffer, 1);
            }
        });

    CUDA_ERROR(cudaDeviceSynchronize());
    return d_buffer[0];
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

template <typename T>
inline void split_long_edges(rxmesh::RXMeshDynamic&             rx,
                             rxmesh::VertexAttribute<T>*        coords,
                             rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                             const T high_edge_len_sq,
                             int*    d_buffer)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 512;


    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();


    float app_time     = 0;
    float slice_time   = 0;
    float cleanup_time = 0;
    // int   num_splits     = 0;
    int num_outer_iter = 0;
    int num_inner_iter = 0;

    GPUTimer timer;
    timer.start();

    while (true) {
        num_outer_iter++;
        rx.reset_scheduler();

        while (!rx.is_queue_empty()) {
            num_inner_iter++;

            // RXMESH_INFO(" Queue size = {}",
            //             rx.get_context().m_patch_scheduler.size());

            LaunchBox<blockThreads> launch_box;
            rx.update_launch_box({Op::EVDiamond},
                                 launch_box,
                                 (void*)edge_split<T, blockThreads>,
                                 true,
                                 false,
                                 false,
                                 false,
                                 [&](uint32_t v, uint32_t e, uint32_t f) {
                                     return detail::mask_num_bytes(e) +
                                            ShmemAllocator::default_alignment;
                                 });

            // CUDA_ERROR(cudaMemset(d_buffer, 0, sizeof(int)));


            GPUTimer app_timer;
            app_timer.start();
            edge_split<T, blockThreads>
                <<<DIVIDE_UP(launch_box.blocks, 8),
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *coords,
                                                *edge_status,
                                                high_edge_len_sq,
                                                d_buffer);
            app_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords, *edge_status);
            slice_timer.stop();

            GPUTimer cleanup_timer2;
            cleanup_timer2.start();
            rx.cleanup();
            cleanup_timer2.stop();

            app_time += app_timer.elapsed_millis();
            slice_time += slice_timer.elapsed_millis();
            cleanup_time += cleanup_timer.elapsed_millis();
            cleanup_time += cleanup_timer2.elapsed_millis();
            CUDA_ERROR(cudaDeviceSynchronize());
            // int dd;
            // CUDA_ERROR(
            //     cudaMemcpy(&dd, d_buffer, sizeof(int),
            //     cudaMemcpyDeviceToHost));
            // num_splits += dd;

            // rx.update_host();
            // EXPECT_TRUE(rx.validate());

            // screen_shot(rx, coords, "Split");

            // stats(rx);
            // bool show = false;
            // if (show) {
            //    rx.update_host();
            //    RXMESH_INFO(" ");
            //    RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
            //    RXMESH_INFO("#Edges {}", rx.get_num_edges());
            //    RXMESH_INFO("#Faces {}", rx.get_num_faces());
            //    RXMESH_INFO("#Patches {}", rx.get_num_patches());
            //    // stats(rx);
            //    coords->move(DEVICE, HOST);
            //    edge_status->move(DEVICE, HOST);
            //    rx.update_polyscope();
            //    auto ps_mesh = rx.get_polyscope_mesh();
            //    ps_mesh->updateVertexPositions(*coords);
            //    ps_mesh->setEnabled(false);
            //
            //    ps_mesh->addEdgeScalarQuantity("EdgeStatus", *edge_status);
            //
            //    rx.render_vertex_patch();
            //    rx.render_edge_patch();
            //    rx.render_face_patch()->setEnabled(false);
            //
            //    rx.render_patch(0);
            //
            //    polyscope::show();
            //}
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
        // RXMESH_INFO("num_splits {}, time {}",
        //             num_splits,
        //             app_time + slice_time + cleanup_time);
    }
    timer.stop();

    // RXMESH_INFO("total num_splits {}", num_splits);
    RXMESH_INFO("num_outer_iter {}", num_outer_iter);
    RXMESH_INFO("num_inner_iter {}", num_inner_iter);
    RXMESH_INFO("Split total time {} (ms)", timer.elapsed_millis());
    RXMESH_INFO("Split time {} (ms)", app_time);
    RXMESH_INFO("Split slice timer {} (ms)", slice_time);
    RXMESH_INFO("Split cleanup timer {} (ms)", cleanup_time);

    split_time_ms += timer.elapsed_millis();
}

template <typename T>
inline void collapse_short_edges(rxmesh::RXMeshDynamic&             rx,
                                 rxmesh::VertexAttribute<T>*        coords,
                                 rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                                 const T low_edge_len_sq,
                                 const T high_edge_len_sq,
                                 int*    d_buffer)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 512;

    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();

    float app_time       = 0;
    float slice_time     = 0;
    float cleanup_time   = 0;
    int   num_outer_iter = 0;
    int   num_inner_iter = 0;
    // int   num_collapses  = 0;

    GPUTimer timer;
    timer.start();
    while (true) {
        num_outer_iter++;
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {
            // RXMESH_INFO(" Queue size = {}",
            //             rx.get_context().m_patch_scheduler.size());
            num_inner_iter++;

            LaunchBox<blockThreads> launch_box;
            rx.update_launch_box(
                //{Op::EVDiamond},
                {Op::EV, Op::VV},
                launch_box,
                (void*)edge_collapse_1<T, blockThreads>,
                true,
                false,
                false,
                false,
                [&](uint32_t v, uint32_t e, uint32_t f) {
                    return detail::mask_num_bytes(e) +
                           2 * v * sizeof(uint16_t) +
                           2 * ShmemAllocator::default_alignment;
                    // 2 * detail::mask_num_bytes(v) +
                    // 3 * ShmemAllocator::default_alignment;
                });

            // CUDA_ERROR(cudaMemset(d_buffer, 0, sizeof(int)));

            GPUTimer app_timer;
            app_timer.start();
            edge_collapse_1<T, blockThreads>
                <<<DIVIDE_UP(launch_box.blocks, 8),
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *coords,
                                                *edge_status,
                                                low_edge_len_sq,
                                                high_edge_len_sq,
                                                d_buffer);
            app_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords, *edge_status);
            slice_timer.stop();

            GPUTimer cleanup_timer2;
            cleanup_timer2.start();
            rx.cleanup();
            cleanup_timer2.stop();


            app_time += app_timer.elapsed_millis();
            slice_time += slice_timer.elapsed_millis();
            cleanup_time += cleanup_timer.elapsed_millis();
            cleanup_time += cleanup_timer2.elapsed_millis();

            // int dd;
            // CUDA_ERROR(
            //     cudaMemcpy(&dd, d_buffer, sizeof(int),
            //     cudaMemcpyDeviceToHost));
            // num_collapses += dd;

            // screen_shot(rx, coords, "Collapse");

            // stats(rx);
            // bool show = false;
            // if (show) {
            //    rx.update_host();
            //    RXMESH_INFO(" ");
            //    RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
            //    RXMESH_INFO("#Edges {}", rx.get_num_edges());
            //    RXMESH_INFO("#Faces {}", rx.get_num_faces());
            //    RXMESH_INFO("#Patches {}", rx.get_num_patches());
            //    // stats(rx);
            //    coords->move(DEVICE, HOST);
            //    edge_status->move(DEVICE, HOST);
            //    rx.update_polyscope();
            //    auto ps_mesh = rx.get_polyscope_mesh();
            //    ps_mesh->updateVertexPositions(*coords);
            //    ps_mesh->setEnabled(false);
            //
            //    ps_mesh->addEdgeScalarQuantity("EdgeStatus", *edge_status);
            //
            //    rx.render_vertex_patch();
            //    rx.render_edge_patch();
            //    rx.render_face_patch()->setEnabled(false);
            //
            //    polyscope::show();
            //}
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
        // RXMESH_INFO("num_collapses {}, time {}",
        //             num_collapses,
        //             app_time + slice_time + cleanup_time);
    }
    timer.stop();
    // RXMESH_INFO("total num_collapses {}", num_collapses);
    RXMESH_INFO("num_outer_iter {}", num_outer_iter);
    RXMESH_INFO("num_inner_iter {}", num_inner_iter);
    RXMESH_INFO("Collapse total time {} (ms)", timer.elapsed_millis());
    RXMESH_INFO("Collapse time {} (ms)", app_time);
    RXMESH_INFO("Collapse slice timer {} (ms)", slice_time);
    RXMESH_INFO("Collapse cleanup timer {} (ms)", cleanup_time);

    collapse_time_ms += timer.elapsed_millis();
}

template <typename T>
inline void equalize_valences(rxmesh::RXMeshDynamic&             rx,
                              rxmesh::VertexAttribute<T>*        coords,
                              rxmesh::VertexAttribute<uint8_t>*  v_valence,
                              rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                              int*                               d_buffer)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 512;

    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();

    float app_time     = 0;
    float slice_time   = 0;
    float cleanup_time = 0;

    // int   num_flips      = 0;
    int num_outer_iter = 0;
    int num_inner_iter = 0;

    GPUTimer timer;
    timer.start();
    while (true) {
        num_outer_iter++;
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {
            // RXMESH_INFO(" Queue size = {}",
            //             rx.get_context().m_patch_scheduler.size());
            num_inner_iter++;
            LaunchBox<blockThreads> launch_box;

            rx.update_launch_box({},
                                 launch_box,
                                 (void*)compute_valence<blockThreads>,
                                 false,
                                 false,
                                 true);
            GPUTimer app_timer;
            app_timer.start();
            compute_valence<blockThreads>
                <<<launch_box.blocks,
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(), *v_valence);

            rx.update_launch_box(
                //{Op::EVDiamond},
                {Op::EVDiamond, Op::VV},
                launch_box,
                (void*)edge_flip_1<T, blockThreads>,
                true,
                false,
                false,
                false,
                [&](uint32_t v, uint32_t e, uint32_t f) {
                    return detail::mask_num_bytes(e) +
                           2 * v * sizeof(uint16_t) +
                           2 * ShmemAllocator::default_alignment;
                    // 2 * detail::mask_num_bytes(v) +
                    // 3 * ShmemAllocator::default_alignment;
                });

            // CUDA_ERROR(cudaMemset(d_buffer, 0, sizeof(int)));

            edge_flip_1<T, blockThreads><<<DIVIDE_UP(launch_box.blocks, 8),
                                           launch_box.num_threads,
                                           launch_box.smem_bytes_dyn>>>(
                rx.get_context(), *coords, *v_valence, *edge_status, d_buffer);
            app_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords, *edge_status);
            slice_timer.stop();

            GPUTimer cleanup_timer2;
            cleanup_timer2.start();
            rx.cleanup();
            cleanup_timer2.stop();

            app_time += app_timer.elapsed_millis();
            slice_time += slice_timer.elapsed_millis();
            cleanup_time += cleanup_timer.elapsed_millis();
            cleanup_time += cleanup_timer2.elapsed_millis();

            // int dd;
            // CUDA_ERROR(
            //     cudaMemcpy(&dd, d_buffer, sizeof(int),
            //     cudaMemcpyDeviceToHost));
            // num_flips += dd;

            // screen_shot(rx, coords, "Flip");

            // stats(rx);
            // bool show = false;
            // if (show) {
            //    rx.update_host();
            //    RXMESH_INFO(" ");
            //    RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
            //    RXMESH_INFO("#Edges {}", rx.get_num_edges());
            //    RXMESH_INFO("#Faces {}", rx.get_num_faces());
            //    RXMESH_INFO("#Patches {}", rx.get_num_patches());
            //    // stats(rx);
            //    coords->move(DEVICE, HOST);
            //    edge_status->move(DEVICE, HOST);
            //    rx.update_polyscope();
            //    auto ps_mesh = rx.get_polyscope_mesh();
            //    ps_mesh->updateVertexPositions(*coords);
            //    ps_mesh->setEnabled(false);
            //
            //    ps_mesh->addEdgeScalarQuantity("EdgeStatus", *edge_status);
            //
            //    rx.render_vertex_patch();
            //    rx.render_edge_patch();
            //    rx.render_face_patch()->setEnabled(false);
            //
            //    polyscope::show();
            //}
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
        // RXMESH_INFO("num_flips {}, time {}",
        //             num_flips,
        //             app_time + slice_time + cleanup_time);
    }
    timer.stop();
    // RXMESH_INFO("total num_flips {}", num_flips);
    RXMESH_INFO("num_outer_iter {}", num_outer_iter);
    RXMESH_INFO("num_inner_iter {}", num_inner_iter);
    RXMESH_INFO("Flip total time {} (ms)", timer.elapsed_millis());
    RXMESH_INFO("Flip time {} (ms)", app_time);
    RXMESH_INFO("Flip slice timer {} (ms)", slice_time);
    RXMESH_INFO("Flip cleanup timer {} (ms)", cleanup_time);

    flip_time_ms += timer.elapsed_millis();
}

template <typename T>
inline void tangential_relaxation(rxmesh::RXMeshDynamic&      rx,
                                  rxmesh::VertexAttribute<T>* coords,
                                  rxmesh::VertexAttribute<T>* new_coords)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 384;

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box({Op::VV},
                         launch_box,
                         (void*)vertex_smoothing<T, blockThreads>,
                         false,
                         true);

    GPUTimer app_timer;
    app_timer.start();
    vertex_smoothing<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, *new_coords);
    app_timer.stop();
    smoothing_time_ms += app_timer.elapsed_millis();
    RXMESH_INFO("Relax time {} (ms)", app_timer.elapsed_millis());
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

#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    rx.get_polyscope_mesh();
    // polyscope::show();
#endif

    auto coords     = rx.get_input_vertex_coordinates();
    auto new_coords = rx.add_vertex_attribute<float>("newCoords", 3);
    new_coords->reset(LOCATION_ALL, 0);
    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);

    auto v_valence = rx.add_vertex_attribute<uint8_t>("Valence", 1);


    auto edge_len       = rx.add_edge_attribute<float>("edgeLen", 1);
    auto vertex_valence = rx.add_vertex_attribute<int>("vertexValence", 1);

    RXMESH_INFO("Input mesh #Vertices {}", rx.get_num_vertices());
    RXMESH_INFO("Input mesh #Edges {}", rx.get_num_edges());
    RXMESH_INFO("Input mesh #Faces {}", rx.get_num_faces());
    RXMESH_INFO("Input mesh #Patches {}", rx.get_num_patches());

    int* d_buffer;
    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));

    // compute stats
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

    split_time_ms     = 0;
    collapse_time_ms  = 0;
    flip_time_ms      = 0;
    smoothing_time_ms = 0;

    GPUTimer timer;
    timer.start();

    for (uint32_t iter = 0; iter < Arg.num_iter; ++iter) {
        RXMESH_INFO(" Edge Split -- iter {}", iter);
        split_long_edges(
            rx, coords.get(), edge_status.get(), high_edge_len_sq, d_buffer);

        RXMESH_INFO(" Edge Collapse -- iter {}", iter);
        collapse_short_edges(rx,
                             coords.get(),
                             edge_status.get(),
                             low_edge_len_sq,
                             high_edge_len_sq,
                             d_buffer);


        RXMESH_INFO(" Edge Flip -- iter {}", iter);
        equalize_valences(
            rx, coords.get(), v_valence.get(), edge_status.get(), d_buffer);

        RXMESH_INFO(" Vertex Smoothing -- iter {}", iter);
        tangential_relaxation(rx, coords.get(), new_coords.get());
        std::swap(new_coords, coords);
    }

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());

    RXMESH_INFO("remesh_rxmesh() took {} (ms)", timer.elapsed_millis());

    rx.update_host();
    coords->move(DEVICE, HOST);
    new_coords->move(DEVICE, HOST);

    RXMESH_INFO("Output mesh #Vertices {}", rx.get_num_vertices());
    RXMESH_INFO("Output mesh #Edges {}", rx.get_num_edges());
    RXMESH_INFO("Output mesh #Faces {}", rx.get_num_faces());
    RXMESH_INFO("Output mesh #Patches {}", rx.get_num_patches());

    report.add_member("total_remesh_time", timer.elapsed_millis());
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


    report.add_member("split_time_ms", split_time_ms);
    report.add_member("collapse_time_ms", collapse_time_ms);
    report.add_member("flip_time_ms", flip_time_ms);
    report.add_member("smoothing_time_ms", smoothing_time_ms);

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