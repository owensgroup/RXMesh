#include <cuda_profiler_api.h>

#include "rxmesh/rxmesh_dynamic.h"

#include "remesh_kernels.cuh"

#include "rxmesh/util/util.h"

uint32_t iter = 0;

inline void stats(rxmesh::RXMeshDynamic& rx)
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


template <typename T>
inline void split_long_edges(rxmesh::RXMeshDynamic&         rx,
                             rxmesh::VertexAttribute<T>*    coords,
                             rxmesh::EdgeAttribute<int8_t>* updated,
                             const T                        high_edge_len_sq)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;


    updated->reset(0, DEVICE);

    rx.reset_scheduler();
    while (!rx.is_queue_empty()) {

        LaunchBox<blockThreads> launch_box;
        rx.update_launch_box({Op::EV},
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

        edge_split<T, blockThreads><<<DIVIDE_UP(launch_box.blocks, 2),
                                      launch_box.num_threads,
                                      launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *updated, high_edge_len_sq);

        CUDA_ERROR(cudaDeviceSynchronize());

        rx.cleanup();
        rx.slice_patches(*coords, *updated);
        rx.cleanup();


        /*RXMESH_TRACE(" ");
        rx.update_host();
        if (!rx.validate()) {
            polyscope::show();
        }
        RXMESH_INFO(" ");
        RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
        RXMESH_INFO("#Edges {}", rx.get_num_edges());
        RXMESH_INFO("#Faces {}", rx.get_num_faces());
        RXMESH_INFO("#Patches {}", rx.get_num_patches());
        // stats(rx);
        bool show = false;
        if (show) {
            coords->move(DEVICE, HOST);
            updated->move(DEVICE, HOST);
            rx.update_polyscope();
            auto ps_mesh = rx.get_polyscope_mesh();
            ps_mesh->updateVertexPositions(*coords);
            ps_mesh->setEnabled(false);

            ps_mesh->addEdgeScalarQuantity("updated", *updated);

            // rx.render_vertex_patch();
            // rx.render_edge_patch();
            rx.render_face_patch()->setEnabled(false);

            // uint32_t cc = 0;
            // rx.render_patch(cc)->setEnabled(false);

            polyscope::show();
        }*/
    }
}

template <typename T>
inline void collapse_short_edges(rxmesh::RXMeshDynamic&         rx,
                                 rxmesh::VertexAttribute<T>*    coords,
                                 rxmesh::EdgeAttribute<int8_t>* updated,
                                 const T                        low_edge_len_sq,
                                 const T high_edge_len_sq)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;

    updated->reset(0, DEVICE);

    rx.reset_scheduler();
    while (!rx.is_queue_empty()) {
        LaunchBox<blockThreads> launch_box;
        rx.update_launch_box({Op::EV},
                             launch_box,
                             (void*)edge_collapse<T, blockThreads>,
                             true,
                             false,
                             true,
                             true,
                             [=](uint32_t v, uint32_t e, uint32_t f) {
                                 return detail::mask_num_bytes(v) +
                                        detail::mask_num_bytes(e) +
                                        e * sizeof(uint8_t) +
                                        3 * ShmemAllocator::default_alignment;
                             });

        edge_collapse<T, blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                            *coords,
                                            *updated,
                                            low_edge_len_sq,
                                            high_edge_len_sq);

        CUDA_ERROR(cudaDeviceSynchronize());

        rx.cleanup();
        rx.slice_patches(*coords, *updated);
        rx.cleanup();

        // stats(rx);
        /*rx.update_host();
        if (!rx.validate()) {
            polyscope::show();
        }
        RXMESH_INFO(" ");
        RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
        RXMESH_INFO("#Edges {}", rx.get_num_edges());
        RXMESH_INFO("#Faces {}", rx.get_num_faces());
        RXMESH_INFO("#Patches {}", rx.get_num_patches());

        bool show = false;
        if (show) {
            // polyscope::removeAllStructures();

            coords->move(DEVICE, HOST);
            rx.export_obj("collaspse_" + std::to_string(++iter) + ".obj",
                          *coords);
            rx.update_polyscope();
            auto ps_mesh = rx.get_polyscope_mesh();
            ps_mesh->updateVertexPositions(*coords);
            ps_mesh->setEnabled(false);

            // updated->move(DEVICE, HOST);
            // ps_mesh->addEdgeScalarQuantity("updated", *updated);

            rx.render_vertex_patch();
            rx.render_edge_patch();
            rx.render_face_patch();

            int p_red = 0;
            rx.render_patch(p_red);

            polyscope::show();
        }*/
    }
}

template <typename T>
inline void equalize_valences(rxmesh::RXMeshDynamic&         rx,
                              rxmesh::VertexAttribute<T>*    coords,
                              rxmesh::EdgeAttribute<int8_t>* updated)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;

    updated->reset(0, DEVICE);

    rx.reset_scheduler();
    while (!rx.is_queue_empty()) {
        LaunchBox<blockThreads> launch_box;
        rx.update_launch_box({Op::EVDiamond},
                             launch_box,
                             (void*)edge_flip<T, blockThreads>,
                             true,
                             false,
                             true);

        edge_flip<T, blockThreads><<<launch_box.blocks,
                                     launch_box.num_threads,
                                     launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *updated);

        rx.cleanup();
        rx.slice_patches(*coords, *updated);
        rx.cleanup();
        // stats(rx);
    }
}

template <typename T>
inline void tangential_relaxation(rxmesh::RXMeshDynamic&      rx,
                                  rxmesh::VertexAttribute<T>* coords,
                                  rxmesh::VertexAttribute<T>* new_coords)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 384;

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::VV},
                          launch_box,
                          (void*)vertex_smoothing<T, blockThreads>,
                          false,
                          true);

    vertex_smoothing<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, *new_coords);
}

inline void remesh_rxmesh(rxmesh::RXMeshDynamic& rx)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

#if USE_POLYSCOPE
    //rx.render_vertex_patch();
    //rx.render_edge_patch();
    //rx.render_face_patch();
    // polyscope::show();
#endif

    auto coords     = rx.get_input_vertex_coordinates();
    auto new_coords = rx.add_vertex_attribute<float>("newCoords", 3);
    new_coords->reset(LOCATION_ALL, 0);
    auto updated = rx.add_edge_attribute<int8_t>("Updated", 1);

    // compute average edge length
    float* average_edge_len;
    CUDA_ERROR(cudaMallocManaged((void**)&average_edge_len, sizeof(float)));

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box(
        {Op::EV},
        launch_box,
        (void*)compute_average_edge_length<float, blockThreads>,
        false);
    compute_average_edge_length<float, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, average_edge_len);
    CUDA_ERROR(cudaDeviceSynchronize());
    average_edge_len[0] /= rx.get_num_edges();

    // 4.0/5.0 * targe_edge_len
    const float low_edge_len =
        (4.f / 5.f) * Arg.relative_len * average_edge_len[0];
    const float low_edge_len_sq = low_edge_len * low_edge_len;

    // 4.0/3.0 * targe_edge_len
    const float high_edge_len =
        (4.f / 3.f) * Arg.relative_len * average_edge_len[0];
    const float high_edge_len_sq = high_edge_len * high_edge_len;

    // stats(rx);

    GPUTimer timer;
    timer.start();

    for (uint32_t iter = 0; iter < Arg.num_iter; ++iter) {
        split_long_edges(rx, coords.get(), updated.get(), high_edge_len_sq);
        collapse_short_edges(
            rx, coords.get(), updated.get(), low_edge_len_sq, high_edge_len_sq);
        equalize_valences(rx, coords.get(), updated.get());
        tangential_relaxation(rx, coords.get(), new_coords.get());
        std::swap(new_coords, coords);
    }

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());

    rx.update_host();
    coords->move(DEVICE, HOST);
    new_coords->move(DEVICE, HOST);

    EXPECT_TRUE(rx.validate());

    rx.export_obj("remesh.obj", *coords);
    RXMESH_INFO("remesh_rxmesh() took {} (ms)", timer.elapsed_millis());
    RXMESH_INFO("Output mesh #Vertices {}", rx.get_num_vertices());
    RXMESH_INFO("Output mesh #Edges {}", rx.get_num_edges());
    RXMESH_INFO("Output mesh #Faces {}", rx.get_num_faces());
    RXMESH_INFO("Output mesh #Patches {}", rx.get_num_patches());


#if USE_POLYSCOPE
    rx.update_polyscope();

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->setEnabled(false);

    // rx.render_patch(0);
    // rx.render_patch(1);
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    polyscope::show();
#endif


    CUDA_ERROR(cudaFree(average_edge_len));
}