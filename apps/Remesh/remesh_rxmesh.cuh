#include <cuda_profiler_api.h>

#include "rxmesh/rxmesh_dynamic.h"

#include "remesh_kernels.cuh"

inline void split_long_edges(float high_edge_len_sq)
{
}

inline void collapse_short_edges(float low_edge_len_sq, float high_edge_len_sq)
{
}

inline void equalize_valences()
{
}

inline void tangential_relaxation()
{
}

inline void remesh_rxmesh(rxmesh::RXMeshDynamic& rx)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    // polyscope::show();
#endif

    auto coords = rx.get_input_vertex_coordinates();

    // compute average edge length
    float* average_edge_len;
    CUDA_ERROR(cudaMallocManaged((void**)&average_edge_len, sizeof(float)));

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box(
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

    // 4.0/3.0 * targe_edge_len
    const float high_edge_len =
        (4.f / 3.f) * Arg.relative_len * average_edge_len[0];

    bool validate = false;


    for (uint32_t iter = 0; iter < Arg.num_iter; ++iter) {
        split_long_edges(high_edge_len);
        collapse_short_edges(low_edge_len, high_edge_len);
        equalize_valences();
        tangential_relaxation();
    }

    rx.update_host();
    coords->move(DEVICE, HOST);

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

    CUDA_ERROR(cudaFree(average_edge_len));
}