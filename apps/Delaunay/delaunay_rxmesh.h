#include "delaunay_edge_flip_kernel.cuh"
#include "rxmesh/rxmesh_dynamic.h"

inline bool delaunay_rxmesh(rxmesh::RXMeshDynamic& rx)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          {rxmesh::DynOp::EdgeFlip},
                          launch_box,
                          (void*)delaunay_edge_flip<float, blockThreads>,
                          true);

    auto coords = rx.get_input_vertex_coordinates();

    GPUTimer timer;
    timer.start();

    delaunay_edge_flip<float, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords);

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    RXMESH_TRACE("delaunay_edge_flip() RXMesh Delaunay Edge Flip took {} (ms)",
                 timer.elapsed_millis());

    return true;
}