#pragma once

#include "geodesic_kernel.cuh"
#include "geodesic_ptp_rxmesh.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

#if CUDART_VERSION >= 12030
inline void add_geodesic_kernel_node(cudaGraph_t              graph,
                                     cudaGraphNode_t*         node,
                                     const cudaGraphNode_t*   deps,
                                     const size_t             num_deps,
                                     void*                    func,
                                     const dim3               grid_dim,
                                     const dim3               block_dim,
                                     const unsigned int       smem_bytes,
                                     void**                   kernel_args)
{
    using rxmesh::HandleError;

    cudaGraphNodeParams params = {};
    params.type               = cudaGraphNodeTypeKernel;
    params.kernel.func        = func;
    params.kernel.gridDim     = grid_dim;
    params.kernel.blockDim    = block_dim;
    params.kernel.sharedMemBytes = smem_bytes;
    params.kernel.kernelParams   = kernel_args;
    params.kernel.extra          = nullptr;

    CUDA_ERROR(cudaGraphAddNode(node, graph, deps, num_deps, &params));
}
#endif


template <typename T>
inline void geodesic_rxmesh_graph(rxmesh::RXMeshStatic&           rx,
                                  const rxmesh::DenseMatrix<int>& h_seeds,
                                  const rxmesh::DenseMatrix<int>& h_limits,
                                  int                             h_limits_size,
                                  rxmesh::VertexAttribute<int>&   d_toplesets)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

#if CUDART_VERSION < 12030
    geodesic_rxmesh<T>(rx, h_seeds, h_limits, h_limits_size, d_toplesets);
    return;
#else
    // Report
    Report report("Geodesic_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", std::string("RXMesh_CUDAGraph"));

    // input coords
    auto input_coord = rx.get_input_vertex_coordinates();

    // RXMesh launch box
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box,
                          (void*)relax_ptp_rxmesh_graph<T, blockThreads>,
                          true);

    // Geodesic distance attribute for all vertices (seeds set to zero
    // and infinity otherwise)
    auto rxmesh_geo = rx.add_vertex_attribute<T>("geo", 1u);
    rxmesh_geo->reset(std::numeric_limits<T>::infinity(), rxmesh::HOST);
    rx.for_each_vertex(rxmesh::HOST, [&](const VertexHandle vh) {
        int v_id = rx.map_to_global(vh);
        for (int k = 0; k < h_seeds.rows(); ++k) {
            if (h_seeds(k, 0) == v_id) {
                (*rxmesh_geo)(vh) = 0;
                break;
            }
        }
    });
    rxmesh_geo->move(rxmesh::HOST, rxmesh::DEVICE);

    // second buffer for geodesic distance for double buffering
    auto rxmesh_geo_2 = rx.add_vertex_attribute<T>("geo2", 1u, rxmesh::DEVICE);
    rxmesh_geo_2->copy_from(*rxmesh_geo, rxmesh::DEVICE, rxmesh::DEVICE);

    VertexAttribute<T>* double_buffer[2] = {rxmesh_geo.get(),
                                            rxmesh_geo_2.get()};

    GeodesicState* d_state(nullptr);
    int*           d_limits(nullptr);
    CUDA_ERROR(cudaMalloc((void**)&d_state, sizeof(GeodesicState)));
    CUDA_ERROR(cudaMalloc((void**)&d_limits, h_limits_size * sizeof(int)));
    CUDA_ERROR(cudaMemcpy(d_limits,
                          h_limits.data(rxmesh::HOST),
                          h_limits_size * sizeof(int),
                          cudaMemcpyHostToDevice));

    cudaGraph_t                  graph(nullptr);
    cudaGraph_t                  body_graph(nullptr);
    cudaGraphExec_t              graph_exec(nullptr);
    cudaGraphNode_t              init_node(nullptr);
    cudaGraphNode_t              while_node(nullptr);
    cudaGraphNode_t              relax_node(nullptr);
    cudaGraphNode_t              advance_node(nullptr);
    cudaGraphConditionalHandle   cond_handle = 0;

    CUDA_ERROR(cudaGraphCreate(&graph, 0));
    CUDA_ERROR(cudaGraphConditionalHandleCreate(
        &cond_handle, graph, 0, cudaGraphCondAssignDefault));

    void* init_args[] = {&d_state, &h_limits_size, &cond_handle};
    add_geodesic_kernel_node(graph,
                             &init_node,
                             nullptr,
                             0,
                             (void*)init_geodesic_graph_state,
                             dim3(1),
                             dim3(1),
                             0,
                             init_args);

    cudaGraphNodeParams while_params = {};
    while_params.type                = cudaGraphNodeTypeConditional;
    while_params.conditional.handle  = cond_handle;
    while_params.conditional.type    = cudaGraphCondTypeWhile;
    while_params.conditional.size    = 1;
    CUDA_ERROR(
        cudaGraphAddNode(&while_node, graph, &init_node, 1, &while_params));
    body_graph = while_params.conditional.phGraph_out[0];

    const T infinity_val = std::numeric_limits<T>::infinity();
    const T error_tol    = T(1e-3);

    auto context       = rx.get_context();
    auto coords        = *input_coord;
    auto geo_0         = *double_buffer[0];
    auto geo_1         = *double_buffer[1];
    auto topleset_attr = d_toplesets;
    void* relax_args[] = {&context,
                          &coords,
                          &geo_0,
                          &geo_1,
                          &topleset_attr,
                          &d_state,
                          (void*)&infinity_val,
                          (void*)&error_tol};
    add_geodesic_kernel_node(
        body_graph,
        &relax_node,
        nullptr,
        0,
        (void*)relax_ptp_rxmesh_graph<T, blockThreads>,
        dim3(launch_box.blocks),
        dim3(blockThreads),
        static_cast<unsigned int>(launch_box.smem_bytes_dyn),
        relax_args);

    void* advance_args[] = {&d_state, &d_limits, &cond_handle};
    add_geodesic_kernel_node(body_graph,
                             &advance_node,
                             &relax_node,
                             1,
                             (void*)advance_geodesic_graph_state,
                             dim3(1),
                             dim3(1),
                             0,
                             advance_args);

    CUDA_ERROR(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    GPUTimer timer;
    timer.start();
    CUDA_ERROR(cudaGraphLaunch(graph_exec, 0));
    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaProfilerStop());

    GeodesicState h_state;
    CUDA_ERROR(cudaMemcpy(&h_state,
                          d_state,
                          sizeof(GeodesicState),
                          cudaMemcpyDeviceToHost));

    rxmesh_geo->copy_from(*double_buffer[h_state.d],
                          rxmesh::DEVICE,
                          rxmesh::HOST);

    RXMESH_INFO("Geodesic_RXMesh took {} (ms) -- #iter= {}",
                timer.elapsed_millis(),
                h_state.iter);

#if USE_POLYSCOPE
    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->addVertexScalarQuantity("geodesic", *rxmesh_geo);
    polyscope::show();
#endif

    CUDA_ERROR(cudaGraphExecDestroy(graph_exec));
    CUDA_ERROR(cudaGraphDestroy(graph));
    GPU_FREE(d_state);
    GPU_FREE(d_limits);

    // Finalize report
    report.add_member("num_iter_taken", h_state.iter);
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
#endif
}
