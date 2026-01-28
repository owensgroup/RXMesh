#pragma once
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/report.h"

#include "secp_kernels.cuh"
#include "secp_pair.h"

inline void secp_rxmesh(rxmesh::RXMeshDynamic& rx,
                        const uint32_t         final_num_vertices,
                        const float            edge_reduce_ratio)
{
    if (!rx.validate()) {
        RXMESH_ERROR("Mesh validation failed");
        return;
    }

    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    Report report("SECP_RXMesh");
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
    timers.add("PriorityQueue");
    timers.add("PriorityQueuePop");
    timers.add("EdgePriority");

    auto to_collapse = rx.add_edge_attribute<bool>("ePop", 1);

    RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
    RXMESH_INFO("#Edges {}", rx.get_num_edges());
    RXMESH_INFO("#Faces {}", rx.get_num_faces());
    RXMESH_INFO("#Patches {}", rx.get_num_patches());

    size_t   max_smem_bytes_dyn           = 0;
    size_t   max_smem_bytes_static        = 0;
    uint32_t max_num_registers_per_thread = 0;
    uint32_t max_num_blocks               = 0;

#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    // polyscope::show();
#endif

    bool validate = false;

    int num_passes = 0;

    CUDA_ERROR(cudaProfilerStart());
    // PriorityQueueT priority_queue(rx.get_num_edges());

    timers.start("Total");
    while (rx.get_num_vertices(true) > final_num_vertices) {
        ++num_passes;

        timers.start("PriorityQueue");

        // rebuild every round? Not necessarily a great way to use a priority
        // queue.
        PriorityQueueT priority_queue(rx.get_num_edges());
        // priority_queue.clear();
        to_collapse->reset(false, DEVICE);
        rx.update_launch_box(
            {Op::EV},
            launch_box,
            (void*)compute_edge_priorities<float, blockThreads>,
            false,
            false,
            false,
            false,
            [&](uint32_t v, uint32_t e, uint32_t f) {
                // Allocate enough additional memory
                // for the priority queue and the intermediate
                // array of PriorityPairT.
                return priority_queue.get_shmem_size(blockThreads) +
                       (e * sizeof(PriorityPairT));
            });

        timers.start("EdgePriority");
        compute_edge_priorities<float, blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(
                rx.get_context(),
                *coords,
                priority_queue.get_mutable_device_view(),
                priority_queue.get_shmem_size(blockThreads));
        timers.stop("EdgePriority");

        // Next kernel needs to pop some percentage of the top
        // elements in the priority queue and store popped elements
        // to be used by the next kernel that actually does the collapses
        const int num_edges_before = int(rx.get_num_edges(true));
        const int reduce_threshold =
            std::max(1, int(edge_reduce_ratio * float(num_edges_before)));
        // Mark the edge attributes to be collapsed
        uint32_t pop_num_edges = reduce_threshold;

        constexpr uint32_t threads_per_block = 256;
        uint32_t           number_of_blocks =
            (pop_num_edges + threads_per_block - 1) / threads_per_block;
        int shared_mem_bytes =
            priority_queue.get_shmem_size(threads_per_block) +
            (threads_per_block * sizeof(PriorityPairT));

        timers.start("PriorityQueuePop");
        pop_and_mark_edges_to_collapse<threads_per_block>
            <<<number_of_blocks, threads_per_block, shared_mem_bytes>>>(
                priority_queue.get_mutable_device_view(),
                *to_collapse,
                pop_num_edges);

        timers.stop("PriorityQueuePop");

        timers.stop("PriorityQueue");

        //{
        //    to_collapse->move(DEVICE, HOST);
        //
        //    rx.update_polyscope();
        //    auto ps_mesh = rx.get_polyscope_mesh();
        //    ps_mesh->updateVertexPositions(*coords);
        //
        //    rx.get_polyscope_mesh()->addEdgeScalarQuantity("ToCollapse",
        //                                                   *to_collapse);
        //    polyscope::show();
        //}


        // loop over the mesh, and try to collapse
        rx.reset_scheduler();
        while (!rx.is_queue_empty() &&
               rx.get_num_vertices(true) > final_num_vertices) {

            // RXMESH_INFO(" Queue size = {}",
            //             rx.get_context().m_patch_scheduler.size());

            // rx.prepare_launch_box(
            rx.update_launch_box(
                {Op::EV},
                launch_box,
                (void*)secp<float, blockThreads>,
                true,
                false,
                false,
                false,
                [&](uint32_t v, uint32_t e, uint32_t f) {
                    return detail::mask_num_bytes(e) +
                           2 * detail::mask_num_bytes(v) +
                           3 * ShmemAllocator::default_alignment;
                });

            timers.start("App");
            secp<float, blockThreads><<<launch_box.blocks,
                                        launch_box.num_threads,
                                        launch_box.smem_bytes_dyn>>>(
                rx.get_context(), *coords, *to_collapse);
            timers.stop("App");

            timers.start("Cleanup");
            rx.cleanup();
            timers.stop("Cleanup");

            timers.start("Slice");
            rx.slice_patches(*coords);
            timers.stop("Slice");

            timers.start("Cleanup");
            rx.cleanup();
            timers.stop("Cleanup");

            {
                // rx.update_host();
                // coords->move(DEVICE, HOST);
                // EXPECT_TRUE(rx.validate());

                // rx.update_polyscope();
                //
                // auto ps_mesh = rx.get_polyscope_mesh();
                // ps_mesh->updateVertexPositions(*coords);
                // ps_mesh->setEnabled(false);
                //
                // rx.render_vertex_patch();
                // rx.render_edge_patch();
                // rx.render_face_patch();
                // polyscope::show();
            }
        }
    }
    timers.stop("Total");
    CUDA_ERROR(cudaDeviceSynchronize());

    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("secp_rxmesh() RXMesh SEC took {} (ms), num_passes= {}",
                timers.elapsed_millis("Total"),
                num_passes);
    RXMESH_INFO("secp_rxmesh() PriorityQ time {} (ms)",
                timers.elapsed_millis("PriorityQueue"));
    RXMESH_INFO("secp_rxmesh() |-Edge priorities time {} (ms)",
                timers.elapsed_millis("EdgePriority"));
    RXMESH_INFO("secp_rxmesh() |-Pop and Mark time {} (ms)",
                timers.elapsed_millis("PriorityQueuePop"));
    RXMESH_INFO("secp_rxmesh() App time {} (ms)", timers.elapsed_millis("App"));
    RXMESH_INFO("secp_rxmesh() Slice timer {} (ms)",
                timers.elapsed_millis("Slice"));
    RXMESH_INFO("secp_rxmesh() Cleanup timer {} (ms)",
                timers.elapsed_millis("Cleanup"));

    RXMESH_INFO("#Vertices {}", rx.get_num_vertices(true));
    RXMESH_INFO("#Edges {}", rx.get_num_edges(true));
    RXMESH_INFO("#Faces {}", rx.get_num_faces(true));
    RXMESH_INFO("#Patches {}", rx.get_num_patches(true));


    rx.update_host();
    coords->move(DEVICE, HOST);
    if (!rx.validate()) {
        RXMESH_ERROR("Mesh validation failed after SECPriority remeshing");
    }

    report.add_member("num_passes", num_passes);
    report.add_member("max_smem_bytes_dyn", max_smem_bytes_dyn);
    report.add_member("max_smem_bytes_static", max_smem_bytes_static);
    report.add_member("max_num_registers_per_thread",
                      max_num_registers_per_thread);
    report.add_member("max_num_blocks", max_num_blocks);
    report.add_member("secp_remesh_time", timers.elapsed_millis("Total"));
    report.add_member("priority_queue_time",
                      timers.elapsed_millis("PriorityQueue"));
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

    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    polyscope::show();
#endif

    report.write(Arg.output_folder + "/rxmesh_secp",
                 "SECP_RXMesh_" + extract_file_name(Arg.obj_file_name));
}