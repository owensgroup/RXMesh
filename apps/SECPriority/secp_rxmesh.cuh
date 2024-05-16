#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>


#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"

// Priority Queue related includes
#include <cuco/priority_queue.cuh>
#include <cuco/detail/pair.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/**
 * @brief Return unique index of the local mesh element composed by the
 * patch id and the local index
 *
 * @param local_id the local within-patch mesh element id
 * @param patch_id the patch owning the mesh element
 * @return
 */
constexpr __device__ __host__ __forceinline__ uint32_t
unique_id32(const uint16_t local_id, const uint16_t patch_id)
{
    uint32_t ret = patch_id;
    ret          = (ret << 16);
    ret |= local_id;
    return ret;
}

/**
 * @brief unpack a 32 uint to its high and low 16 bits. 
 * This is used to convert the unique id to its local id (16
 * low bit) and patch id (high 16 bit)
 * @param uid unique id
 * @return a std::pair storing the patch id and local id
 */
constexpr __device__ __host__ __forceinline__ std::pair<uint16_t, uint16_t>
                                              unpack32(uint32_t uid)
{
    uint16_t local_id = uid & ((1 << 16) - 1);
    uint16_t patch_id = uid >> 16;
    return std::make_pair(patch_id, local_id);
}

// Priority queue setup. Use 'pair_less' to prioritize smaller values.
template <typename T>
struct pair_less 
{
    __host__ __device__ bool operator()(const T& a, const T& b) const
    {
        return a.first < b.first;
    }
};

using PriorityPair_t        = cuco::pair<float, uint32_t>;
using PriorityCompare       = pair_less<PriorityPair_t>;
using PriorityQueue_t       = cuco::priority_queue<PriorityPair_t, PriorityCompare>;
using PQView_t              = PriorityQueue_t::device_mutable_view;


template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;

#include "secp_kernels.cuh"

#include "rxmesh/util/report.h"

template <typename T>
void render_edge_attr(rxmesh::RXMeshDynamic& rx,
    const std::shared_ptr<rxmesh::EdgeAttribute<T>>& edge_attr)
{
    using namespace rxmesh;
    //make sure the attribute is on the HOST
    edge_attr->move(DEVICE, HOST);

    std::vector<float> edgeColors(rx.get_num_edges());
    rx.for_each_edge(HOST,
        [&](EdgeHandle eh) {
            if(true == (*edge_attr)(eh))
            {
                edgeColors[rx.linear_id(eh)] = 200.0f;
            }
            else
            {
                edgeColors[rx.linear_id(eh)] = eh.patch_id();
            }
        });

    auto ps_mesh = rx.get_polyscope_mesh();
    auto edge_colors = ps_mesh->addEdgeScalarQuantity("Edges to Collapse", edgeColors);
    edge_colors->setEnabled(true);
}

inline void secp_rxmesh(rxmesh::RXMeshDynamic& rx,
                       const uint32_t         final_num_vertices)
{
    EXPECT_TRUE(rx.validate());

    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    rxmesh::Report report("SECP_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name + "_before", rx, "model_before");
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);

    auto coords = rx.get_input_vertex_coordinates();

    LaunchBox<blockThreads> launch_box;

    float total_time   = 0;
    float app_time     = 0;
    float slice_time   = 0;
    float cleanup_time = 0;
    float pq_time      = 0;
    float pop_mark_time      = 0;
    float e_priority_time      = 0;

    auto e_pop_attr = rx.add_edge_attribute<bool>("ePop", 1);

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
    GPUTimer timer;
    timer.start();
    while(rx.get_num_vertices(true) > final_num_vertices)
    {
        ++num_passes;

        GPUTimer pq_timer;
        pq_timer.start();

        // rebuild every round? Not necessarily a great way to use a pq.
        PriorityQueue_t pq(rx.get_num_edges());
        e_pop_attr->reset(false, DEVICE);

        //rx.prepare_launch_box(
        rx.update_launch_box(
            {Op::EV},
            launch_box,
            (void*)compute_edge_priorities<float, blockThreads>,
            false, false, false, false,
            [&](uint32_t v, uint32_t e, uint32_t f){
              // Allocate enough additional memory
              // for the priority queue and the intermediate
              // array of PriorityPair_t.
              return pq.get_shmem_size(blockThreads) + (e*sizeof(PriorityPair_t));
            }
        );

        GPUTimer edge_priorities_timer;
        edge_priorities_timer.start();
        compute_edge_priorities<float, blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>( rx.get_context(), *coords, pq.get_mutable_device_view(), pq.get_shmem_size(blockThreads));
        edge_priorities_timer.stop();
        e_priority_time += edge_priorities_timer.elapsed_millis();
        //cudaDeviceSynchronize();
        //RXMESH_TRACE("launch_box.smem_bytes_dyn = {}", launch_box.smem_bytes_dyn);
        //RXMESH_TRACE("pq.get_shmem_size = {}", pq.get_shmem_size(blockThreads));

        // Next kernel needs to pop some percentage of the top
        // elements in the priority queue and store popped elements
        // to be used by the next kernel that actually does the collapses

        float reduce_ratio = 0.1f;
        const int num_edges_before = int(rx.get_num_edges());
        const int reduce_threshold =
            std::max(1, int(reduce_ratio * float(num_edges_before)));
        // Mark the edge attributes to be collapsed
        uint32_t pop_num_edges = reduce_threshold; //reduce_ratio * rx.get_num_edges();
        //RXMESH_TRACE("pop_num_edges: {}", pop_num_edges);

        constexpr uint32_t threads_per_block = 256;
        uint32_t number_of_blocks = (pop_num_edges + threads_per_block - 1) / threads_per_block;
        int shared_mem_bytes = pq.get_shmem_size(threads_per_block) +
                               (threads_per_block * sizeof(PriorityPair_t));
        //RXMESH_TRACE("threads_per_block: {}", threads_per_block);
        //RXMESH_TRACE("number_of_blocks: {}", number_of_blocks);
        //RXMESH_TRACE("shared_mem_bytes: {}", shared_mem_bytes);

        GPUTimer pop_mark_timer;
        pop_mark_timer.start();
        pop_and_mark_edges_to_collapse<threads_per_block>
            <<<number_of_blocks, threads_per_block, shared_mem_bytes>>>
                (pq.get_mutable_device_view(),
                 *e_pop_attr,
                 pop_num_edges);

       // if(num_passes == 1)
       // {
       //     render_edge_attr<bool>(rx, e_pop_attr);
       // }
        CUDA_ERROR(cudaDeviceSynchronize());
        CUDA_ERROR(cudaGetLastError());
        pop_mark_timer.stop();
        pop_mark_time += pop_mark_timer.elapsed_millis();

        pq_timer.stop();

        pq_time += pq_timer.elapsed_millis();

        // loop over the mesh, and try to collapse

        rx.reset_scheduler();
        while(!rx.is_queue_empty() &&
              rx.get_num_vertices(true) > final_num_vertices)
        {

            //RXMESH_INFO(" Queue size = {}",
            //            rx.get_context().m_patch_scheduler.size());

            //rx.prepare_launch_box(
            rx.update_launch_box(
                {Op::EV},
                launch_box,
                (void*)secp<float, blockThreads>,
                true, false, false, false,
                [&](uint32_t v, uint32_t e, uint32_t f) {
                    return detail::mask_num_bytes(e) +
                           2 * detail::mask_num_bytes(v) +
                           3 * ShmemAllocator::default_alignment;
                }
            );

            max_smem_bytes_dyn =
                std::max(max_smem_bytes_dyn, launch_box.smem_bytes_dyn);
            max_smem_bytes_static =
                std::max(max_smem_bytes_static, launch_box.smem_bytes_static);
            max_num_registers_per_thread =
                std::max(max_num_registers_per_thread,
                         launch_box.num_registers_per_thread);
            max_num_blocks =
                std::max(max_num_blocks, DIVIDE_UP(launch_box.blocks, 8));
            GPUTimer app_timer;

            app_timer.start();
            secp<float, blockThreads>
                <<<DIVIDE_UP(launch_box.blocks, 8),
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *coords,
                                                reduce_threshold,
                                                *e_pop_attr);
            // should we cudaDeviceSyn here? stopping timers too soon?
            //CUDA_ERROR(cudaDeviceSynchronize());
            //CUDA_ERROR(cudaGetLastError());
            
            app_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords);
            slice_timer.stop();

            GPUTimer cleanup_timer2;
            cleanup_timer2.start();
            rx.cleanup();
            cleanup_timer2.stop();


            CUDA_ERROR(cudaDeviceSynchronize());
            CUDA_ERROR(cudaGetLastError());

            app_time += app_timer.elapsed_millis();
            slice_time += slice_timer.elapsed_millis();
            cleanup_time += cleanup_timer.elapsed_millis();
            cleanup_time += cleanup_timer2.elapsed_millis();
        }
    }
    timer.stop();
    total_time += timer.elapsed_millis();
    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("secp_rxmesh() RXMesh SEC took {} (ms), num_passes= {}",
                total_time,
                num_passes);
    RXMESH_INFO("secp_rxmesh() PriorityQ time {} (ms)", pq_time);
    RXMESH_INFO("secp_rxmesh() |-Edge priorities time {} (ms)", e_priority_time);
    RXMESH_INFO("secp_rxmesh() |-Pop and Mark time {} (ms)", pop_mark_time);
    RXMESH_INFO("secp_rxmesh() App time {} (ms)", app_time);
    RXMESH_INFO("secp_rxmesh() Slice timer {} (ms)", slice_time);
    RXMESH_INFO("secp_rxmesh() Cleanup timer {} (ms)", cleanup_time);

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
    report.add_member("secp_remesh_time", total_time);
    report.add_member("priority_queue_time", pq_time);
    report.add_member("app_time", app_time);
    report.add_member("slice_time", slice_time);
    report.add_member("cleanup_time", cleanup_time);
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