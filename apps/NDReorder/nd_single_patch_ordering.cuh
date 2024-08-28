#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"


#include "nd_partition_manager.cuh"

template <uint32_t blockThreads>
__global__ static void nd_single_patch_main(
    rxmesh::Context                   context,
    rxmesh::VertexAttribute<uint16_t> v_ordering,
    rxmesh::VertexAttribute<uint16_t> attr_matched_v,
    rxmesh::EdgeAttribute<uint16_t>   attr_active_e,
    uint16_t                          req_levels)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    // Init the struct and alloc the shared memory
    PartitionManager<blockThreads> partition_manager(
        block, context, shrd_alloc, req_levels);


    // Start the matching process and the result are saved in bit masks
    partition_manager.local_matching(block, attr_matched_v, attr_active_e, 0);
    partition_manager.local_coarsening(block, 0);

    partition_manager.local_matching(block, attr_matched_v, attr_active_e, 1);
    partition_manager.local_coarsening(block, 1);

    partition_manager.local_matching(block, attr_matched_v, attr_active_e, 2);
    partition_manager.local_coarsening(block, 2);

    partition_manager.local_matching(block, attr_matched_v, attr_active_e, 3);
    partition_manager.local_coarsening(block, 3);

    partition_manager.local_matching(block, attr_matched_v, attr_active_e, 4);
    partition_manager.local_coarsening(block, 4);

    partition_manager.local_multi_level_partition(block, 5, 2);

    // // iteration num known before kernel -> shared mem known before kernel
    // int i = 0;
    // while (i < req_levels) {
    //     partition_manager.matching(block, i);
    //     partition_manager.coarsening(block, i);
    //     ++i;
    // }

    // // multi-level bipartition one block per patch
    // partition_manager.partition(block, i);

    // i -= 1;
    // while (i > 0) {
    //     partition_manager.uncoarsening(block, i);
    //     // TODO: refinement
    //     // refinement(block, shared_alloc, i);

    //     --i;
    // }

    // partition_manager.genrate_reordering(block, v_ordering);

    // TMP: Check that debug mode is working
    // if (idx == 0)
    //     assert(1 == 0);
}


template <uint32_t blockThreads>
__global__ static void nd_single_patch_test_v_count(rxmesh::Context context,
                                                    uint16_t*       v_count)
{
    using namespace rxmesh;
    // auto           block = cooperative_groups::this_thread_block();
    // ShmemAllocator shrd_alloc;

    // // Init the struct and alloc the shared memory
    // PartitionManager<blockThreads> partition_manager(
    //     block, context, shrd_alloc, 1);

    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx == 0) {
        for (int i = 0; i < context.m_num_patches[0]; i++) {
            v_count[1] += context.m_patches_info[i].num_vertices[0];
            v_count[2] +=
                context.m_patches_info[i].get_num_owned<VertexHandle>();
        }
    }
}

void nd_reorder_single_patch(rxmesh::RXMeshDynamic& rx)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    // vertex color attribute
    auto attr_matched_v =
        rx.add_vertex_attribute<uint16_t>("attr_matched_v", 1);
    auto attr_active_e = rx.add_edge_attribute<uint16_t>("attr_active_e", 1);

    uint16_t req_levels     = 5;
    uint32_t blocks         = rx.get_num_patches();
    uint32_t threads        = blockThreads;
    size_t   smem_bytes_dyn = 0;

    smem_bytes_dyn += (1 + 1 * req_levels) * rx.max_bitmask_size<LocalEdgeT>();
    smem_bytes_dyn +=
        (6 + 4 * req_levels) * rx.max_bitmask_size<LocalVertexT>();
    smem_bytes_dyn +=
        (4 + 5 * req_levels) * rx.get_per_patch_max_edges() * sizeof(uint16_t);
    smem_bytes_dyn += (1 + 4 * req_levels) * rx.get_per_patch_max_vertices() *
                      sizeof(uint16_t);
    smem_bytes_dyn +=
        (11 + 11 * req_levels) * ShmemAllocator::default_alignment;

    RXMESH_TRACE("blocks: {}, threads: {}, smem_bytes: {}",
                 blocks,
                 threads,
                 smem_bytes_dyn);

    // vertex ordering attribute to store the result
    auto v_ordering = rx.add_vertex_attribute<uint16_t>(
        "v_ordering", 1, rxmesh::LOCATION_ALL);
    v_ordering->reset(INVALID16, rxmesh::DEVICE);

    uint32_t* reorder_array;
    CUDA_ERROR(cudaMallocManaged(&reorder_array,
                                 sizeof(uint32_t) * rx.get_num_vertices()));


    // Phase: single patch reordering
    nd_single_patch_main<blockThreads>
        <<<blocks, threads, smem_bytes_dyn>>>(rx.get_context(),
                                              *v_ordering,
                                              *attr_matched_v,
                                              *attr_active_e,
                                              req_levels);
    CUDA_ERROR(cudaDeviceSynchronize());
    RXMESH_TRACE("single patch ordering done");


    // correctness check
    uint16_t* v_count;
    CUDA_ERROR(cudaMallocManaged(&v_count, sizeof(uint16_t) * 5));
    cudaMemset(v_count, 0, sizeof(uint16_t) * 5);
    v_count[0] = rx.get_num_vertices();
    nd_single_patch_test_v_count<blockThreads>
        <<<blocks, threads>>>(rx.get_context(), v_count);

    CUDA_ERROR(cudaDeviceSynchronize());

    RXMESH_INFO("v_count[0] = {}, v_count[1] = {}, v_count[2] = {}",
                v_count[0],
                v_count[1],
                v_count[2]);

    // for timing purposes: measure the time for cuda metis
    // rxmesh::SparseMatrix<float> A_mat(rx);
    // A_mat.spmat_chol_reorder(Reorder::NSTDIS);

#if USE_POLYSCOPE
    // Tests using coloring
    // Move vertex color to the host
    attr_matched_v->move(rxmesh::DEVICE, rxmesh::HOST);
    attr_active_e->move(rxmesh::DEVICE, rxmesh::HOST);

    // polyscope instance associated with rx
    auto polyscope_mesh = rx.get_polyscope_mesh();

    // pass vertex color to polyscope
    polyscope_mesh->addVertexScalarQuantity("attr_matched_v", *attr_matched_v);
    polyscope_mesh->addEdgeScalarQuantity("attr_active_e", *attr_active_e);

    // render
    polyscope::show();
#endif

    RXMESH_TRACE("DONE!!!!!!!!!!!!!!");
}