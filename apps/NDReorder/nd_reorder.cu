#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"

#include "nd_cross_patch_oedering.cuh"
#include "nd_reorder_kernel.cuh"

#include <vector>
#include "cusparse.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/types.h"

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/rxmesh_dynamic.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    uint32_t    device_id     = 0;
} Arg;

template <typename T>
void cub_prefix_sum_wrap(T* in_arr, T* out_arr, uint16_t size)
{
    using namespace rxmesh;

    void*  d_cub_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_cub_temp_storage, temp_storage_bytes, in_arr, out_arr, size);
    CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage, temp_storage_bytes));

    cub::DeviceScan::ExclusiveSum(
        d_cub_temp_storage, temp_storage_bytes, in_arr, out_arr, size);

    CUDA_ERROR(cudaFree(d_cub_temp_storage));
}

template<typename T>
void swapPointers(T*& ptr1, T*& ptr2) {
    T* temp = ptr1;
    ptr1 = ptr2;
    ptr2 = temp;
}

template <uint32_t blockThreads>
void cross_patch_ordering(rxmesh::RXMeshStatic&             rx,
                          rxmesh::VertexAttribute<uint16_t> v_ordering,
                          uint32_t                          smem_bytes_dyn)
{
    using namespace rxmesh;

    uint16_t* num_node;
    cudaMallocManaged(&num_node, sizeof(int));
    *num_node = rx.get_num_patches();

    uint16_t level = 0;

    RXMESH_TRACE("Matching");
    match_patches_init_edge_weight<blockThreads>
        <<<rx.get_num_patches(), blockThreads, smem_bytes_dyn>>>(
            rx.get_context());
    CUDA_ERROR(cudaDeviceSynchronize());

    match_patches_init_param<blockThreads>
        <<<rx.get_num_patches(), blockThreads>>>(rx.get_context());
    CUDA_ERROR(cudaDeviceSynchronize());

    while (*num_node > 1) {
        *num_node                    = 1;
        uint16_t is_matching_counter = 0;
        while (is_matching_counter < 10) {
            match_patches_select<blockThreads>
                <<<rx.get_num_patches(), blockThreads>>>(rx.get_context(),
                                                         level);
            CUDA_ERROR(cudaDeviceSynchronize());

            match_patches_confirm<blockThreads>
                <<<rx.get_num_patches(), blockThreads>>>(rx.get_context(),
                                                         level);
            CUDA_ERROR(cudaDeviceSynchronize());

            // update the is_matching_flag here
            ++is_matching_counter;
        }

        match_patches_update_node<blockThreads>
            <<<rx.get_num_patches(), blockThreads>>>(rx.get_context(), level);
        CUDA_ERROR(cudaDeviceSynchronize());

        match_patches_update_not_node<blockThreads>
            <<<rx.get_num_patches(), blockThreads>>>(rx.get_context(), level);
        CUDA_ERROR(cudaDeviceSynchronize());

        match_patches_update_next_level<blockThreads>
            <<<rx.get_num_patches(), blockThreads>>>(
                rx.get_context(), level, num_node);
        CUDA_ERROR(cudaDeviceSynchronize());

        check<blockThreads>
            <<<rx.get_num_patches(), blockThreads>>>(rx.get_context(), level);
        CUDA_ERROR(cudaDeviceSynchronize());

        // update level counter
        ++level;

        printf("level: %d, num_node: %d\n", level, *num_node);

        // update the num_node here
        if (level > 50) {
            *num_node = 1;
            RXMESH_ERROR("Too many levels");
        }
    }


    uint16_t* patch_list_size;
    CUDA_ERROR(cudaMallocManaged(&patch_list_size, sizeof(uint16_t) * 1));

    // patch_prefix_sum is just for preparing the next level of the patch list
    // marking the whether the patch would be expanded or not
    uint16_t* patch_prefix_sum;
    CUDA_ERROR(cudaMallocManaged(
        &patch_prefix_sum, sizeof(uint16_t) * 2 * (rx.get_num_patches() + 2)));
    CUDA_ERROR(cudaMemset(patch_prefix_sum,
                          INVALID16,
                          sizeof(uint16_t) * 2 * (rx.get_num_patches() + 2)));

    // record how many pieces including the vertex separtors
    uint16_t* patch_list;
    CUDA_ERROR(cudaMallocManaged(&patch_list,
                                 sizeof(uint16_t) * 2 * rx.get_num_patches()));
    CUDA_ERROR(cudaMemset(
        patch_list, INVALID16, sizeof(uint16_t) * 2 * rx.get_num_patches()));

    uint16_t* next_patch_list;
    CUDA_ERROR(cudaMallocManaged(&next_patch_list,
                                 sizeof(uint16_t) * 2 * rx.get_num_patches()));
    CUDA_ERROR(cudaMemset(next_patch_list,
                          INVALID16,
                          sizeof(uint16_t) * 2 * rx.get_num_patches()));

    // patch_num_v_prefix_sum record the start index and the end index for each
    // pieces in a prefix sum manner
    uint32_t* patch_num_v_prefix_sum;
    CUDA_ERROR(
        cudaMallocManaged(&patch_num_v_prefix_sum,
                          sizeof(uint32_t) * 2 * (rx.get_num_patches() + 2)));
    CUDA_ERROR(cudaMemset(patch_num_v_prefix_sum,
                          0,
                          sizeof(uint32_t) * 2 * (rx.get_num_patches() + 2)));

    uint32_t* next_patch_num_v_prefix_sum;
    CUDA_ERROR(
        cudaMallocManaged(&next_patch_num_v_prefix_sum,
                          sizeof(uint32_t) * 2 * (rx.get_num_patches() + 2)));
    CUDA_ERROR(cudaMemset(next_patch_num_v_prefix_sum,
                          0,
                          sizeof(uint32_t) * 2 * (rx.get_num_patches() + 2)));

    // init the parameters for the coarest level
    ordering_init_top_layer<blockThreads>
        <<<rx.get_num_patches(), blockThreads>>>(rx.get_context(),
                                                 level,
                                                 patch_list_size,
                                                 patch_prefix_sum,
                                                 patch_list,
                                                 patch_num_v_prefix_sum);
    CUDA_ERROR(cudaDeviceSynchronize());

    uint16_t tmp_counter = 0;
    while (level > 0) {
        ordering_generate_prefix_sum<blockThreads>
            <<<rx.get_num_patches(), blockThreads>>>(rx.get_context(),
                                                     level,
                                                     patch_list_size,
                                                     patch_prefix_sum,
                                                     patch_list,
                                                     patch_num_v_prefix_sum);
        CUDA_ERROR(cudaDeviceSynchronize());

        cub_prefix_sum_wrap(
            patch_prefix_sum, patch_prefix_sum, patch_list_size[0] + 1);

        ordering_extract_vertices<blockThreads>
            <<<rx.get_num_patches(), blockThreads, smem_bytes_dyn>>>(
                rx.get_context(),
                level,
                patch_list_size,
                patch_prefix_sum,
                patch_list,
                patch_num_v_prefix_sum,
                next_patch_num_v_prefix_sum,
                v_ordering);
        CUDA_ERROR(cudaDeviceSynchronize());

        printf("next_patch_num_v_prefix_sum: ");
        for (uint16_t i = 0; i < patch_list_size[0] + 1; ++i) {
            printf(" %d ", next_patch_num_v_prefix_sum[i]);
        }
        printf("\n");

        ordering_generate_finer_level<blockThreads>
            <<<rx.get_num_patches(), blockThreads, smem_bytes_dyn>>>(
                rx.get_context(),
                level,
                patch_list_size,
                patch_prefix_sum,
                patch_list,
                next_patch_list,
                patch_num_v_prefix_sum,
                next_patch_num_v_prefix_sum,
                v_ordering);
        CUDA_ERROR(cudaDeviceSynchronize());

        printf("next_patch_num_v_prefix_sum: ");
        for (uint16_t i = 0; i < patch_list_size[0] + 1; ++i) {
            printf(" %d ", next_patch_num_v_prefix_sum[i]);
        }
        printf("\n");

        cub_prefix_sum_wrap(
            next_patch_num_v_prefix_sum, next_patch_num_v_prefix_sum, patch_list_size[0] + 1);

        printf("--------------------\n");
        printf("uncoarsen level: %d\n", level);

        printf("patch_list_size: %d\n", patch_list_size[0]);

        printf("patch_prefix_sum: ");
        for (uint16_t i = 0; i < patch_list_size[0] + 1; ++i) {
            printf(" %d ", patch_prefix_sum[i]);
        }
        printf("\n");

        printf("patch_list: ");
        for (uint16_t i = 0; i < patch_list_size[0] + 1; ++i) {
            printf(" %d ", patch_list[i]);
        }
        printf("\n");

        printf("next_patch_list: ");
        for (uint16_t i = 0; i < patch_list_size[0] + 1; ++i) {
            printf(" %d ", next_patch_list[i]);
        }
        printf("\n");

        printf("patch_num_v_prefix_sum: ");
        for (uint16_t i = 0; i < patch_list_size[0] + 1; ++i) {
            printf(" %d ", patch_num_v_prefix_sum[i]);
        }
        printf("\n");

        printf("next_patch_num_v_prefix_sum: ");
        for (uint16_t i = 0; i < patch_list_size[0] + 1; ++i) {
            printf(" %d ", next_patch_num_v_prefix_sum[i]);
        }
        printf("\n");
        printf("--------------------\n");

        swapPointers(patch_list, next_patch_list);
        swapPointers(patch_num_v_prefix_sum, next_patch_num_v_prefix_sum);

        // clear the next level
        CUDA_ERROR(cudaMemset(next_patch_list,
                              INVALID16,
                              sizeof(uint16_t) * 2 * rx.get_num_patches()));
        CUDA_ERROR(cudaMemset(next_patch_num_v_prefix_sum,
                          0,
                          sizeof(uint32_t) * 2 * (rx.get_num_patches() + 2)));
            
        --level;
        ++tmp_counter;
        if (tmp_counter == 2) {
            break;
        }
    }
}

void nd_reorder()
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    RXMeshStatic rx(Arg.obj_file_name);

    // rx.save(STRINGIFY(OUTPUT_DIR) + extract_file_name(Arg.obj_file_name) +
    //         "_nd_patches");

    // RXMeshDynamic rx(Arg.obj_file_name,
    //                  STRINGIFY(OUTPUT_DIR) +
    //                      extract_file_name(Arg.obj_file_name) +
    //                      "_nd_patches");

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

    // Phase: cross patch reordering
    cross_patch_ordering<blockThreads>(rx, *v_ordering, smem_bytes_dyn);

    // Phase: single patch reordering
    // nd_single_patch_main<blockThreads><<<blocks, threads, smem_bytes_dyn>>>(
    //     rx.get_context(), *v_reorder, *attr_matched_v, *attr_active_e,
    //     req_levels);

    CUDA_ERROR(cudaDeviceSynchronize());

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

TEST(Apps, NDReorder)
{
    using namespace rxmesh;

    // Select device
    cuda_query(Arg.device_id);

    // nd reorder implementation
    nd_reorder();
}

int main(int argc, char** argv)
{
    using namespace rxmesh;
    Log::init();

    ::testing::InitGoogleTest(&argc, argv);

    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: NDReorder.exe < -option X>\n"
                        " -h:          Display this massage and exits\n"
                        " -input:      Input file. Input file should under the input/ subdirectory\n"
                        "              Default is {} \n"
                        "              Hint: Only accepts OBJ files\n"                                              
                        " -device_id:  GPU device ID. Default is {}",
            Arg.obj_file_name,  Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-input")) {
            Arg.obj_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-input"));
        }

        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            Arg.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}


// batch info file