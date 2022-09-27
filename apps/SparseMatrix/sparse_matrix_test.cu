#include <cuda_profiler_api.h>
#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "sparse_matrix.cuh"

template <uint32_t blockThreads>
__global__ static void sparse_mat_test(const rxmesh::Context context,
                                       uint32_t*             patch_ptr_v,
                                       uint32_t*             vet_degree)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        // printf(" %" PRIu32 " - %" PRIu32 " - %" PRIu32 " - %" PRIu32 " \n",
        //        row_ptr[0],
        //        row_ptr[1],
        //        row_ptr[2],
        //        row_ptr[3]);
        auto     ids                                 = v_id.unpack();
        uint32_t patch_id                            = ids.first;
        uint16_t local_id                            = ids.second;
        vet_degree[patch_ptr_v[patch_id] + local_id] = iter.size() + 1;
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, init_lambda);
}

template <uint32_t blockThreads>
__global__ static void sparse_mat_query_test(
    const rxmesh::Context      context,
    rxmesh::SparseMatInfo<int> sparse_mat)
{
    using namespace rxmesh;
    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        sparse_mat(v_id, v_id) = 2;
        for (uint32_t v = 0; v < iter.size(); ++v) {
            sparse_mat(v_id, iter[v]) = 2;
            sparse_mat(iter[v], v_id) = 2;
        }
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, init_lambda);
}

template <uint32_t blockThreads>
__global__ static void sparse_mat_edge_len_test(
    const rxmesh::Context          context,
    rxmesh::VertexAttribute<float> coords,
    rxmesh::SparseMatInfo<int>     sparse_mat,
    int*                           arr_ref)
{
    using namespace rxmesh;
    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        // reference value calculation
        auto     r_ids      = v_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index =
            sparse_mat.m_d_patch_ptr_v[r_patch_id] + r_local_id;

        arr_ref[row_index]     = 0;
        sparse_mat(v_id, v_id) = 0;

        Vector<3, float> v_coord(
            coords(v_id, 0), coords(v_id, 1), coords(v_id, 2));
        for (uint32_t v = 0; v < iter.size(); ++v) {
            Vector<3, float> vi_coord(
                coords(iter[v], 0), coords(iter[v], 1), coords(iter[v], 2));
            sparse_mat(v_id, iter[v]) = dist(v_coord, vi_coord);
            sparse_mat(iter[v], v_id) = sparse_mat(v_id, iter[v]);

            arr_ref[row_index] += dist(v_coord, vi_coord);
        }
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, init_lambda);
}

__global__ void spmat_multi_hardwired_kernel(
    int*                       vec,
    rxmesh::SparseMatInfo<int> sparse_mat,
    int*                       out,
    const int                  N)
{
    int   tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    if (tid < N) {
        for (int i = 0;
             i < sparse_mat.m_d_row_ptr[tid + 1] - sparse_mat.m_d_row_ptr[tid];
             i++)
            sum += vec[sparse_mat.m_d_col_idx[tid + i]] *
                   sparse_mat.m_d_val[tid + i];
        out[tid] = sum;
    }
}

TEST(Apps, PatchPointer)
{
    using namespace rxmesh;

    cuda_query(0);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    // move the patch ptr to the host so we can test it
    std::vector<uint32_t> h_ptchptr(rx.get_num_patches() + 1);

    SparseMatInfo<int> spmat(rx);

    // vertices
    CUDA_ERROR(cudaMemcpy(h_ptchptr.data(),
                          spmat.m_d_patch_ptr_v,
                          h_ptchptr.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_ptchptr.back(), rx.get_num_vertices());

    for (uint32_t i = 0; i < rx.get_num_patches(); ++i) {
        EXPECT_EQ(h_ptchptr[i + 1] - h_ptchptr[i],
                  rx.get_patches_info()[i].num_owned_vertices);
    }

    // edges
    CUDA_ERROR(cudaMemcpy(h_ptchptr.data(),
                          spmat.m_d_patch_ptr_e,
                          h_ptchptr.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_ptchptr.back(), rx.get_num_edges());

    for (uint32_t i = 0; i < rx.get_num_patches(); ++i) {
        EXPECT_EQ(h_ptchptr[i + 1] - h_ptchptr[i],
                  rx.get_patches_info()[i].num_owned_edges);
    }

    // faces
    CUDA_ERROR(cudaMemcpy(h_ptchptr.data(),
                          spmat.m_d_patch_ptr_f,
                          h_ptchptr.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_ptchptr.back(), rx.get_num_faces());

    for (uint32_t i = 0; i < rx.get_num_patches(); ++i) {
        EXPECT_EQ(h_ptchptr[i + 1] - h_ptchptr[i],
                  rx.get_patches_info()[i].num_owned_faces);
    }

    spmat.free();
}

TEST(Apps, SparseMatrix)
{
    using namespace rxmesh;

    // Select device
    cuda_query(0);

    // generate rxmesh obj
    std::string  obj_path = STRINGIFY(INPUT_DIR) "dragon.obj";
    RXMeshStatic rxmesh(obj_path);

    uint32_t num_vertices = rxmesh.get_num_vertices();

    const uint32_t threads = 256;
    const uint32_t blocks  = DIVIDE_UP(num_vertices, threads);

    int* d_arr_ones;
    int* d_result;

    std::vector<uint32_t> init_tmp_arr(num_vertices, 1);
    CUDA_ERROR(cudaMalloc((void**)&d_arr_ones, (num_vertices) * sizeof(int)));
    CUDA_ERROR(cudaMemcpy(d_arr_ones,
                          init_tmp_arr.data(),
                          num_vertices * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMalloc((void**)&d_result, (num_vertices) * sizeof(int)));

    SparseMatInfo<int> spmat(rxmesh);
    spmat.set_ones();

    spmat_multi_hardwired_kernel<<<blocks, threads>>>(
        d_arr_ones, spmat, d_result, num_vertices);

    std::vector<uint32_t> h_result(num_vertices);
    CUDA_ERROR(cudaMemcpy(
        h_result.data(), d_result, num_vertices, cudaMemcpyDeviceToHost));

    // get reference result
    uint32_t* vet_degree;
    CUDA_ERROR(
        cudaMalloc((void**)&vet_degree, (num_vertices) * sizeof(uint32_t)));

    LaunchBox<threads> launch_box;
    rxmesh.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_test<threads>);

    sparse_mat_test<threads><<<launch_box.blocks,
                               launch_box.num_threads,
                               launch_box.smem_bytes_dyn>>>(
        rxmesh.get_context(), spmat.m_d_patch_ptr_v, vet_degree);

    std::vector<uint32_t> h_vet_degree(num_vertices);
    CUDA_ERROR(cudaMemcpy(
        h_vet_degree.data(), vet_degree, num_vertices, cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < num_vertices; ++i) {
        EXPECT_EQ(h_result[i], h_vet_degree[i]);
    }


    CUDA_ERROR(cudaFree(d_arr_ones));
    CUDA_ERROR(cudaFree(d_result));
    CUDA_ERROR(cudaFree(vet_degree));
    spmat.free();
}

TEST(Apps, SparseMatrixQuery)
{
    using namespace rxmesh;

    // Select device
    cuda_query(0);

    // generate rxmesh obj
    std::string  obj_path = STRINGIFY(INPUT_DIR) "cube.obj";
    RXMeshStatic rxmesh(obj_path);

    uint32_t num_vertices = rxmesh.get_num_vertices();

    const uint32_t threads = 256;
    const uint32_t blocks  = DIVIDE_UP(num_vertices, threads);

    SparseMatInfo<int> spmat(rxmesh);
    spmat.set_ones();

    LaunchBox<threads> launch_box;
    rxmesh.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_query_test<threads>);

    sparse_mat_query_test<threads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rxmesh.get_context(), spmat);

    std::vector<uint32_t> h_result(spmat.m_nnz_entry_size);
    CUDA_ERROR(cudaMemcpy(h_result.data(),
                          spmat.m_d_val,
                          spmat.m_nnz_entry_size * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<uint32_t> h_ref(spmat.m_nnz_entry_size, 2);

    for (uint32_t i = 0; i < spmat.m_nnz_entry_size; ++i) {
        EXPECT_EQ(h_result[i], h_ref[i]);
    }

    spmat.free();
}

TEST(Apps, SparseMatrixEdgeLen)
{
    using namespace rxmesh;

    // Select device
    cuda_query(0);

    // generate rxmesh obj
    std::string  obj_path = STRINGIFY(INPUT_DIR) "dragon.obj";
    RXMeshStatic rxmesh(obj_path);

    uint32_t num_vertices = rxmesh.get_num_vertices();

    const uint32_t threads = 256;
    const uint32_t blocks  = DIVIDE_UP(num_vertices, threads);

    std::vector<std::vector<float>> Verts;
    auto coords = rxmesh.get_input_vertex_coordinates();

    int* d_arr_ones;

    std::vector<uint32_t> init_tmp_arr(num_vertices, 1);
    CUDA_ERROR(cudaMalloc((void**)&d_arr_ones, (num_vertices) * sizeof(int)));
    CUDA_ERROR(cudaMemcpy(d_arr_ones,
                          init_tmp_arr.data(),
                          num_vertices * sizeof(int),
                          cudaMemcpyHostToDevice));

    SparseMatInfo<int> spmat(rxmesh);

    int* d_arr_ref;
    int* d_result;

    CUDA_ERROR(
        cudaMalloc((void**)&d_arr_ref, (spmat.m_nnz_entry_size) * sizeof(int)));
    CUDA_ERROR(
        cudaMalloc((void**)&d_result, (spmat.m_nnz_entry_size) * sizeof(int)));

    LaunchBox<threads> launch_box;
    rxmesh.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_edge_len_test<threads>);

    sparse_mat_edge_len_test<threads><<<launch_box.blocks,
                                        launch_box.num_threads,
                                        launch_box.smem_bytes_dyn>>>(
        rxmesh.get_context(), *coords, spmat, d_arr_ref);

    // Spmat matrix multiply

    spmat_multi_hardwired_kernel<<<blocks, threads>>>(
        d_arr_ones, spmat, d_result, num_vertices);

    // copy the value back to host
    std::vector<uint32_t> h_arr_ref(num_vertices);
    CUDA_ERROR(cudaMemcpy(h_arr_ref.data(),
                          d_arr_ref,
                          num_vertices * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<uint32_t> h_result(num_vertices);
    CUDA_ERROR(cudaMemcpy(h_result.data(),
                          d_result,
                          num_vertices * sizeof(int),
                          cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < num_vertices; ++i) {
        EXPECT_EQ(h_result[i], h_arr_ref[i]);
    }

    CUDA_ERROR(cudaFree(d_arr_ref));
    CUDA_ERROR(cudaFree(d_arr_ones));
    CUDA_ERROR(cudaFree(d_result));
    spmat.free();
}

int main(int argc, char** argv)
{
    using namespace rxmesh;
    Log::init();

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
