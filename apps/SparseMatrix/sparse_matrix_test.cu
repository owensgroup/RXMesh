#include <cuda_profiler_api.h>
#include "gtest/gtest.h"

#include "rxmesh/attribute.h"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "matrix_operation.cuh"
#include "sparse_matrix_mcf.cuh"

template <uint32_t blockThreads, typename IndexT = int>
__global__ static void sparse_mat_test(const rxmesh::Context context,
                                       IndexT*               vet_degree)
{
    using namespace rxmesh;

    auto compute_valence = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     ids      = v_id.unpack();
        uint32_t patch_id = ids.first;
        uint16_t local_id = ids.second;
        vet_degree[context.m_vertex_prefix[patch_id] + local_id] =
            iter.size() + 1;
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, compute_valence);
}

template <uint32_t blockThreads>
__global__ static void sparse_mat_query_test(
    const rxmesh::Context     context,
    rxmesh::SparseMatrix<int> sparse_mat)
{
    using namespace rxmesh;
    auto fillin = [&](VertexHandle& v_id, const VertexIterator& iter) {
        sparse_mat(v_id, v_id) = 2;
        for (uint32_t v = 0; v < iter.size(); ++v) {
            sparse_mat(v_id, iter[v]) = 2;
            sparse_mat(iter[v], v_id) = 2;
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, fillin);
}

template <typename T, uint32_t blockThreads>
__global__ static void sparse_mat_edge_len_test(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> coords,
    rxmesh::SparseMatrix<T>    sparse_mat,
    T*                         arr_ref)
{
    using namespace rxmesh;
    auto compute_edge_len = [&](VertexHandle&         v_id,
                                const VertexIterator& iter) {
        // reference value calculation
        auto     r_ids      = v_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index =
            sparse_mat.m_context.m_vertex_prefix[r_patch_id] + r_local_id;

        arr_ref[row_index]     = 0;
        sparse_mat(v_id, v_id) = 0;

        Vector<3, T> v_coord(coords(v_id, 0), coords(v_id, 1), coords(v_id, 2));
        for (uint32_t v = 0; v < iter.size(); ++v) {
            Vector<3, T> vi_coord(
                coords(iter[v], 0), coords(iter[v], 1), coords(iter[v], 2));

            sparse_mat(v_id, iter[v]) = dist(v_coord, vi_coord);

            arr_ref[row_index] += dist(v_coord, vi_coord);
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, compute_edge_len);
}

template <typename T>
__global__ void spmat_multi_hardwired_kernel(T*                      vec,
                                             rxmesh::SparseMatrix<T> sparse_mat,
                                             T*                      out,
                                             const int               N)
{
    int   tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    if (tid < N) {
        uint32_t start = sparse_mat.m_d_row_ptr[tid];
        uint32_t end   = sparse_mat.m_d_row_ptr[tid + 1];
        for (int i = 0; i < end - start; i++) {
            sum += vec[sparse_mat.m_d_col_idx[start + i]] *
                   sparse_mat.m_d_val[start + i];
        }
        out[tid] = sum;
    }
}

template <typename T, uint32_t blockThreads>
__global__ static void simple_A_X_B_setup(const rxmesh::Context      context,
                                          rxmesh::VertexAttribute<T> coords,
                                          rxmesh::SparseMatrix<T>    A_mat,
                                          rxmesh::DenseMatrix<T>     X_mat,
                                          rxmesh::DenseMatrix<T>     B_mat,
                                          const T                    time_step)
{
    using namespace rxmesh;
    auto mat_setup = [&](VertexHandle& v_id, const VertexIterator& iter) {
        T sum_e_weight(0);

        T v_weight = iter.size();

        // reference value calculation
        auto     r_ids      = v_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index =
            A_mat.m_context.m_vertex_prefix[r_patch_id] + r_local_id;

        B_mat(row_index, 0) = iter.size() * 7;
        B_mat(row_index, 1) = iter.size() * 2;
        B_mat(row_index, 2) = iter.size() * 10;

        X_mat(row_index, 0) = coords(v_id, 0) * v_weight;
        X_mat(row_index, 1) = coords(v_id, 1) * v_weight;
        X_mat(row_index, 2) = coords(v_id, 2) * v_weight;

        Vector<3, float> vi_coord(
            coords(v_id, 0), coords(v_id, 1), coords(v_id, 2));
        for (uint32_t v = 0; v < iter.size(); ++v) {
            T e_weight           = 1;
            A_mat(v_id, iter[v]) = -time_step * e_weight;

            sum_e_weight += e_weight;
        }

        A_mat(v_id, v_id) = v_weight + time_step * sum_e_weight;
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, mat_setup);
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

    std::vector<int> init_tmp_arr(num_vertices, 1);
    CUDA_ERROR(cudaMalloc((void**)&d_arr_ones, (num_vertices) * sizeof(int)));
    CUDA_ERROR(cudaMemcpy(d_arr_ones,
                          init_tmp_arr.data(),
                          num_vertices * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMalloc((void**)&d_result, (num_vertices) * sizeof(int)));

    SparseMatrix<int> spmat(rxmesh);
    spmat.set_ones();

    spmat_multi_hardwired_kernel<<<blocks, threads>>>(
        d_arr_ones, spmat, d_result, num_vertices);

    std::vector<int> h_result(num_vertices);
    CUDA_ERROR(cudaMemcpy(
        h_result.data(), d_result, num_vertices, cudaMemcpyDeviceToHost));

    // get reference result
    int* vet_degree;
    CUDA_ERROR(cudaMalloc((void**)&vet_degree, (num_vertices) * sizeof(int)));

    LaunchBox<threads> launch_box;
    rxmesh.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_test<threads>);

    sparse_mat_test<threads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rxmesh.get_context(), vet_degree);

    std::vector<int> h_vet_degree(num_vertices);
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

    SparseMatrix<int> spmat(rxmesh);
    spmat.set_ones();

    LaunchBox<threads> launch_box;
    rxmesh.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_query_test<threads>);

    sparse_mat_query_test<threads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rxmesh.get_context(), spmat);

    std::vector<uint32_t> h_result(spmat.m_nnz);
    CUDA_ERROR(cudaMemcpy(h_result.data(),
                          spmat.m_d_val,
                          spmat.m_nnz * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<uint32_t> h_ref(spmat.m_nnz, 2);

    for (int i = 0; i < spmat.m_nnz; ++i) {
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
    std::string  obj_path = STRINGIFY(INPUT_DIR) "cube.obj";
    RXMeshStatic rxmesh(obj_path);

    uint32_t num_vertices = rxmesh.get_num_vertices();

    const uint32_t threads = 256;
    const uint32_t blocks  = DIVIDE_UP(num_vertices, threads);

    auto coords = rxmesh.get_input_vertex_coordinates();

    float* d_arr_ones;

    std::vector<float> init_tmp_arr(num_vertices, 1.f);
    CUDA_ERROR(cudaMalloc((void**)&d_arr_ones, (num_vertices) * sizeof(float)));
    CUDA_ERROR(cudaMemcpy(d_arr_ones,
                          init_tmp_arr.data(),
                          num_vertices * sizeof(float),
                          cudaMemcpyHostToDevice));

    SparseMatrix<float> spmat(rxmesh);

    float* d_arr_ref;
    float* d_result;

    CUDA_ERROR(cudaMalloc((void**)&d_arr_ref, (num_vertices) * sizeof(float)));
    CUDA_ERROR(cudaMalloc((void**)&d_result, (num_vertices) * sizeof(float)));

    LaunchBox<threads> launch_box;
    rxmesh.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_edge_len_test<float, threads>);

    sparse_mat_edge_len_test<float, threads><<<launch_box.blocks,
                                               launch_box.num_threads,
                                               launch_box.smem_bytes_dyn>>>(
        rxmesh.get_context(), *coords, spmat, d_arr_ref);

    // Spmat matrix multiply

    // spmat_multi_hardwired_kernel<float>
    //     <<<blocks, threads>>>(d_arr_ones, spmat, d_result, num_vertices);
    spmat_arr_mul(spmat, d_arr_ones, d_result);

    // copy the value back to host
    std::vector<float> h_arr_ref(num_vertices);
    CUDA_ERROR(cudaMemcpy(h_arr_ref.data(),
                          d_arr_ref,
                          num_vertices * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::vector<float> h_result(num_vertices);
    CUDA_ERROR(cudaMemcpy(h_result.data(),
                          d_result,
                          num_vertices * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < num_vertices; ++i) {        
        EXPECT_FLOAT_EQ(h_result[i], h_arr_ref[i]);
    }

    CUDA_ERROR(cudaFree(d_arr_ref));
    CUDA_ERROR(cudaFree(d_arr_ones));
    CUDA_ERROR(cudaFree(d_result));
    spmat.free();
}

TEST(Apps, SparseMatrixSimpleSolve)
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

    auto                coords = rxmesh.get_input_vertex_coordinates();
    SparseMatrix<float> A_mat(rxmesh);
    DenseMatrix<float>  X_mat(num_vertices, 3);
    DenseMatrix<float>  B_mat(num_vertices, 3);
    DenseMatrix<float>  ret_mat(num_vertices, 3);

    float time_step = 1.f;

    LaunchBox<threads> launch_box;
    rxmesh.prepare_launch_box(
        {Op::VV}, launch_box, (void*)simple_A_X_B_setup<float, threads>);

    simple_A_X_B_setup<float, threads><<<launch_box.blocks,
                                         launch_box.num_threads,
                                         launch_box.smem_bytes_dyn>>>(
        rxmesh.get_context(), *coords, A_mat, X_mat, B_mat, time_step);

    spmat_linear_solve(A_mat, B_mat, X_mat, Solver::CHOL, Reorder::NONE);

    spmat_denmat_mul(A_mat, X_mat, ret_mat);
    // spmat_denmat_mul_cw(A_mat, X_mat, ret_mat);

    std::vector<float> h_ret_mat(num_vertices * 3);
    cudaMemcpy(h_ret_mat.data(),
               ret_mat.data(),
               num_vertices * 3,
               cudaMemcpyDeviceToHost);
    std::vector<float> h_B_mat(num_vertices * 3);
    cudaMemcpy(
        h_B_mat.data(), B_mat.data(), num_vertices * 3, cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < num_vertices * 3; ++i) {
        EXPECT_NEAR(h_ret_mat[i], h_B_mat[i], 1e-3);
    }
}

int main(int argc, char** argv)
{
    using namespace rxmesh;
    Log::init();

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}