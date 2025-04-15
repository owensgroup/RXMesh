#include "gtest/gtest.h"

#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include <Eigen/Sparse>

template <uint32_t blockThreads, typename IndexT = int>
__global__ static void sparse_mat_test(const rxmesh::Context context,
                                       IndexT*               vet_degree)
{
    using namespace rxmesh;
    auto compute_valence = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     ids      = v_id.unpack();
        uint32_t patch_id = ids.first;
        uint16_t local_id = ids.second;
        vet_degree[context.vertex_prefix()[patch_id] + local_id] =
            iter.size() + 1;
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, compute_valence);
}

template <typename T, uint32_t blockThreads>
__global__ static void sparse_mat_edge_len_test(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> coords,
    rxmesh::SparseMatrix<T>   sparse_mat,
    T*                         arr_ref)
{
    using namespace rxmesh;
    auto compute_edge_len = [&](VertexHandle&         v_id,
                                const VertexIterator& iter) {
        // reference value calculation
        auto     r_ids      = v_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index = context.vertex_prefix()[r_patch_id] + r_local_id;

        arr_ref[row_index]     = 0;
        sparse_mat(v_id, v_id) = 0;

        vec3<T> v_coord(coords(v_id, 0), coords(v_id, 1), coords(v_id, 2));
        for (uint32_t v = 0; v < iter.size(); ++v) {
            vec3<T> vi_coord(
                coords(iter[v], 0), coords(iter[v], 1), coords(iter[v], 2));

            sparse_mat(v_id, iter[v]) = 1;  // dist(v_coord, vi_coord);

            arr_ref[row_index] += 1;  // dist(v_coord, vi_coord);
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, compute_edge_len);
}


template <uint32_t blockThreads, typename T>
__global__ static void test_transpose(const rxmesh::Context    context,
                                      rxmesh::SparseMatrix<T> mat,
                                      rxmesh::SparseMatrix<T> trans_mat,
                                      int*                     err_count)
{
    using namespace rxmesh;

    for_each<Op::V, blockThreads>(context, [&](VertexHandle& vh) {
        int row_id  = mat.get_row_id(vh);
        int row_nnz = mat.non_zeros(row_id);

        for (int i = 0; i < row_nnz; ++i) {
            int col_id = mat.col_id(row_id, i);

            if (std::abs(mat(row_id, col_id) - trans_mat(col_id, row_id)) >
                1e-6) {
                atomicAdd(err_count, 1);
            }
        }
    });
}

template <typename T>
__global__ void spmat_multi_hardwired_kernel(
    T*                       vec,
    rxmesh::SparseMatrix<T> sparse_mat,
    T*                       out,
    const int                N)
{
    int   tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    if (tid < N) {
        uint32_t start = sparse_mat.row_ptr()[tid];
        uint32_t end   = sparse_mat.row_ptr()[tid + 1];
        for (int i = 0; i < end - start; i++) {
            sum += vec[sparse_mat.col_idx()[start + i]] *
                   sparse_mat.get_val_at(start + i);
        }
        out[tid] = sum;
    }
}

TEST(RXMeshStatic, SparseMatrix)
{
    // Test accessing of the sparse matrix in CSR format in device

    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    uint32_t num_vertices = rx.get_num_vertices();

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

    SparseMatrix<int> spmat(rx);
    spmat.reset(1, LOCATION_ALL);

    spmat_multi_hardwired_kernel<<<blocks, threads>>>(
        d_arr_ones, spmat, d_result, num_vertices);

    std::vector<int> h_result(num_vertices);
    CUDA_ERROR(cudaMemcpy(
        h_result.data(), d_result, num_vertices, cudaMemcpyDeviceToHost));

    // get reference result
    int* vet_degree;
    CUDA_ERROR(cudaMalloc((void**)&vet_degree, (num_vertices) * sizeof(int)));

    LaunchBox<threads> launch_box;
    rx.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_test<threads>);

    // test kernel
    sparse_mat_test<threads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), vet_degree);

    std::vector<int> h_vet_degree(num_vertices);
    CUDA_ERROR(cudaMemcpy(
        h_vet_degree.data(), vet_degree, num_vertices, cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < num_vertices; ++i) {
        EXPECT_EQ(h_result[i], h_vet_degree[i]);
    }


    CUDA_ERROR(cudaFree(d_arr_ones));
    CUDA_ERROR(cudaFree(d_result));
    CUDA_ERROR(cudaFree(vet_degree));
    spmat.release();
}

TEST(RXMeshStatic, SparseMatrixEdgeLen)
{
    // First replace the sparse matrix entry with the edge length and then do
    // spmv with an all one array and check the result
    //
    using namespace rxmesh;


    // generate rxmesh obj
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    const uint32_t threads = 256;
    const uint32_t blocks  = DIVIDE_UP(num_vertices, threads);

    auto coords = rx.get_input_vertex_coordinates();

    float* d_arr_ones;

    std::vector<float> init_tmp_arr(num_vertices, 1.f);
    CUDA_ERROR(cudaMalloc((void**)&d_arr_ones, (num_vertices) * sizeof(float)));
    CUDA_ERROR(cudaMemcpy(d_arr_ones,
                          init_tmp_arr.data(),
                          num_vertices * sizeof(float),
                          cudaMemcpyHostToDevice));

    SparseMatrix<float> spmat(rx);

    float* d_arr_ref;
    float* d_result;

    CUDA_ERROR(cudaMalloc((void**)&d_arr_ref, (num_vertices) * sizeof(float)));
    CUDA_ERROR(cudaMalloc((void**)&d_result, (num_vertices) * sizeof(float)));

    LaunchBox<threads> launch_box;
    rx.prepare_launch_box(
        {Op::VV}, launch_box, (void*)sparse_mat_edge_len_test<float, threads>);

    sparse_mat_edge_len_test<float, threads><<<launch_box.blocks,
                                               launch_box.num_threads,
                                               launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *coords, spmat, d_arr_ref);

    spmat.multiply(d_arr_ones, d_result);

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
    spmat.release();
}

TEST(RXMeshStatic, SparseMatrixUserManaged)
{
    using namespace rxmesh;
    using T = float;

    int rows = 10;
    int cols = 10;
    int nnz  = static_cast<int>(0.2 * rows * cols);

    std::random_device                 rd;
    std::mt19937                       gen(rd());
    std::uniform_real_distribution<T>  value_dist(0.0, 1.0);
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);

    std::vector<Eigen::Triplet<T>> triplets;
    for (int i = 0; i < nnz; ++i) {
        int    r   = row_dist(gen);
        int    c   = col_dist(gen);
        double val = value_dist(gen);
        triplets.emplace_back(r, c, val);
    }

    Eigen::SparseMatrix<T, Eigen::RowMajor, int> eigen_mat(rows, cols);
    eigen_mat.setFromTriplets(triplets.begin(), triplets.end());
    eigen_mat.makeCompressed();

    int* h_row_ptr = eigen_mat.outerIndexPtr();  // Row pointers
    int* h_col_idx = eigen_mat.innerIndexPtr();  // Column indices
    T*   h_val     = eigen_mat.valuePtr();       // Nonzero values

    int *d_row_ptr(nullptr), *d_col_idx(nullptr);
    T*   d_val(nullptr);

    CUDA_ERROR(cudaMalloc((void**)&d_row_ptr, sizeof(int) * (rows + 1)));
    CUDA_ERROR(cudaMalloc((void**)&d_col_idx, sizeof(int) * nnz));
    CUDA_ERROR(cudaMalloc((void**)&d_val, sizeof(T) * nnz));

    CUDA_ERROR(cudaMemcpy(d_row_ptr,
                          h_row_ptr,
                          sizeof(int) * (rows + 1),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMemcpy(
        d_col_idx, h_col_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CUDA_ERROR(
        cudaMemcpy(d_val, h_val, sizeof(T) * nnz, cudaMemcpyHostToDevice));


    SparseMatrix mat(rows,
                      cols,
                      nnz,
                      d_row_ptr,
                      d_col_idx,
                      d_val,
                      h_row_ptr,
                      h_col_idx,
                      h_val);


    EXPECT_EQ(rows, mat.rows());
    EXPECT_EQ(cols, mat.cols());
    EXPECT_EQ(nnz, mat.non_zeros());

    for (int row = 0; row < eigen_mat.outerSize(); ++row) {
        for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(
                 eigen_mat, row);
             it;
             ++it) {
            EXPECT_NEAR(it.value(), mat(it.row(), it.col()), 1e-6);
        }
    }

    GPU_FREE(d_row_ptr);
    GPU_FREE(d_col_idx);
    GPU_FREE(d_val);

    mat.release();
}


TEST(RXMeshStatic, SparseMatrixTranspose)
{
    using namespace rxmesh;
    using T = float;

    std::random_device                rd;
    std::mt19937                      gen(rd());
    std::uniform_real_distribution<T> value_dist(0.0, 1.0);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    SparseMatrix<T> mat(rx);

    for (int i = 0; i < mat.non_zeros(); ++i) {
        mat.get_val_at(i) = value_dist(gen);
    }
    mat.move(HOST, DEVICE);

    SparseMatrix<T> mat_trans = mat.transpose();

    int* d_err_count(nullptr);
    CUDA_ERROR(cudaMalloc((void**)&d_err_count, sizeof(int)));
    CUDA_ERROR(cudaMemset(d_err_count, 0, sizeof(int)));

    rx.run_kernel<256>(
        {Op::V}, test_transpose<256, T>, mat, mat_trans, d_err_count);

    int h_err_count = 0;
    CUDA_ERROR(cudaMemcpy(
        &h_err_count, d_err_count, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_err_count, 0);

    mat.release();
    mat_trans.release();
    GPU_FREE(d_err_count);
}