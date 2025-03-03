#include "gtest/gtest.h"

#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

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

template <typename T>
__global__ void spmat_multi_hardwired_kernel(T*                      vec,
                                             rxmesh::SparseMatrix<T> sparse_mat,
                                             T*                      out,
                                             const int               N)
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

        B_mat(v_id, 0) = iter.size() * 7.4f;
        B_mat(v_id, 1) = iter.size() * 2.6f;
        B_mat(v_id, 2) = iter.size() * 10.3f;

        X_mat(v_id, 0) = coords(v_id, 0) * v_weight;
        X_mat(v_id, 1) = coords(v_id, 1) * v_weight;
        X_mat(v_id, 2) = coords(v_id, 2) * v_weight;

        vec3<T> vi_coord(coords(v_id, 0), coords(v_id, 1), coords(v_id, 2));
        for (uint32_t v = 0; v < iter.size(); ++v) {
            T e_weight           = 1;
            A_mat(v_id, iter[v]) = time_step * e_weight;

            sum_e_weight += e_weight;
        }

        A_mat(v_id, v_id) = v_weight + time_step * sum_e_weight +
                            iter.size() * iter.size() + 1000000;
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, mat_setup);
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

TEST(RXMeshStatic, SparseMatrixSimpleSolve)
{
    // set up a simple AX=B system where A is a sparse matrix, B and C are dense
    // matrix.

    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    const uint32_t threads = 256;
    const uint32_t blocks  = DIVIDE_UP(num_vertices, threads);

    auto                coords = rx.get_input_vertex_coordinates();
    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  X_mat(rx, num_vertices, 3);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3);
    DenseMatrix<float>  ret_mat(rx, num_vertices, 3);

    float time_step = 1.f;

    LaunchBox<threads> launch_box;
    rx.prepare_launch_box(
        {Op::VV}, launch_box, (void*)simple_A_X_B_setup<float, threads>);

    simple_A_X_B_setup<float, threads><<<launch_box.blocks,
                                         launch_box.num_threads,
                                         launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *coords, A_mat, X_mat, B_mat, time_step);

    A_mat.solve(B_mat, X_mat, Solver::CHOL, PermuteMethod::NSTDIS);


    A_mat.multiply(X_mat, ret_mat);


    std::vector<vec3<float>> h_ret_mat(num_vertices);
    CUDA_ERROR(cudaMemcpy(h_ret_mat.data(),
                          ret_mat.data(),
                          num_vertices * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::vector<vec3<float>> h_B_mat(num_vertices);
    CUDA_ERROR(cudaMemcpy(h_B_mat.data(),
                          B_mat.data(),
                          num_vertices * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < num_vertices; ++i) {
        for (uint32_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(h_ret_mat[i][j], h_B_mat[i][j], 1e-3);
        }
    }


    A_mat.release();
    X_mat.release();
    B_mat.release();
    ret_mat.release();
}

TEST(RXMeshStatic, SparseMatrixLowerLevelAPISolve)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    const uint32_t threads = 256;
    const uint32_t blocks  = DIVIDE_UP(num_vertices, threads);

    auto                coords = rx.get_input_vertex_coordinates();
    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  X_mat(rx, num_vertices, 3);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3);
    DenseMatrix<float>  ret_mat(rx, num_vertices, 3);

    float time_step = 1.f;

    LaunchBox<threads> launch_box;
    rx.prepare_launch_box(
        {Op::VV}, launch_box, (void*)simple_A_X_B_setup<float, threads>);

    simple_A_X_B_setup<float, threads><<<launch_box.blocks,
                                         launch_box.num_threads,
                                         launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *coords, A_mat, X_mat, B_mat, time_step);

    // A_mat.solve(B_mat, X_mat, Solver::CHOL, PermuteMethod::NSTDIS);
    A_mat.pre_solve(rx, Solver::CHOL, PermuteMethod::NSTDIS);
    A_mat.solve(B_mat, X_mat);

    A_mat.multiply(X_mat, ret_mat);

    std::vector<vec3<float>> h_ret_mat(num_vertices);
    CUDA_ERROR(cudaMemcpy(h_ret_mat.data(),
                          ret_mat.data(),
                          num_vertices * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::vector<vec3<float>> h_B_mat(num_vertices);
    CUDA_ERROR(cudaMemcpy(h_B_mat.data(),
                          B_mat.data(),
                          num_vertices * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < num_vertices; ++i) {
        for (uint32_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(h_ret_mat[i][j], h_B_mat[i][j], 1e-3);
        }
    }

    A_mat.release();
    X_mat.release();
    B_mat.release();
    ret_mat.release();
}

TEST(RXMeshStatic, SparseMatrixToEigen)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t num_vertices = rx.get_num_vertices();

    const uint32_t threads = 256;
    const uint32_t blocks  = DIVIDE_UP(num_vertices, threads);

    auto                coords = rx.get_input_vertex_coordinates();
    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  X_mat(rx, num_vertices, 3);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3);


    float time_step = 1.f;

    LaunchBox<threads> launch_box;
    rx.prepare_launch_box(
        {Op::VV}, launch_box, (void*)simple_A_X_B_setup<float, threads>);

    simple_A_X_B_setup<float, threads><<<launch_box.blocks,
                                         launch_box.num_threads,
                                         launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *coords, A_mat, X_mat, B_mat, time_step);

    A_mat.solve(B_mat, X_mat, Solver::CHOL, PermuteMethod::NSTDIS);


    DenseMatrix<float> X_copy(rx, num_vertices, 3);
    X_copy.copy_from(X_mat, DEVICE, HOST);

    A_mat.move(DEVICE, HOST);
    B_mat.move(DEVICE, HOST);

    auto A_eigen = A_mat.to_eigen();
    auto X_eigen = X_mat.to_eigen();
    auto B_eigen = B_mat.to_eigen();

    // Note: there is a bug with Eigen if we use the default reordering
    // which is Eigen::AMDOrdering<int>
    // (https://gitlab.com/libeigen/eigen/-/issues/2839)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>,
                          Eigen::UpLoType::Lower,
                          Eigen::COLAMDOrdering<int>>
        eigen_solver;

    eigen_solver.compute(A_eigen);
    X_eigen = eigen_solver.solve(B_eigen);

    for (int i = 0; i < X_copy.rows(); ++i) {
        for (int j = 0; j < X_copy.cols(); ++j) {
            EXPECT_NEAR(X_eigen(i, j), X_copy(i, j), 0.0000001);
        }
    }

    A_mat.release();
    X_mat.release();
    B_mat.release();
    X_copy.release();
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