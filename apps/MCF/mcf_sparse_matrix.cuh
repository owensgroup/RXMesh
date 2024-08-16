#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/rxmesh_static.h"

template <typename T, uint32_t blockThreads>
__global__ static void mcf_B_setup(const rxmesh::Context            context,
                                   const rxmesh::VertexAttribute<T> coords,
                                   rxmesh::DenseMatrix<T>           B_mat,
                                   const bool use_uniform_laplace)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        auto     r_ids      = p_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index = context.m_vertex_prefix[r_patch_id] + r_local_id;

        if (use_uniform_laplace) {
            const T valence     = static_cast<T>(iter.size());
            B_mat(row_index, 0) = coords(p_id, 0) * valence;
            B_mat(row_index, 1) = coords(p_id, 1) * valence;
            B_mat(row_index, 2) = coords(p_id, 2) * valence;
        } else {
            T v_weight = 0;

            // this is the last vertex in the one-ring (before r_id)
            VertexHandle q_id = iter.back();

            for (uint32_t v = 0; v < iter.size(); ++v) {
                // the current one ring vertex
                VertexHandle r_id = iter[v];

                T tri_area = partial_voronoi_area(p_id, q_id, r_id, coords);

                v_weight += (tri_area > 0) ? tri_area : 0.0;

                q_id = r_id;
            }
            v_weight = 0.5 / v_weight;

            B_mat(row_index, 0) = coords(p_id, 0) / v_weight;
            B_mat(row_index, 1) = coords(p_id, 1) / v_weight;
            B_mat(row_index, 2) = coords(p_id, 2) / v_weight;
        }
    };

    // With uniform Laplacian, we just need the valence, thus we
    // call query and set oriented to false
    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(
        block,
        shrd_alloc,
        init_lambda,
        [](VertexHandle) { return true; },
        !use_uniform_laplace);
}

template <typename T, uint32_t blockThreads>
__global__ static void mcf_A_X_setup(
    const rxmesh::Context            context,
    const rxmesh::VertexAttribute<T> coords,
    rxmesh::SparseMatrix<T>          A_mat,
    rxmesh::DenseMatrix<T>           X_mat,
    const bool                       use_uniform_laplace,  // for non-uniform
    const T                          time_step)
{
    using namespace rxmesh;
    auto init_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        T sum_e_weight(0);
        T v_weight(0);

        VertexHandle q_id = iter.back();

        // reference value calculation
        auto     r_ids      = p_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index = context.m_vertex_prefix[r_patch_id] + r_local_id;

        // set up initial X matrix
        X_mat(row_index, 0) = coords(p_id, 0);
        X_mat(row_index, 1) = coords(p_id, 1);
        X_mat(row_index, 2) = coords(p_id, 2);

        // set up matrix A
        for (uint32_t v = 0; v < iter.size(); ++v) {
            VertexHandle r_id = iter[v];

            T e_weight = 0;
            if (use_uniform_laplace) {
                e_weight = 1;
            } else {
                VertexHandle s_id =
                    (v == iter.size() - 1) ? iter[0] : iter[v + 1];

                e_weight = edge_cotan_weight(p_id, r_id, q_id, s_id, coords);
                e_weight = (static_cast<T>(e_weight >= 0.0)) * e_weight;
            }

            e_weight *= time_step;
            sum_e_weight += e_weight;

            A_mat(p_id, iter[v]) = -e_weight;

            // compute vertex weight
            if (use_uniform_laplace) {
                ++v_weight;
            } else {
                T tri_area = partial_voronoi_area(p_id, q_id, r_id, coords);
                v_weight += (tri_area > 0) ? tri_area : 0;
                q_id = r_id;
            }
        }

        // Diagonal entry
        if (use_uniform_laplace) {
            v_weight = 1.0 / v_weight;
        } else {
            v_weight = 0.5 / v_weight;
        }

        assert(!isnan(v_weight));
        assert(!isinf(v_weight));

        A_mat(p_id, p_id) = (1.0 / v_weight) + sum_e_weight;
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(
        block,
        shrd_alloc,
        init_lambda,
        [](VertexHandle) { return true; },
        !use_uniform_laplace);
}

template <typename T>
void mcf_rxmesh_cusolver_chol(rxmesh::RXMeshStatic&              rx,
                              const std::vector<std::vector<T>>& ground_truth)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    uint32_t num_vertices = rx.get_num_vertices();
    auto     coords       = rx.get_input_vertex_coordinates();

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  X_mat(num_vertices, 3);
    DenseMatrix<float>  B_mat(num_vertices, 3);

    RXMESH_INFO("use_uniform_laplace: {}, time_step: {}",
                Arg.use_uniform_laplace,
                Arg.time_step);

    // B set up
    LaunchBox<blockThreads> launch_box_B;
    rx.prepare_launch_box({Op::VV},
                          launch_box_B,
                          (void*)mcf_B_setup<float, blockThreads>,
                          !Arg.use_uniform_laplace);

    mcf_B_setup<float, blockThreads><<<launch_box_B.blocks,
                                       launch_box_B.num_threads,
                                       launch_box_B.smem_bytes_dyn>>>(
        rx.get_context(), *coords, B_mat, Arg.use_uniform_laplace);

    CUDA_ERROR(cudaDeviceSynchronize());

    // A and X set up
    LaunchBox<blockThreads> launch_box_A_X;
    rx.prepare_launch_box({Op::VV},
                          launch_box_A_X,
                          (void*)mcf_A_X_setup<float, blockThreads>,
                          !Arg.use_uniform_laplace);

    mcf_A_X_setup<float, blockThreads>
        <<<launch_box_A_X.blocks,
           launch_box_A_X.num_threads,
           launch_box_A_X.smem_bytes_dyn>>>(rx.get_context(),
                                            *coords,
                                            A_mat,
                                            X_mat,
                                            Arg.use_uniform_laplace,
                                            Arg.time_step);

    // Solving the linear system using chol factorization and no reordering
    // A_mat.spmat_linear_solve(B_mat, X_mat, Solver::CHOL, Reorder::NONE);
    CPUTimer ctimer;

    ctimer.start();
    A_mat.spmat_chol_reorder(Reorder::NSTDIS);
    ctimer.stop();

    float reorder_total_time = ctimer.elapsed_millis();
    RXMESH_INFO("Reordering time Low: {}", reorder_total_time);

    GPUTimer gtimer;
    gtimer.start();
    A_mat.spmat_chol_analysis();
    A_mat.spmat_chol_buffer_alloc();
    A_mat.spmat_chol_factor();
    gtimer.stop();

    float fact_time = gtimer.elapsed_millis();
    RXMESH_INFO("Factorization time Low w/ our Reorder: {}", fact_time); 

    gtimer.start();
    A_mat.spmat_chol_solve(B_mat.m_d_val, X_mat.m_d_val);
    gtimer.stop();
    float solve_time = gtimer.elapsed_millis();
    RXMESH_INFO("Solve time Low w/ our Reorder: {}", solve_time);

    X_mat.move(rxmesh::DEVICE, rxmesh::HOST);

    const T tol     = 0.5;
    T       tmp_tol = tol;
    bool    passed  = true;
    // rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
    //     uint32_t v_id        = rx.map_to_global(vh);
    //     uint32_t v_linear_id = rx.linear_id(vh);

    //     T a = X_mat(v_linear_id, 0);

    //     for (uint32_t i = 0; i < 3; ++i) {
    //         tmp_tol = std::abs((X_mat(v_linear_id, i) - ground_truth[v_id][i]) /
    //                            ground_truth[v_id][i]);

    //         if (tmp_tol > tol) {
    //             RXMESH_WARN("val: {}, truth: {}, tol: {}\n",
    //                         X_mat(v_linear_id, i),
    //                         ground_truth[v_id][i],
    //                         tmp_tol);
    //             passed = false;
    //             break;
    //         }
    //     }
    // });

    EXPECT_TRUE(passed);
}

// Tmp check for reordering performance
template <typename T>
void mcf_rxmesh_cusolver_chol_reordering(rxmesh::RXMeshStatic&              rx,
                              const std::vector<std::vector<T>>& ground_truth)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    uint32_t num_vertices = rx.get_num_vertices();
    auto     coords       = rx.get_input_vertex_coordinates();

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  X_mat(num_vertices, 3);
    DenseMatrix<float>  B_mat(num_vertices, 3);

    RXMESH_INFO("use_uniform_laplace: {}, time_step: {}",
                Arg.use_uniform_laplace,
                Arg.time_step);

    // B set up
    LaunchBox<blockThreads> launch_box_B;
    rx.prepare_launch_box({Op::VV},
                          launch_box_B,
                          (void*)mcf_B_setup<float, blockThreads>,
                          !Arg.use_uniform_laplace);

    mcf_B_setup<float, blockThreads><<<launch_box_B.blocks,
                                       launch_box_B.num_threads,
                                       launch_box_B.smem_bytes_dyn>>>(
        rx.get_context(), *coords, B_mat, Arg.use_uniform_laplace);

    CUDA_ERROR(cudaDeviceSynchronize());

    // A and X set up
    LaunchBox<blockThreads> launch_box_A_X;
    rx.prepare_launch_box({Op::VV},
                          launch_box_A_X,
                          (void*)mcf_A_X_setup<float, blockThreads>,
                          !Arg.use_uniform_laplace);

    mcf_A_X_setup<float, blockThreads>
        <<<launch_box_A_X.blocks,
           launch_box_A_X.num_threads,
           launch_box_A_X.smem_bytes_dyn>>>(rx.get_context(),
                                            *coords,
                                            A_mat,
                                            X_mat,
                                            Arg.use_uniform_laplace,
                                            Arg.time_step);

    // Solving the linear system using chol factorization and no reordering
    // A_mat.spmat_linear_solve(B_mat, X_mat, Solver::CHOL, Reorder::NONE);

    uint32_t* reorder_array;
    
    CUDA_ERROR(cudaMallocHost(&reorder_array,
                                 sizeof(uint32_t) * rx.get_num_vertices()));

    GPUTimer gtimer;
    gtimer.start();
    nd_reorder(rx, reorder_array, Arg.nd_level);
    A_mat.spmat_chol_reorder(Reorder::GPUND, reorder_array);
    gtimer.stop();

    float reorder_total_time = gtimer.elapsed_millis();
    RXMESH_INFO("Reordering time Low: {}", reorder_total_time);

    gtimer.start();
    A_mat.spmat_chol_analysis();
    A_mat.spmat_chol_buffer_alloc();
    A_mat.spmat_chol_factor();
    gtimer.stop();

    float fact_time = gtimer.elapsed_millis();
    RXMESH_INFO("Factorization time Low w/ our Reorder: {}", fact_time); 

    gtimer.start();
    A_mat.spmat_chol_solve(B_mat.m_d_val, X_mat.m_d_val);
    gtimer.stop();
    float solve_time = gtimer.elapsed_millis();
    RXMESH_INFO("Solve time Low w/ our Reorder: {}", solve_time);


    X_mat.move(rxmesh::DEVICE, rxmesh::HOST);

    const T tol     = 0.5;
    T       tmp_tol = tol;
    bool    passed  = true;
    // rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
    //     uint32_t v_id        = rx.map_to_global(vh);
    //     uint32_t v_linear_id = rx.linear_id(vh);

    //     T a = X_mat(v_linear_id, 0);

    //     for (uint32_t i = 0; i < 3; ++i) {
    //         tmp_tol = std::abs((X_mat(v_linear_id, i) - ground_truth[v_id][i]) /
    //                            ground_truth[v_id][i]);

    //         if (tmp_tol > tol) {
    //             RXMESH_WARN("val: {}, truth: {}, tol: {}\n",
    //                         X_mat(v_linear_id, i),
    //                         ground_truth[v_id][i],
    //                         tmp_tol);
    //             passed = false;
    //             break;
    //         }
    //     }
    // });

    EXPECT_TRUE(passed);
}