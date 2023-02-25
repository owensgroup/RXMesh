#pragma once
#include "mcf_util.h"
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

        uint32_t row_index =
            A_mat.m_context.m_vertex_prefix[r_patch_id] + r_local_id;

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

template <typename T, uint32_t blockThreads>
__global__ static void update_smooth_result(const rxmesh::Context      context,
                                            rxmesh::VertexAttribute<T> smooth_X,
                                            rxmesh::SparseMatrix<T>    A_mat,
                                            rxmesh::DenseMatrix<T>     X_mat)
{
    using namespace rxmesh;
    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     r_ids      = v_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index =
            A_mat.m_context.m_vertex_prefix[r_patch_id] + r_local_id;

        // printf("check: %f\n", X_mat(row_index, 0));

        smooth_X(v_id, 0) = X_mat(row_index, 0);
        smooth_X(v_id, 1) = X_mat(row_index, 1);
        smooth_X(v_id, 2) = X_mat(row_index, 2);

        // printf("s_check: %f\n", smooth_X(v_id, 0));
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, init_lambda);
}

template <typename T>
void mcf_rxmesh_solver(rxmesh::RXMeshStatic&              rxmesh,
                       const std::vector<std::vector<T>>& ground_truth)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    uint32_t num_vertices = rxmesh.get_num_vertices();
    auto     coords       = rxmesh.get_input_vertex_coordinates();

    SparseMatrix<float> A_mat(rxmesh);
    DenseMatrix<float>  X_mat(num_vertices, 3);
    DenseMatrix<float>  B_mat(num_vertices, 3);

    // B set up
    LaunchBox<blockThreads> launch_box_B;
    rxmesh.prepare_launch_box(
        {Op::VV}, launch_box_B, (void*)mcf_B_setup<float, blockThreads>);

    mcf_B_setup<float, blockThreads><<<launch_box_B.blocks,
                                       launch_box_B.num_threads,
                                       launch_box_B.smem_bytes_dyn>>>(
        rxmesh.get_context(), *coords, B_mat, Arg.use_uniform_laplace);

    printf("use_uniform_laplace: %d, time_step: %f\n",
           Arg.use_uniform_laplace,
           Arg.time_step);

    // A and X set up
    LaunchBox<blockThreads> launch_box_A_X;
    rxmesh.prepare_launch_box(
        {Op::VV}, launch_box_A_X, (void*)mcf_A_X_setup<float, blockThreads>);

    mcf_A_X_setup<float, blockThreads>
        <<<launch_box_A_X.blocks,
           launch_box_A_X.num_threads,
           launch_box_A_X.smem_bytes_dyn>>>(rxmesh.get_context(),
                                            *coords,
                                            A_mat,
                                            X_mat,
                                            Arg.use_uniform_laplace,
                                            Arg.time_step);

    printf("use_uniform_laplace: %d, time_step: %f\n",
           Arg.use_uniform_laplace,
           Arg.time_step);

    // Solving the linear system
    A_mat.spmat_linear_solve(B_mat, X_mat, Solver::CHOL, Reorder::NONE);

    auto smooth_X =
        rxmesh.add_vertex_attribute<T>("smooth_X", 3, rxmesh::LOCATION_ALL);

    LaunchBox<blockThreads> launch_box_smooth;
    rxmesh.prepare_launch_box({Op::VV},
                              launch_box_smooth,
                              (void*)update_smooth_result<float, blockThreads>);

    update_smooth_result<float, blockThreads>
        <<<launch_box_smooth.blocks,
           launch_box_smooth.num_threads,
           launch_box_smooth.smem_bytes_dyn>>>(
            rxmesh.get_context(), *smooth_X, A_mat, X_mat);
    smooth_X->move(rxmesh::DEVICE, rxmesh::HOST);

    printf("use_uniform_laplace: %d, time_step: %f\n",
           Arg.use_uniform_laplace,
           Arg.time_step);

    rxmesh.export_obj("mcf_rxmesh_solver.obj", *smooth_X);

    const T tol     = 0.001;
    T       tmp_tol = tol;
    bool    passed  = true;
    rxmesh.for_each_vertex(HOST, [&](const VertexHandle vh) {
        uint32_t v_id = rxmesh.map_to_global(vh);

        for (uint32_t i = 0; i < 3; ++i) {
            tmp_tol = std::abs(((*smooth_X)(vh, i) - ground_truth[v_id][i]) /
                               ground_truth[v_id][i]);
            if (tmp_tol > tol) {
                printf("val: %f, truth: %f, tol: %f\n",
                       (*smooth_X)(vh, i),
                       ground_truth[v_id][i],
                       tmp_tol);
            }

            if (std::abs(((*smooth_X)(vh, i) - ground_truth[v_id][i]) /
                         ground_truth[v_id][i]) > tol) {
                passed = false;
                break;
            }
        }
    });

    EXPECT_TRUE(passed);
}