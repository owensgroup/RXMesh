#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/rxmesh_static.h"


template <typename T, uint32_t blockThreads>
__global__ static void mcf_A_X_B_setup(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> coords,  // for non-uniform
    rxmesh::SparseMatrix<T>    A_mat,
    rxmesh::DenseMatrix<T>     X_mat,
    rxmesh::DenseMatrix<T>     B_mat,
    const bool                 use_uniform_laplace,  // for non-uniform
    const T                    time_step)
{
    using namespace rxmesh;
    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        T sum_e_weight(0);

        T v_weight = iter.size();

        // reference value calculation
        auto     r_ids      = v_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index =
            A_mat.m_context.m_vertex_prefix[r_patch_id] + r_local_id;

        // set up B matrix
        B_mat(row_index, 0) = coords(v_id, 0) * v_weight;
        B_mat(row_index, 1) = coords(v_id, 1) * v_weight;
        B_mat(row_index, 2) = coords(v_id, 2) * v_weight;

        // set up initial X matrix
        X_mat(row_index, 0) = coords(v_id, 0);
        X_mat(row_index, 1) = coords(v_id, 1);
        X_mat(row_index, 2) = coords(v_id, 2);

        Vector<3, float> vi_coord(
            coords(v_id, 0), coords(v_id, 1), coords(v_id, 2));
        for (uint32_t v = 0; v < iter.size(); ++v) {
            T e_weight = 1;
            e_weight *= time_step;
            sum_e_weight += e_weight;

            A_mat(v_id, iter[v]) = -e_weight;
        }

        A_mat(v_id, v_id) = v_weight + sum_e_weight;
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

    printf("use_uniform_laplace: %d, time_step: %f\n",
           Arg.use_uniform_laplace,
           Arg.time_step);

    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box(
        {Op::VV}, launch_box, (void*)mcf_A_X_B_setup<float, blockThreads>);

    mcf_A_X_B_setup<float, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rxmesh.get_context(),
                                        *coords,
                                        A_mat,
                                        X_mat,
                                        B_mat,
                                        true,
                                        Arg.time_step);

    A_mat.spmat_linear_solve(B_mat, X_mat, Solver::CHOL, Reorder::NONE);

    auto smooth_X =
        rxmesh.add_vertex_attribute<T>("smooth_X", 3, rxmesh::LOCATION_ALL);
    auto truth_X =
        rxmesh.add_vertex_attribute<T>("truth_X", 3, rxmesh::LOCATION_ALL);

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
    truth_X->move(rxmesh::DEVICE, rxmesh::HOST);

    const T tol     = 0.001;
    T       tmp_tol = tol;
    bool    passed  = true;
    rxmesh.for_each_vertex(HOST, [&](const VertexHandle vh) {
        uint32_t v_id = rxmesh.map_to_global(vh);

        for (uint32_t i = 0; i < 3; ++i) {
            (*truth_X)(vh, i) = ground_truth[v_id][i];
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

    auto ps_mesh = rxmesh.get_polyscope_mesh();
    ps_mesh->addVertexColorQuantity("smooth_x", *smooth_X);
    ps_mesh->addVertexColorQuantity("smooth_om", *truth_X);
    polyscope::show();

    EXPECT_TRUE(passed);
}