#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/rxmesh_static.h"

#include "mcf_kernels.cuh"

template <typename T, uint32_t blockThreads>
__global__ static void mcf_B_setup(const rxmesh::Context            context,
                                   const rxmesh::VertexAttribute<T> coords,
                                   rxmesh::DenseMatrix<T>           B_mat,
                                   const bool use_uniform_laplace)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        if (use_uniform_laplace) {
            const T valence = static_cast<T>(iter.size());
            B_mat(p_id, 0)  = coords(p_id, 0) * valence;
            B_mat(p_id, 1)  = coords(p_id, 1) * valence;
            B_mat(p_id, 2)  = coords(p_id, 2) * valence;
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

            B_mat(p_id, 0) = coords(p_id, 0) / v_weight;
            B_mat(p_id, 1) = coords(p_id, 1) / v_weight;
            B_mat(p_id, 2) = coords(p_id, 2) / v_weight;
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
__global__ static void mcf_A_setup(
    const rxmesh::Context            context,
    const rxmesh::VertexAttribute<T> coords,
    rxmesh::SparseMatrix<T>          A_mat,
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
void mcf_cusolver_chol(rxmesh::RXMeshStatic& rx,
                       rxmesh::PermuteMethod permute_method)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    uint32_t num_vertices = rx.get_num_vertices();

    auto coords = rx.get_input_vertex_coordinates();

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3);

    std::shared_ptr<DenseMatrix<float>> X_mat = coords->to_matrix();

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


    // A and X set up
    LaunchBox<blockThreads> launch_box_A_X;
    rx.prepare_launch_box({Op::VV},
                          launch_box_A_X,
                          (void*)mcf_A_setup<float, blockThreads>,
                          !Arg.use_uniform_laplace);

    mcf_A_setup<float, blockThreads>
        <<<launch_box_A_X.blocks,
           launch_box_A_X.num_threads,
           launch_box_A_X.smem_bytes_dyn>>>(rx.get_context(),
                                            *coords,
                                            A_mat,
                                            Arg.use_uniform_laplace,
                                            Arg.time_step);


    // To Use LU, we have to move the data to the host
    // A_mat.move(DEVICE, HOST);
    // B_mat.move(DEVICE, HOST);
    // X_mat->move(DEVICE, HOST);
    // A_mat.solve(B_mat, *X_mat, Solver::LU, permute_method);

    // Solving using QR or CHOL
    // A_mat.solve(B_mat, *X_mat, Solver::QR, permute_method);
    // A_mat.solve(B_mat, *X_mat, Solver::CHOL, permute_method);


    // pre-solve
    //A_mat.pre_solve(rx, Solver::CHOL, permute_method);
    // Solve
    //A_mat.solve(B_mat, *X_mat);

    Report report("MCF_Chol");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);

    CPUTimer timer;
    GPUTimer gtimer;

    timer.start();
    gtimer.start();
    A_mat.permute_alloc(permute_method);
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("permute_alloc took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member("permute_alloc", timer.elapsed_millis());

    timer.start();
    gtimer.start();
    A_mat.permute(rx, permute_method);
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("permute took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member("permute", timer.elapsed_millis());


    timer.start();
    gtimer.start();
    A_mat.analyze_pattern(Solver::CHOL);
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("analyze_pattern took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member("analyze_pattern", timer.elapsed_millis());


    timer.start();
    gtimer.start();
    A_mat.post_analyze_alloc(Solver::CHOL);
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("post_analyze_alloc took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member("post_analyze_alloc", timer.elapsed_millis());


    timer.start();
    gtimer.start();
    A_mat.factorize(Solver::CHOL);
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("factorize took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member("factorize", timer.elapsed_millis());


    timer.start();
    gtimer.start();
    A_mat.solve(B_mat, *X_mat);
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("solve took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member("solve", timer.elapsed_millis());
    

    // move the results to the host
    // if we use LU, the data will be on the host and we should not move the
    // device to the host
    X_mat->move(rxmesh::DEVICE, rxmesh::HOST);

    // copy the results to attributes
    coords->from_matrix(X_mat.get());

#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->updateVertexPositions(*coords);
    polyscope::show();
#endif

    B_mat.release();
    X_mat->release();
    A_mat.release();

    report.write(Arg.output_folder + "/rxmesh",
                 "MCF_SpMat_" + extract_file_name(Arg.obj_file_name));
}