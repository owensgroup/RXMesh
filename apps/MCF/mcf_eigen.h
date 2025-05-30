#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/rxmesh_static.h"

#include "mcf_kernels.cuh"


#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

template <typename T>
void mcf_eigen_solver(rxmesh::RXMeshStatic& rx,
                       rxmesh::PermuteMethod permute_method)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    uint32_t num_vertices = rx.get_num_vertices();

    auto coords = rx.get_input_vertex_coordinates();

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3);

    DenseMatrix<float> X_mat = *coords->to_matrix();

    RXMESH_INFO("use_uniform_laplace: {}, time_step: {}",
                Arg.use_uniform_laplace,
                Arg.time_step);

    // B set up
    rx.run_kernel<blockThreads>({Op::VV},
                                mcf_B_setup<float, blockThreads>,
                                *coords,
                                B_mat,
                                Arg.use_uniform_laplace);


    // A set up
    rx.run_kernel<blockThreads>({Op::VV},
                                mcf_A_setup<float, blockThreads>,
                                *coords,
                                A_mat,
                                Arg.use_uniform_laplace,
                                Arg.time_step);


    if (Arg.create_AB) {
        A_mat.move(DEVICE, HOST);
        B_mat.move(DEVICE, HOST);
        std::string output_dir = Arg.output_folder + "systems";
        auto        A_mat_copy = A_mat.to_eigen();
        auto        B_mat_copy = B_mat.to_eigen();
        std::filesystem::create_directories(output_dir);
        Eigen::saveMarketDense(
            B_mat_copy,
            output_dir + "/" + extract_file_name(Arg.obj_file_name) + "_B.mtx");
        Eigen::saveMarket(
            A_mat_copy,
            output_dir + "/" + extract_file_name(Arg.obj_file_name) + "_A.mtx");

        std::cout << "\nWrote A and b .mtx files to " << output_dir + "/";
        exit(1);
    }

    Report report("MCF_Eigen");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", std::string("Eigen"));
    report.add_member("application", std::string("MCF"));
    report.add_member("blockThreads", blockThreads);
    report.add_member("PermuteMethod",
                      permute_method_to_string(permute_method));

    RXMESH_INFO("permute_method took {}",
                permute_method_to_string(permute_method));

    float total_time = 0;

    A_mat.move(DEVICE, HOST);
    B_mat.move(DEVICE, HOST);
    auto A_mat_copy = A_mat.to_eigen();
    auto B_mat_copy = B_mat.to_eigen();

    auto                         X_mat_copy = X_mat.to_eigen();
    Eigen::LDLT<Eigen::MatrixXf> ldlt(A_mat_copy);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("LDLT decomposition failed.");
    }

    CPUTimer timer;
    GPUTimer gtimer;
    timer.start();
    gtimer.start();

    X_mat_copy = ldlt.solve(B_mat_copy);
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("solve took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member(
        "solve",
        std::max(timer.elapsed_millis(), gtimer.elapsed_millis()));
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());
    report.add_member("total_time", total_time);

    RXMESH_INFO("total_time {} (ms)", total_time);
    
#if USE_POLYSCOPE
    /*polyscope::registerSurfaceMesh("old mesh",
                                   rx.get_polyscope_mesh()->vertices,
                                   rx.get_polyscope_mesh()->faces);
    rx.get_polyscope_mesh()->updateVertexPositions(*coords);*/
   
    Eigen::MatrixXf                    new_vertices = X_mat_copy;  // n × 3
    std::vector<std::array<double, 3>> vertices_vec;
    vertices_vec.reserve(new_vertices.rows());

     for (int i = 0; i < new_vertices.rows(); ++i) {
        vertices_vec.push_back(
            {new_vertices(i, 0), new_vertices(i, 1), new_vertices(i, 2)});

        /*std::cout << "\n"
                  << new_vertices(i, 0) << " " << new_vertices(i, 1) << " "
                  << new_vertices(i, 2);*/
    }

    polyscope::registerSurfaceMesh("eigensolvedMesh", vertices_vec, rx.get_polyscope_mesh()->faces);
    //polyscope::show();
#endif

    B_mat.release();
    X_mat.release();
    A_mat.release();

    // rx.export_obj("cholesky_mcf_result.obj", *coords);

    report.write(Arg.output_folder + "/rxmesh",
                 "MCF_EIGEN_LDLT_" + extract_file_name(Arg.obj_file_name));
}
