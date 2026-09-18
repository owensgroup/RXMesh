#include <CLI/CLI.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include "rxmesh/matrix/cg_solver.h"
#ifdef USE_CUDSS
#include "rxmesh/matrix/cudss_cholesky_solver.h"
#endif
#include "rxmesh/matrix/pcg_solver.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/timer.h"

using namespace rxmesh;

struct SolveStats
{
    float    pre_solve_ms          = 0.0f;
    float    time_stepping_ms      = 0.0f;
    uint64_t total_iterations      = 0;
    double   final_solver_residual = 0.0;
    bool     converged             = true;
};

template <typename T>
void assemble_heat_system(const RXMeshStatic&       rx,
                          const VertexAttribute<T>& coordinates,
                          const T                   dt,
                          SparseMatrix<T>&          system,
                          VertexAttribute<T>&       mass)
{
    system.reset(T(0), DEVICE);
    mass.reset(T(0), DEVICE);

    rx.for_each<Op::TV, 256>([=] __device__(const TetHandle&,
                                            const VertexIterator& tv) mutable {
        const vec3<T> x[4] = {coordinates.template to_glm<3>(tv[0]),
                              coordinates.template to_glm<3>(tv[1]),
                              coordinates.template to_glm<3>(tv[2]),
                              coordinates.template to_glm<3>(tv[3])};

        T max_edge = T(0);
        for (int i = 0; i < 4; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                max_edge = std::max(max_edge, glm::length(x[j] - x[i]));
            }
        }

        const vec3<T> e1  = x[1] - x[0];
        const vec3<T> e2  = x[2] - x[0];
        const vec3<T> e3  = x[3] - x[0];
        const T       det = glm::dot(e1, glm::cross(e2, e3));

        // const T det_tolerance = T(32) * std::numeric_limits<T>::epsilon() *
        //                         max_edge * max_edge * max_edge;
        //  if (!isfinite(det) || !isfinite(max_edge) ||
        //      std::abs(det) <= det_tolerance) {
        //      return;
        //  }

        const T       volume      = std::abs(det) / T(6);
        const vec3<T> g1          = glm::cross(e2, e3) / det;
        const vec3<T> g2          = glm::cross(e3, e1) / det;
        const vec3<T> g3          = glm::cross(e1, e2) / det;
        const vec3<T> gradient[4] = {-(g1 + g2 + g3), g1, g2, g3};

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                ::atomicAdd(&system(tv[i], tv[j]),
                            dt * (volume * glm::dot(gradient[i], gradient[j])));
            }
        }

        const T vertex_mass = volume / T(4);
        for (int i = 0; i < 4; ++i) {
            ::atomicAdd(&mass(tv[i]), vertex_mass);
            ::atomicAdd(&system(tv[i], tv[i]), vertex_mass);
        }
    });
}

template <typename T>
void build_rhs(const RXMeshStatic&       rx,
               const VertexAttribute<T>& mass,
               const DenseMatrix<T>&     temperature,
               DenseMatrix<T>&           rhs)
{
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle vh) mutable {
        rhs(vh) = mass(vh) * temperature(vh);
    });
}


template <typename T, typename SolverT>
SolveStats solve_iterative(const RXMeshStatic&       rx,
                           const VertexAttribute<T>& mass,
                           DenseMatrix<T>&           rhs,
                           DenseMatrix<T>&           temperature,
                           const int                 steps,
                           const int                 max_iterations,
                           SolverT&                  solver)
{
    SolveStats stats;
    GPUTimer   timer;
    timer.start();
    for (int step = 0; step < steps; ++step) {
        build_rhs(rx, mass, temperature, rhs);
        solver.pre_solve(rhs, temperature);
        solver.solve(rhs, temperature);
        stats.total_iterations += solver.iter_taken();
        stats.final_solver_residual = solver.final_residual();
        stats.converged             = stats.converged &&
                          solver.iter_taken() < max_iterations &&
                          std::isfinite(stats.final_solver_residual);
    }
    timer.stop();
    stats.time_stepping_ms = timer.elapsed_millis();
    return stats;
}

int main(int argc, char** argv)
{
    using T = rx_coord_t;

    CLI::App app{"Implicit heat diffusion on a tet mesh"};

    std::string mesh_path   = STRINGIFY(INPUT_DIR) "car.msh";
    std::string solver_name = "pcg";
    uint32_t    device_id   = 0;
    uint32_t    source_vid  = 0;
    T           t_factor    = T(1);
    int         steps       = 10;
    int         cg_max_iter = 10000;


    app.add_option("-i,--input", mesh_path, "Input tetrahedral MSH file")
        ->default_val(mesh_path);
    app.add_option("-d,--device_id", device_id, "GPU device ID")
        ->default_val(device_id);
    app.add_option("-s,--source", source_vid, "Initial hot vertex")
        ->default_val(source_vid);
    app.add_option("-t,--t-factor",
                   t_factor,
                   "Time-step multiplier on mean edge length squared")
        ->default_val(t_factor);
    app.add_option("-n,--steps", steps, "Number of implicit time steps")
        ->default_val(steps);
    app.add_option("-l,--solver", solver_name, "Solver: cg, pcg, or cudss")
        ->default_val(solver_name);
    app.add_option(
           "-c,--cg_max_iter", cg_max_iter, "Maximum iterations for CG and PCG")
        ->default_val(cg_max_iter);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    if (solver_name != "cg" && solver_name != "pcg" && solver_name != "cudss") {
        RXMESH_ERROR("Unsupported solver '{}'. Use cg, pcg, or cudss",
                     solver_name);
        return EXIT_FAILURE;
    }
#ifndef USE_CUDSS
    if (solver_name == "cudss") {
        RXMESH_ERROR("The cudss solver requires RX_USE_CUDSS=ON");
        return EXIT_FAILURE;
    }
#endif

    rx_init(device_id);

    RXMESH_INFO("input = {}", mesh_path);
    RXMESH_INFO("device_id = {}", device_id);
    RXMESH_INFO("source = {}", source_vid);
    RXMESH_INFO("t_factor = {}", t_factor);
    RXMESH_INFO("steps = {}", steps);
    RXMESH_INFO("solver = {}", solver_name);
    RXMESH_INFO("cg_max_iter = {}", cg_max_iter);

    RXMeshStatic rx(mesh_path);

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();
    const uint32_t num_tets     = rx.get_num_tets();

    // allocate attributes and sparse matrices
    auto            coordinates = *rx.get_input_vertex_coordinates();
    auto            edge_length = *rx.add_edge_attribute<T>("edge_length", 1);
    auto            mass        = *rx.add_vertex_attribute<T>("mass", 1);
    DenseMatrix<T>  rhs(rx, num_vertices, 1, DEVICE);
    DenseMatrix<T>  temperature(rx, num_vertices, 1, LOCATION_ALL);
    SparseMatrix<T> system(rx, Op::VV);

    if (rx.get_num_components() > 1) {
        RXMESH_WARN("Input mesh has {} components", rx.get_num_components());
    }

    // pick the handle
    VertexHandle source_handle;
    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        if (rx.map_to_global(vh) == source_vid) {
            source_handle = vh;
        }
    });
    temperature.reset(T(0), LOCATION_ALL);
    temperature(source_handle) = T(100);
    temperature.move(HOST, DEVICE);

    // calc mean edge len
    rx.for_each<Op::EV, 256>(
        [=] __device__(const EdgeHandle& eh, const VertexIterator& ev) mutable {
            const vec3<T> a = coordinates.template to_glm<3>(ev[0]);
            const vec3<T> b = coordinates.template to_glm<3>(ev[1]);
            edge_length(eh) = glm::length(a - b);
        });
    EdgeReduceHandle<T> edge_reducer(edge_length);
    const T             edge_length_sum =
        edge_reducer.reduce(edge_length, cub::Sum(), T(0));
    const T mean_edge_length = edge_length_sum / static_cast<T>(num_edges);

    const T dt = t_factor * mean_edge_length * mean_edge_length;

    RXMESH_INFO("#vertices = {}, #edges = {}, #tets = {}",
                num_vertices,
                num_edges,
                num_tets);
    RXMESH_INFO("Mean edge length = {}, dt = {}", mean_edge_length, dt);


    // assemble the system
    GPUTimer assembly_timer;
    assembly_timer.start();
    assemble_heat_system(rx, coordinates, dt, system, mass);
    assembly_timer.stop();
    const float assembly_ms = assembly_timer.elapsed_millis();

    // solve
    const T iterative_abs_tolerance = std::numeric_limits<T>::min();
    const T iterative_rel_tolerance =
        std::is_same_v<T, double> ? T(1e-24) : T(1e-12);

    SolveStats solve_stats;
    if (solver_name == "pcg") {
        PCGSolver<T> solver(system,
                            1,
                            cg_max_iter,
                            iterative_abs_tolerance,
                            iterative_rel_tolerance);
        solve_stats = solve_iterative(
            rx, mass, rhs, temperature, steps, cg_max_iter, solver);
    } else if (solver_name == "cg") {
        CGSolver<T> solver(system,
                           1,
                           cg_max_iter,
                           iterative_abs_tolerance,
                           iterative_rel_tolerance);
        solve_stats = solve_iterative(
            rx, mass, rhs, temperature, steps, cg_max_iter, solver);
#ifdef USE_CUDSS
    } else if (solver_name == "cudss") {
        cuDSSCholeskySolver<SparseMatrix<T>> solver(&system);
        build_rhs(rx, mass, temperature, rhs);

        GPUTimer pre_solve_timer;
        pre_solve_timer.start();
        solver.pre_solve(rx, rhs, temperature);
        pre_solve_timer.stop();
        solve_stats.pre_solve_ms = pre_solve_timer.elapsed_millis();

        GPUTimer time_stepping_timer;
        time_stepping_timer.start();
        for (int step = 0; step < steps; ++step) {
            if (step != 0) {
                build_rhs(rx, mass, temperature, rhs);
            }
            solver.solve(rhs, temperature);
        }
        time_stepping_timer.stop();
        solve_stats.time_stepping_ms = time_stepping_timer.elapsed_millis();
#endif
    }

    if (!solve_stats.converged) {
        RXMESH_WARN("An iterative heat solve did not converge");
    }

    // calc residual
    DenseMatrix<T> residual(rx, num_vertices, 1, DEVICE);
    system.multiply(temperature, residual);
    residual.axpy(rhs, T(-1));
    const double rhs_norm          = static_cast<double>(rhs.norm2());
    const double residual_norm     = static_cast<double>(residual.norm2());
    const double relative_residual = residual_norm / rhs_norm;

    RXMESH_INFO("Assembly took {} ms", assembly_ms);
    if (solver_name == "cudss") {
        RXMESH_INFO("cuDSS analysis and factorization took {} ms",
                    solve_stats.pre_solve_ms);
    } else {
        RXMESH_INFO("Solver took {} iterations and final solver residual = {}",
                    solve_stats.total_iterations,
                    solve_stats.final_solver_residual);
    }
    RXMESH_INFO("{} heat steps took {} ms ({} ms/step)",
                steps,
                solve_stats.time_stepping_ms,
                solve_stats.time_stepping_ms / static_cast<float>(steps));


    RXMESH_INFO("Final ordinary relative residual = {}", relative_residual);


#if USE_POLYSCOPE
    temperature.move(DEVICE, HOST);
    auto final_temperature = *rx.add_vertex_attribute<T>("temperature", 1);
    final_temperature.from_matrix(&temperature);
    rx.get_polyscope_volume_mesh()
        ->addVertexScalarQuantity("temperature", final_temperature)
        ->setEnabled(true);
    polyscope::show();

#endif

    residual.release();
    rhs.release();
    temperature.release();
    system.release();

    return 0;
}
