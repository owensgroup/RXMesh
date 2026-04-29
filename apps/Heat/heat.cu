// Crane, Weischedel, Wardetzky.
// "The Heat Method for Distance Computation"
// https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/
// Reference:
// https://github.com/nmwsharp/geometry-central/blob/master/src/surface/heat_method_distance.cpp

#include <CLI/CLI.hpp>
#include <algorithm>
#include <limits>

#include "rxmesh/geometry_util.cuh"
#include "rxmesh/matrix/cudss_cholesky_solver.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/timer.h"

using namespace rxmesh;

int main(int argc, char** argv)
{
    using T = rx_coord_t;

    CLI::App app{"Heat geodesics (Crane et al. 2017)"};

    std::string mesh_path  = STRINGIFY(INPUT_DIR) "sphere3.obj";
    uint32_t    device_id  = 0;
    uint32_t    source_vid = 0;
    T           t_factor   = 1.0f;

    app.add_option("-i,--input", mesh_path, "Input OBJ mesh file")
        ->default_val(mesh_path);
    app.add_option("-d,--device_id", device_id, "GPU device ID")
        ->default_val(device_id);
    app.add_option("-s,--source", source_vid, "Source vertex")
        ->default_val(source_vid);
    app.add_option("-t,--t-factor",
                   t_factor,
                   "Heat time-step multiplier on h^2 (default 1)")
        ->default_val(1.0f);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(device_id);

    RXMeshStatic rx(mesh_path);
    
    auto coords = *rx.get_input_vertex_coordinates();

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();

    // Find the VerteHandle of source_vid
    VertexHandle source_handle;
    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        if (rx.map_to_global(vh) == source_vid) {
            source_handle = vh;
        }
    });
    if (!source_handle.is_valid()) {
        RXMESH_ERROR(
            "Could not find source vertex with index {} in the input mesh "
            "(#vertices = {}).",
            source_vid,
            num_vertices);
        return EXIT_FAILURE;
    }

    if (!rx.is_closed()) {
        RXMESH_ERROR("Input mesh should be closed/watertight");
        return EXIT_FAILURE;
    }

    if (rx.get_num_components() > 1) {
        RXMESH_WARN("Input mesh has {} components!", rx.get_num_components());
    }

    // 0) Allocate all matrices
    // L is the cotan Laplacian
    // A_heat shares the same VV pattern and stores M - t L
    SparseMatrix<T> L(rx);
    SparseMatrix<T> A_heat(rx);

    auto mass  = *rx.add_vertex_attribute<T>("mass", 1);
    auto X     = *rx.add_face_attribute<T>("X", 3);
    auto dist  = *rx.add_vertex_attribute<T>("geodesic", 1);
    auto e_len = *rx.add_edge_attribute<T>("e_len", 1);

    DenseMatrix<T> rhs(rx, num_vertices, 1, DEVICE);
    // using single matrix to store phi or u
    DenseMatrix<T> phi_or_u(rx, num_vertices, 1, LOCATION_ALL);

    EdgeReduceHandle<T> e_reducer(e_len);

    GPUTimer timer;
    timer.start();

    // 1) compute avg edge length h and t
    rx.for_each<Op::EV, 256>(
        [=] __device__(const EdgeHandle& eh, const VertexIterator& iter) {
            const vec3<T> a = coords.to_glm<3>(iter[0]);
            const vec3<T> b = coords.to_glm<3>(iter[1]);
            e_len(eh)       = glm::length(a - b);
        });

    T h_sum = e_reducer.reduce(e_len, cub::Sum(), 0.0);

    const T h = h_sum / static_cast<T>(num_edges);
    const T t = t_factor * h * h;

    RXMESH_INFO("Avg edge length h = {}, t = {}", h, t);

    // 2) Build cotangent Laplacian L and lumped mass
    rx.for_each<Op::VV, 256>(
        [=] __device__(const VertexHandle&   p,
                       const VertexIterator& iter) mutable {
            const vec3<T> P = coords.to_glm<3>(p);

            T sum_w  = 0.f;
            T v_area = 0.f;

            VertexHandle q = iter.back();
            for (uint32_t v = 0; v < iter.size(); ++v) {
                VertexHandle r = iter[v];
                VertexHandle s = (v + 1 == iter.size()) ? iter[0] : iter[v + 1];

                const vec3<T> R = coords.to_glm<3>(r);
                const vec3<T> Q = coords.to_glm<3>(q);
                const vec3<T> S = coords.to_glm<3>(s);

                T w = edge_cotan_weight(P, R, Q, S);
                w   = (w >= 0.f) ? w : 0.f;
                sum_w += w;
                L(p, r) = -w;

                const T a = tri_area(P, Q, R);
                v_area += (a > 0.f) ? (a / T(3)) : 0.f;
                q = r;
            }
            L(p, p)    = sum_w + 1e-6f;
            mass(p, 0) = v_area;
        },
        /*oriented=*/true);


    // 3) Assemble A_heat = M - t L (same sparsity pattern as L)
    // and set rhs (0 everywhere, 1 at the source vertex)
    rx.for_each<Op::VV, 256>(
        [=] __device__(const VertexHandle&   p,
                       const VertexIterator& iter) mutable {
            for (uint32_t v = 0; v < iter.size(); ++v) {
                A_heat(p, iter[v]) = t * L(p, iter[v]);
            }
            A_heat(p, p) = mass(p, 0) + t * L(p, p);

            rhs(p, 0) = T(0);
            if (p == source_handle) {
                rhs(p, 0) = T(1);
            }
        });


    // 4) solve A_heat*u = rhs
    cuDSSCholeskySolver<SparseMatrix<T>> heat_solver(&A_heat);
    heat_solver.pre_solve(rx, rhs, phi_or_u);
    heat_solver.solve(rhs, phi_or_u);


    // 5) Per-face X = -grad(u) / |grad(u)|.
    //  For a triangle (p0, p1, p2) with face normal N:
    //  grad(u) = (1 / (2A)) * sum_i u_i * (N_hat x e_i)
    //  where e_i is the edge OPPOSITE vertex i, oriented CCW
    rx.for_each<Op::FV, 256>([=] __device__(
                                 const FaceHandle&     f,
                                 const VertexIterator& iter) mutable {
        const vec3<T> p0 = coords.to_glm<3>(iter[0]);
        const vec3<T> p1 = coords.to_glm<3>(iter[1]);
        const vec3<T> p2 = coords.to_glm<3>(iter[2]);

        const T u0 = phi_or_u(iter[0], 0);
        const T u1 = phi_or_u(iter[1], 0);
        const T u2 = phi_or_u(iter[2], 0);

        const vec3<T> N = glm::cross(p1 - p0, p2 - p0);

        const T two_a = glm::length(N);

        if (two_a <= 0.f) {
            X(f, 0) = 0.f;
            X(f, 1) = 0.f;
            X(f, 2) = 0.f;
        } else {

            const vec3<T> Nh = N / two_a;

            const vec3<T> g =
                (u0 * glm::cross(Nh, p2 - p1) + u1 * glm::cross(Nh, p0 - p2) +
                 u2 * glm::cross(Nh, p1 - p0)) /
                two_a;

            const T gn = glm::length(g);

            const vec3<T> Xi = (gn > 1e-20f) ? (g / gn) : vec3<T>(0.f);

            X(f, 0) = Xi.x;
            X(f, 1) = Xi.y;
            X(f, 2) = Xi.z;
        }
    });


    // 6) Divergence
    //    For each face f and each vertex i of f, contribute to div(X)_i:
    //       0.5 * ( cot(angle at k) * dot(p_j - p_i, X_f)
    //             + cot(angle at j) * dot(p_k - p_i, X_f) )
    //    where (i, j, k) walks the triangle CCW
    rhs.reset(0.f, DEVICE);

    rx.for_each<Op::FV, 256>([=] __device__(
                                 const FaceHandle&     f,
                                 const VertexIterator& iter) mutable {
        const vec3<T> p[3] = {coords.to_glm<3>(iter[0]),
                              coords.to_glm<3>(iter[1]),
                              coords.to_glm<3>(iter[2])};
        const vec3<T> Xi(X(f, 0), X(f, 1), X(f, 2));

        auto cot_at = [](const vec3<T>& a, const vec3<T>& b) -> T {
            const T denom = glm::length(glm::cross(a, b));
            return (denom > 1e-20f) ? (glm::dot(a, b) / denom) : 0.f;
        };

        for (int i = 0; i < 3; ++i) {
            const int     j  = (i + 1) % 3;
            const int     k  = (i + 2) % 3;
            const vec3<T> e1 = p[j] - p[i];
            const vec3<T> e2 = p[k] - p[i];

            const T cot_k = cot_at(p[i] - p[k], p[j] - p[k]);
            const T cot_j = cot_at(p[i] - p[j], p[k] - p[j]);

            const T contrib =
                0.5f * (cot_k * glm::dot(e1, Xi) + cot_j * glm::dot(e2, Xi));

            ::atomicAdd(&rhs(iter[i], 0), contrib);
        }
    });


    // 7) Solve L phi = div(X)
    cuDSSCholeskySolver<SparseMatrix<T>> poisson_solver(&L);
    phi_or_u.reset(0, DEVICE);
    poisson_solver.pre_solve(rx, rhs, phi_or_u);
    poisson_solver.solve(rhs, phi_or_u);


    // Distance = phi - phi(source)
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle vh) mutable {
        dist(vh, 0) = phi_or_u(vh, 0) - phi_or_u(source_handle, 0);
    });
    dist.move(DEVICE, HOST);   

    timer.stop();

    RXMESH_INFO("Heat Geodesics took {} (ms)", timer.elapsed_millis());
#if USE_POLYSCOPE

    rx.get_polyscope_mesh()
        ->addVertexScalarQuantity("geodesic", dist)
        ->setEnabled(true);
    polyscope::show();
#endif

    return 0;
}