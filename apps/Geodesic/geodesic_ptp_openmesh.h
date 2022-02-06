#pragma once

// This implementation is a modified version of https://github.com/larc/gproshan
// Copyright (c) 2018 Luciano Arnaldo Romero Calla, Lizeth Joseline Fuentes
// Perez
// The original implementation uses CHE. Here we use OpenMesh

#include <assert.h>
#include "../common/openmesh_report.h"
#include "../common/openmesh_trimesh.h"
#include "gtest/gtest.h"
#include "rxmesh/util/export_tools.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

inline float compute_toplesets(TriMesh&                     mesh,
                               std::vector<uint32_t>&       sorted_index,
                               std::vector<uint32_t>&       limits,
                               std::vector<uint32_t>&       toplesets,
                               const std::vector<uint32_t>& h_seeds)
{
    limits.clear();
    limits.reserve(mesh.n_vertices() / 2);
    sorted_index.resize(mesh.n_vertices());
    if (h_seeds.size() == 0) {
        return 0;
    }

    rxmesh::CPUTimer timer;
    timer.start();

    toplesets.clear();
    toplesets.resize(mesh.n_vertices(), INVALID32);
    uint32_t level = 0;
    uint32_t p     = 0;
    for (const uint32_t& s : h_seeds) {
        sorted_index[p] = s;
        p++;
        if (toplesets[s] == INVALID32) {
            toplesets[s] = level;
        }
    }

    limits.push_back(0);
    for (uint32_t i = 0; i < p; i++) {
        const uint32_t v = sorted_index[i];
        if (toplesets[v] > level) {
            level++;
            limits.push_back(i);
        }

        TriMesh::VertexIter v_iter = mesh.vertices_begin() + v;
        for (TriMesh::VertexVertexIter vv_iter = mesh.vv_iter(*v_iter);
             vv_iter.is_valid();
             ++vv_iter) {
            int vv = (*vv_iter).idx();
            if (toplesets[vv] == INVALID32) {
                toplesets[vv]   = toplesets[v] + 1;
                sorted_index[p] = vv;
                p++;
            }
        }
    }

    assert(p <= mesh.n_vertices());
    limits.push_back(p);

    if (limits.back() != mesh.n_vertices()) {
        RXMESH_ERROR(
            "compute_toplesets() could not compute toplesets for all "
            "vertices maybe because the input is not manifold or contain "
            "duplicate vertices!");
        exit(EXIT_FAILURE);
    }
    timer.stop();
    return timer.elapsed_millis();
}

template <typename T>
inline T update_step(TriMesh&        mesh,
                     const uint32_t  v0,
                     const uint32_t  v1,
                     const uint32_t  v2,
                     std::vector<T>& geo_distance)
{
    TriMesh::VertexIter v0_it = mesh.vertices_begin() + v0;
    TriMesh::VertexIter v1_it = mesh.vertices_begin() + v1;
    TriMesh::VertexIter v2_it = mesh.vertices_begin() + v2;

    auto X0 = mesh.point(*v1_it) - mesh.point(*v0_it);
    auto X1 = mesh.point(*v2_it) - mesh.point(*v0_it);


    T t[2];
    t[0] = geo_distance[v1];
    t[1] = geo_distance[v2];

    T q[2][2];
    q[0][0] = (X0 | X0);  // X0 dot_product X0
    q[0][1] = (X0 | X1);
    q[1][0] = (X1 | X0);
    q[1][1] = (X1 | X1);


    T det = q[0][0] * q[1][1] - q[0][1] * q[1][0];
    T Q[2][2];
    Q[0][0] = q[1][1] / det;
    Q[0][1] = -q[0][1] / det;
    Q[1][0] = -q[1][0] / det;
    Q[1][1] = q[0][0] / det;

    T delta = t[0] * (Q[0][0] + Q[1][0]) + t[1] * (Q[0][1] + Q[1][1]);
    T dis   = delta * delta -
            (Q[0][0] + Q[0][1] + Q[1][0] + Q[1][1]) *
                (t[0] * t[0] * Q[0][0] + t[0] * t[1] * (Q[1][0] + Q[0][1]) +
                 t[1] * t[1] * Q[1][1] - 1);
    T p = (delta + std::sqrt(dis)) / (Q[0][0] + Q[0][1] + Q[1][0] + Q[1][1]);
    T tp[2];
    tp[0] = t[0] - p;
    tp[1] = t[1] - p;

    // OpenMesh::Vec3f
    decltype(X0) n;
    n[0] = tp[0] * (X0[0] * Q[0][0] + X1[0] * Q[1][0]) +
           tp[1] * (X0[0] * Q[0][1] + X1[0] * Q[1][1]);
    n[1] = tp[0] * (X0[1] * Q[0][0] + X1[1] * Q[1][0]) +
           tp[1] * (X0[1] * Q[0][1] + X1[1] * Q[1][1]);
    n[2] = tp[0] * (X0[2] * Q[0][0] + X1[2] * Q[1][0]) +
           tp[1] * (X0[2] * Q[0][1] + X1[2] * Q[1][1]);

    T cond[2];
    cond[0] = (X0 | n);
    cond[1] = (X1 | n);

    T c[2];
    c[0] = cond[0] * Q[0][0] + cond[1] * Q[0][1];
    c[1] = cond[0] * Q[1][0] + cond[1] * Q[1][1];

    if (t[0] == std::numeric_limits<T>::infinity() ||
        t[1] == std::numeric_limits<T>::infinity() || dis < 0 || c[0] >= 0 ||
        c[1] >= 0) {
        T dp[2];
        dp[0] = geo_distance[v1] + X0.norm();
        dp[1] = geo_distance[v2] + X1.norm();

        p = dp[dp[1] < dp[0]];
    }
    return p;
}

template <typename T>
inline float toplesets_propagation(TriMesh&                     mesh,
                                   const std::vector<uint32_t>& h_seeds,
                                   const std::vector<uint32_t>& limits,
                                   const std::vector<uint32_t>& sorted_index,
                                   std::vector<T>&              geo_distance,
                                   uint32_t&                    iter)
{
    // second buffer for geodesic distance
    std::vector<T>  geo_distance_2(geo_distance);
    std::vector<T>* double_buffer[2] = {&geo_distance, &geo_distance_2};
    // error buffer
    std::vector<T> error(mesh.n_vertices(), 0);

    rxmesh::CPUTimer timer;
    timer.start();

    // source distance
    for (auto v : h_seeds) {
        geo_distance[v]   = 0;
        geo_distance_2[v] = 0;
    }

    uint32_t d = 0;
    uint32_t i(1), j(2);
    iter              = 0;
    uint32_t max_iter = 2 * limits.size();

    while (i < j && iter < max_iter) {
        iter++;
        if (i < (j / 2)) {
            i = j / 2;
        }

        const uint32_t start  = limits[i];
        const uint32_t end    = limits[j];
        const uint32_t n_cond = limits[i + 1] - start;

        for (uint32_t vi = start; vi < end; vi++) {
            const uint32_t      v      = sorted_index[vi];
            TriMesh::VertexIter v_iter = mesh.vertices_begin() + v;

            (*double_buffer[!d])[v] = (*double_buffer[d])[v];


            // The last vertex in v one ring
            TriMesh::VertexVertexIter p_iter = mesh.vv_iter(*v_iter);
            --p_iter;
            assert(p_iter.is_valid());

            // iterate over one-ring
            for (TriMesh::VertexVertexIter vv_iter = mesh.vv_iter(*v_iter);
                 vv_iter.is_valid();
                 ++vv_iter) {

                // current vv
                uint32_t vv_id = (*vv_iter).idx();

                // previous vertex
                uint32_t p_id = (*p_iter).idx();

                // working on triangle v,vv_id, p_id
                T dist = update_step(mesh, v, p_id, vv_id, *double_buffer[d]);
                if (dist < (*double_buffer[!d])[v]) {
                    (*double_buffer[!d])[v] = dist;
                }


                // advance previous vv
                p_iter++;
                assert(p_iter == vv_iter);
            }
        }
        // calc error
        for (uint32_t vi = start; vi < start + n_cond; vi++) {
            const uint32_t v = sorted_index[vi];
            error[vi] =
                std::abs((*double_buffer[!d])[v] - (*double_buffer[d])[v]) /
                (*double_buffer[d])[v];
        }

        uint32_t count = 0;
        for (uint32_t vi = start; vi < start + n_cond; vi++) {
            count += error[vi] < 1e-3;
        }

        if (n_cond == count) {
            i++;
        }
        if (j < limits.size() - 1) {
            j++;
        }
        d = !d;
    }

    timer.stop();

    // copy most updated results (if needed)
    if (&geo_distance != double_buffer[!d]) {
        for (size_t i = 0; i < geo_distance.size(); ++i) {
            geo_distance[i] = geo_distance_2[i];
        }
    }

    return timer.elapsed_millis();
}

template <typename T>
void geodesic_ptp_openmesh(const std::vector<uint32_t>& h_seeds,
                           std::vector<uint32_t>&       sorted_index,
                           std::vector<uint32_t>&       limits,
                           std::vector<uint32_t>&       toplesets)
{
    TriMesh input_mesh;
    ASSERT_TRUE(OpenMesh::IO::read_mesh(input_mesh, Arg.obj_file_name));

    // Report
    OpenMeshReport report("Geodesic_OpenMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.system();
    report.model_data(Arg.obj_file_name, input_mesh);
    report.add_member("seeds", h_seeds);
    std::string method = "OpenMeshSingleCore";
    report.add_member("method", method);


    std::vector<T> geo_distance(input_mesh.n_vertices(),
                                std::numeric_limits<T>::infinity());

    // sorted indices for toplesets
    sorted_index.clear();

    // toplesets limits
    limits.clear();

    // compute toplesets
    float compute_toplesets_time =
        compute_toplesets(input_mesh, sorted_index, limits, toplesets, h_seeds);

    RXMESH_TRACE("OpenMesh: Computing toplesets took {} (ms)",
                 compute_toplesets_time);

    report.add_member("compute_toplesets_time", compute_toplesets_time);

    // compute geodesic distance
    uint32_t iter            = 0;
    float    processing_time = toplesets_propagation(
        input_mesh, h_seeds, limits, sorted_index, geo_distance, iter);
    RXMESH_TRACE("geodesic_ptp_openmesh() took {} (ms)", processing_time);

    // Finalize report
    report.add_member("num_iter_taken", iter);
    rxmesh::TestData td;
    td.test_name   = "Geodesic";
    td.num_threads = 1;
    td.time_ms.push_back(processing_time);
    td.passed.push_back(true);
    report.add_test(td);
    report.write(
        Arg.output_folder + "/openmesh",
        "Geodesic_OpenMesh" + rxmesh::extract_file_name(Arg.obj_file_name));
}