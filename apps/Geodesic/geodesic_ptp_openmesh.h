#pragma once

// This implementation is a modified version of https://github.com/larc/gproshan
// Copyright (c) 2018 Luciano Arnaldo Romero Calla, Lizeth Joseline Fuentes
// Perez
// The original implementation uses CHE. Here we use OpenMesh

#include <assert.h>
#include "../common/openmesh_report.h"
#include "../common/openmesh_trimesh.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/util/export_tools.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

inline float compute_toplesets(TriMesh&                        mesh,
                               rxmesh::DenseMatrix<int>&       sorted_index,
                               rxmesh::DenseMatrix<int>&       limits,
                               int&                            limits_size,
                               std::vector<int>&               toplesets,
                               const rxmesh::DenseMatrix<int>& h_seeds)
{
    limits_size = 0;
    if (h_seeds.rows() == 0) {
        return 0;
    }

    rxmesh::CPUTimer timer;
    timer.start();

    toplesets.clear();
    toplesets.resize(mesh.n_vertices(), INVALID32);
    int level = 0;
    int p     = 0;
    for (int k = 0; k < h_seeds.rows(); ++k) {
        const int s        = h_seeds(k, 0);
        sorted_index(p, 0) = s;
        p++;
        if (toplesets[s] == INVALID32) {
            toplesets[s] = level;
        }
    }

    limits(limits_size++, 0) = 0;
    for (int i = 0; i < p; i++) {
        const int v = sorted_index(i, 0);
        if (toplesets[v] > level) {
            level++;
            limits(limits_size++, 0) = i;
        }

        TriMesh::VertexIter v_iter = mesh.vertices_begin() + v;
        for (TriMesh::VertexVertexIter vv_iter = mesh.vv_iter(*v_iter);
             vv_iter.is_valid();
             ++vv_iter) {
            int vv = (*vv_iter).idx();
            if (toplesets[vv] == INVALID32) {
                toplesets[vv]      = toplesets[v] + 1;
                sorted_index(p, 0) = vv;
                p++;
            }
        }
    }

    assert(p <= mesh.n_vertices());
    limits(limits_size++, 0) = p;

    if (limits(limits_size - 1, 0) != mesh.n_vertices()) {
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
void geodesic_ptp_openmesh(const rxmesh::DenseMatrix<int>& h_seeds,
                           rxmesh::DenseMatrix<int>&       limits,
                           int&                            limits_size,
                           std::vector<int>&               toplesets)
{
    TriMesh input_mesh;
    if (!OpenMesh::IO::read_mesh(input_mesh, Arg.obj_file_name)) {
        RXMESH_ERROR("Failed to read mesh from {}", Arg.obj_file_name);
        return;
    }

    // sorted indices for toplesets
    rxmesh::DenseMatrix<int> sorted_index(
        input_mesh.n_vertices(), 1, rxmesh::HOST);

    // compute toplesets
    float compute_toplesets_time = compute_toplesets(
        input_mesh, sorted_index, limits, limits_size, toplesets, h_seeds);

    RXMESH_INFO("OpenMesh: Computing toplesets took {} (ms)",
                compute_toplesets_time);
}