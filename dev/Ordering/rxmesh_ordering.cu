//
// Created by behrooz on 2025-09-29.
//

#include "rxmesh_ordering.h"

#include <cassert>
#include <iostream>
#include <chrono>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>

// Include RXMesh headers
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/matrix/nd_permute.cuh"
#include "compute_inverse_perm.h"
#include "csv_utils.h"


namespace RXMESH_SOLVER {
RXMeshOrdering::~RXMeshOrdering()
{
}

void RXMeshOrdering::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp = Gp;
    this->Gi = Gi;
    this->G_N = G_N;
    this->G_NNZ = NNZ;
}

void RXMeshOrdering::setMesh(const double* V_data, int V_rows, int V_cols,
                             const int* F_data, int F_rows, int F_cols)
{
    m_has_mesh = true;

    // RXMesh expects std::vector<std::vector<uint32_t>> for faces
    spdlog::info("setMesh: F_rows = {}, F_cols = {}, V_rows = {}, V_cols = {}", 
                 F_rows, F_cols, V_rows, V_cols);
    
    fv.resize(F_rows);
    for (int i = 0; i < F_rows; ++i) {
        fv[i].resize(F_cols);
        for (int j = 0; j < F_cols; ++j) {
            // Eigen stores data in column-major order by default
            fv[i][j] = static_cast<uint32_t>(F_data[i + j * F_rows]);
        }
    }

    // Optionally add vertex coordinates (not strictly needed for ND ordering) 
    vertices.resize(V_rows);
    for (int i = 0; i < V_rows; ++i) {
        vertices[i].resize(V_cols);
        for (int j = 0; j < V_cols; ++j) {
            // Eigen stores data in column-major order by default
            vertices[i][j] = static_cast<float>(V_data[i + j * V_rows]);
        }
    }

}

bool RXMeshOrdering::needsMesh() const
{
    return true;
}

void RXMeshOrdering::compute_permutation(std::vector<int>& perm)
{
    rxmesh::rx_init(0);
    if (!m_has_mesh) {
        spdlog::error("RXMeshOrdering requires mesh data. Call setMesh() before compute_permutation()");
        // Fallback to identity permutation
        perm.resize(G_N);
        for (int i = 0; i < G_N; i++) {
            perm[i] = i;
        }
        return;
    }

    // Synchronize any pending GPU operations before creating RXMesh
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        spdlog::error("CUDA error before RXMesh creation: {}", cudaGetErrorString(err));
    }

    // Create RXMeshStatic object
    spdlog::info("Creating RXMesh with {} faces and {} vertices", fv.size(), vertices.size());
    rxmesh::RXMeshStatic rx(fv);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        spdlog::error("CUDA error after RXMesh creation: {}", cudaGetErrorString(err));
    }

    spdlog::info("RXMesh created successfully");
    rx.add_vertex_coordinates(vertices, "mesh");

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        spdlog::error("CUDA error after adding vertex coordinates: {}", cudaGetErrorString(err));
    }

    // Allocate permutation array
    perm.clear();
    perm.resize(rx.get_num_vertices(), 0);

    // Call RXMesh ND permutation
    spdlog::info("Computing ND permutation...");
    std::vector<int> rx_perm(rx.get_num_vertices(), 0);
    
    // Start wall clock timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    rxmesh::nd_permute(rx, rx_perm.data());
    
    // End wall clock timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    spdlog::info("nd_permute wall clock time: {} ms", duration.count());

    //Converting rx permute into global permute
    std::vector<uint32_t> linear_to_global(rx.get_num_vertices(), -1);
    rx.for_each_vertex(
        rxmesh::HOST,
        [&](const rxmesh::VertexHandle vh) {
            uint32_t rx_id   = rx.linear_id(vh);
            uint32_t g_id  = rx.map_to_global(vh);
            linear_to_global[rx_id] = g_id;
        },
        NULL,
        false);

    //Finding rx mesh linear ids to permuted name
    std::vector<int> rx_mesh_new_labels(rx_perm.size(), 0);
    compute_inverse_perm(rx_perm, rx_mesh_new_labels);

    //Mapping global ids to permuted new labels
    std::vector<int> global_new_labels(rx_perm.size(), -1);
    for (int i = 0; i < linear_to_global.size(); ++i) {
        int global_id = linear_to_global[i];
        int permute_new_label = rx_mesh_new_labels[i];
        global_new_labels[global_id] = permute_new_label;
    }


    //Finally computing the global permutation
    perm.resize(global_new_labels.size(), 0);
    for (int i = 0; i < global_new_labels.size(); ++i) {
        int new_label = global_new_labels[i];
        perm[new_label] = i;
    }


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        spdlog::error("CUDA error after nd_permute: {}", cudaGetErrorString(err));
    }

    spdlog::info("RXMesh ND ordering computed for {} vertices", perm.size());
}


RXMESH_Ordering_Type RXMeshOrdering::type() const
{
    return RXMESH_Ordering_Type::RXMESH_ND;
}

std::string RXMeshOrdering::typeStr() const
{
    return "RXMesh_ND";
}

void RXMeshOrdering::add_record(std::string save_address, std::map<std::string, double> extra_info, std::string mesh_name)
{
    std::string csv_name = save_address + "/sep_runtime_analysis";
    std::vector<std::string> header;
    header.emplace_back("mesh_name");
    header.emplace_back("G_N");
    header.emplace_back("G_NNZ");

    header.emplace_back("ordering_type");
    header.emplace_back("local_permute_method");
    header.emplace_back("separator_finding_method");
    header.emplace_back("separator_refinement_method");
    header.emplace_back("separator_ratio");
    header.emplace_back("fill-ratio");

    PARTH::CSVManager runtime_csv(csv_name, "some address", header,
                                  false);
    runtime_csv.addElementToRecord(mesh_name, "mesh_name");
    runtime_csv.addElementToRecord(G_N, "G_N");
    runtime_csv.addElementToRecord(G_NNZ, "G_NNZ");
    runtime_csv.addElementToRecord(typeStr(), "ordering_type");
    runtime_csv.addElementToRecord("", "local_permute_method");
    runtime_csv.addElementToRecord("", "separator_finding_method");
    runtime_csv.addElementToRecord("", "separator_refinement_method");
    runtime_csv.addElementToRecord("", "separator_ratio");
    runtime_csv.addElementToRecord(extra_info.at("fill-ratio"), "fill-ratio");
    runtime_csv.addRecord();
}
}