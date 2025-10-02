//
// Created by behrooz on 2025-09-29.
//

#include "rxmesh_ordering.h"

#include <cassert>
#include <iostream>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>

// Include RXMesh headers
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/matrix/nd_permute.cuh"
#include "compute_inverse_perm.h"


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

void RXMeshOrdering::setMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    m_has_mesh = true;

    // RXMesh expects std::vector<std::vector<uint32_t>> for faces
    
    fv.resize(F.rows());
    for (int i = 0; i < F.rows(); ++i) {
        fv[i].resize(F.cols());
        for (int j = 0; j < F.cols(); ++j) {
            fv[i][j] = static_cast<uint32_t>(F(i, j));
        }
    }

    // Optionally add vertex coordinates (not strictly needed for ND ordering) 
    vertices.resize(V.rows());
    for (int i = 0; i < V.rows(); ++i) {
        vertices[i].resize(V.cols());
        for (int j = 0; j < V.cols(); ++j) {
            vertices[i][j] = static_cast<float>(V(i, j));
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
    rxmesh::nd_permute(rx, rx_perm.data());

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

    // std::vector<int> inv_perm;
    // compute_inverse_perm(perm, inv_perm);
    // perm = inv_perm;

    spdlog::info("RXMesh ND ordering computed for {} vertices", perm.size());
}


RXMESH_Ordering_Type RXMeshOrdering::type() const
{
    return RXMESH_Ordering_Type::RXMESH_ND;
}


}