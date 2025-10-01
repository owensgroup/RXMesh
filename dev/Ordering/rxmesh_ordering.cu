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

    // Print GPU memory info
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    spdlog::info("GPU memory: {:.2f} MB free out of {:.2f} MB total", 
                 free_mem / 1024.0 / 1024.0, 
                 total_mem / 1024.0 / 1024.0);

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
    perm.resize(rx.get_num_vertices());

    // Call RXMesh ND permutation
    spdlog::info("Computing ND permutation...");
    rxmesh::nd_permute(rx, perm.data());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        spdlog::error("CUDA error after nd_permute: {}", cudaGetErrorString(err));
    }

    // Ensure all GPU operations complete before returning
    cudaDeviceSynchronize();

    spdlog::info("RXMesh ND ordering computed for {} vertices", perm.size());
}


RXMESH_Ordering_Type RXMeshOrdering::type() const
{
    return RXMESH_Ordering_Type::RXMESH_ND;
}


}