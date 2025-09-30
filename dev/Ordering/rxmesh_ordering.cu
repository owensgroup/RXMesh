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
    m_V = V;
    m_F = F;
    m_has_mesh = true;

    // RXMesh expects std::vector<std::vector<uint32_t>> for faces
    
    fv.resize(m_F.rows());
    for (int i = 0; i < m_F.rows(); ++i) {
        fv[i].resize(m_F.cols());
        for (int j = 0; j < m_F.cols(); ++j) {
            fv[i][j] = static_cast<uint32_t>(m_F(i, j));
        }
    }

    // Optionally add vertex coordinates (not strictly needed for ND ordering) 
    vertices.resize(m_V.rows());
    for (int i = 0; i < m_V.rows(); ++i) {
        vertices[i].resize(m_V.cols());
        for (int j = 0; j < m_V.cols(); ++j) {
            vertices[i][j] = static_cast<float>(m_V(i, j));
        }
    }

}

bool RXMeshOrdering::needsMesh() const
{
    return true;
}

void RXMeshOrdering::compute_permutation(std::vector<int>& perm)
{
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
    spdlog::info("Creating RXMesh with {} faces and {} vertices", m_F.rows(), m_V.rows());
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