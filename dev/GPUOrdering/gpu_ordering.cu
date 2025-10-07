//
// Created by behrooz on 2025-10-07.
//
#include "gpu_ordering.h"
#include "spdlog/spdlog.h"
#include "rxmesh/rxmesh_static.h"
#include <algorithm>


namespace RXMESH_SOLVER {

GPUOrdering::GPUOrdering()
    : Gp(nullptr), Gi(nullptr), G_n(0), G_nnz(0), Q_n(0)
{
}

GPUOrdering::~GPUOrdering()
{
}

void GPUOrdering::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp = Gp;
    this->Gi = Gi;
    this->G_n = G_N;
    this->G_nnz = NNZ;
}


void GPUOrdering::setMesh(const double* V_data, int V_rows, int V_cols,
                          const int* F_data, int F_rows, int F_cols)
{
    spdlog::info("Mesh has {} vertices and {} faces", V_rows, F_rows);
    spdlog::info("Faces have {} vertices each", F_cols);

    // Convert raw data to std::vector format for RXMesh
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


void GPUOrdering::init_patches()
{
    // Create RXMeshStatic from the mesh data (face-vertex connectivity)
    // Use default patch size of 512 (can be adjusted)
    rxmesh::rx_init(0);
    int patch_size = std::ceil(G_n * 1.0 / num_patches);
    spdlog::info("Initializing roughly {} patches...", num_patches);
    spdlog::info("Patch size: {}", patch_size);
    rxmesh::RXMeshStatic rx(fv, "", patch_size);
    
    spdlog::info("RXMesh initialized with {} vertices, {} edges, {} faces, {} patches",
                 rx.get_num_vertices(), rx.get_num_edges(), rx.get_num_faces(), 
                 rx.get_num_patches());

    node_to_patch.resize(rx.get_num_vertices());
    rx.for_each_vertex(
        rxmesh::HOST,
        [&](const rxmesh::VertexHandle vh) {
            uint32_t node_id = rx.map_to_global(vh);
            node_to_patch[node_id] = static_cast<int>(vh.patch_id());
        },
        NULL,
        false);
}



void GPUOrdering::step1_find_boundary_vertices()
{        
    // Initialize node_to_patch vector to store vertex-to-patch mapping
    is_boundary_vertex.resize(G_n, false);
    for (int i = 0; i < G_n; ++i) {
        int patch_id = node_to_patch[i];
        for(int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            int nbr_patch_id = node_to_patch[nbr_id];
            if (patch_id > nbr_patch_id) {
                is_boundary_vertex[i] = true;
                break;
            }
        }
    }
    spdlog::info("Found {} boundary vertices", std::count(is_boundary_vertex.begin(),
        is_boundary_vertex.end(), true));
}

void GPUOrdering::step2_create_quotient_graph()
{
}

void GPUOrdering::step3_compute_quotient_permutation()
{
}

void GPUOrdering::step4_compute_patch_permutation()
{
}

void GPUOrdering::step5_map_patch_permutation_to_vertex_permutation()
{
}

void GPUOrdering::compute_permutation(std::vector<int>& perm)
{
    step1_find_boundary_vertices();
    perm.resize(G_n);
    for (int i = 0; i < G_n; ++i) {
        perm[i] = i;
    }
}


}