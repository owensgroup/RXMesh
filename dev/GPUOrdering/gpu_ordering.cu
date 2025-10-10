//
// Created by behrooz on 2025-10-07.
//
#include "gpu_ordering.h"
#include "spdlog/spdlog.h"
#include "rxmesh/rxmesh_static.h"
#include <unordered_set>
#include <metis.h>



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
    int num_patches = std::ceil(G_n * 1.0 / patch_size);
    spdlog::info("Initializing roughly {} patches...", patch_size);
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
    int total_boundary_vertices = 0;
    for (int i = 0; i < G_n; ++i) {
        int patch_id = node_to_patch[i];
        for(int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            int nbr_patch_id = node_to_patch[nbr_id];
            if (patch_id > nbr_patch_id) {
                is_boundary_vertex[i] = true;
                total_boundary_vertices++;
                break;
            }
        }
    }
    spdlog::info("Found {} boundary vertices", total_boundary_vertices);
    spdlog::info("The ratio of boundary vertices to total vertices is {:.2f}%",
                 (total_boundary_vertices * 100.0) / G_n);
}

void GPUOrdering::step2_create_quotient_graph()
{
    // Given node to patch, first give each separator node a unique patch ID
    //Step 1: assign patch-id -1 to each boundary vertex
    //Count the number of patch ids
    std::unordered_set<int> unique_ids;
    for (int i = 0; i < is_boundary_vertex.size(); ++i) {
        if (is_boundary_vertex[i]) {
            node_to_patch[i] = -1;
        } else {
            unique_ids.insert(node_to_patch[i]);
        }
    }
    //Fix the patch ids to be continuous



    // //DEBUGING
    // std::vector<bool> is_id_present(unique_ids.size(), false);
    // for (int id : unique_ids) {
    //     if (id >= 0 && id < is_id_present.size()) {
    //         is_id_present[id] = true;
    //     }
    // }
    //
    // for (int i = 0; i < is_id_present.size(); ++i) {
    //     if (!is_id_present[i]) {
    //         spdlog::info("Found patch {} with no vertices", i);
    //         assert(false);
    //     }
    // }
    
    //Step 2: create quotient graph
    //Step 2.1: rename the vertices of the quotient graph
    int max_patch_id = unique_ids.size();//Assuming patches start from 0
    int quotient_n = max_patch_id + 1;
    map_graph_to_quotient_node.clear();
    map_graph_to_quotient_node.resize(G_n, -1);
    for (int i = 0; i < G_n; ++i) {
        if (node_to_patch[i] != -1) {
            map_graph_to_quotient_node[i] = node_to_patch[i];
        } else {
            map_graph_to_quotient_node[i] = quotient_n++;
        }
    }
    //Step 2.2: create patch nodes
    quotient_nodes.resize(quotient_n);
    for (int i = 0; i < G_n; ++i) {
        int q_id = map_graph_to_quotient_node[i];
        quotient_nodes[q_id].nodes.push_back(i);
        quotient_nodes[q_id].q_id = q_id;
    }

    //Step 2.3: compute the edge and node weights
    Q_n = quotient_n;
    Q_node_weights.clear();
    Q_node_weights.resize(Q_n, 0);
    //Compute the edge and node weights
    int edge_count = 0;
    //Create triplet for sparse matrix creation
    std::vector<Eigen::Triplet<int>> triplets;
    for (int i = 0; i < G_n; ++i) {
        int node_label = map_graph_to_quotient_node[i];
        Q_node_weights[node_label]++;
        for(int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            int nbr_label = map_graph_to_quotient_node[nbr_id];
            if (nbr_label == node_label) continue; // Skip boundary vertices and self-loops
            assert(nbr_label != -1);
            triplets.emplace_back(node_label, nbr_label, 1);
        }
    }
    spdlog::info("Found {} edges", triplets.size() / 2);

    //Step 2.4: Create the graph (note that the values are the weight of each edge)
    Q = Eigen::SparseMatrix<int>(Q_n, Q_n);
    Q.setFromTriplets(triplets.begin(), triplets.end());
    Q.makeCompressed();
}

void GPUOrdering::step3_compute_quotient_permutation()
{
    Q_perm.resize(Q_n);
    idx_t* perm = Q_perm.data();
    idx_t* Qp = Q.outerIndexPtr();
    idx_t* Qi = Q.innerIndexPtr();
    idx_t* edge_weights = Q.valuePtr();  // Edge weights available but not used by METIS_NodeND
    idx_t* node_weights = Q_node_weights.data();
    idx_t N = Q.rows();
    idx_t NNZ = Q.nonZeros();
    
    // Handle the case where there are no edges
    if (NNZ == 0) {
        spdlog::info("Quotient graph has no edges, using identity permutation");
        for (int i = 0; i < Q_n; ++i) {
            Q_perm[i] = i;
        }
        return;
    }
    
    
    // Prepare temporary array for inverse permutation
    std::vector<idx_t> iperm(Q_n);
    
    // Call METIS_NodeND with node weights (edge weights are not supported by METIS_NodeND)
    int metis_status = METIS_NodeND(&N, Qp, Qi, node_weights, NULL, 
                                   perm, iperm.data());
    
    // Handle METIS errors
    if (metis_status == METIS_ERROR_INPUT) {
        spdlog::error("METIS ERROR: Invalid input parameters");
        // Fallback to identity permutation
        for (int i = 0; i < Q_n; ++i) {
            Q_perm[i] = i;
        }
        return;
    } else if (metis_status == METIS_ERROR_MEMORY) {
        spdlog::error("METIS ERROR: Memory allocation failed");
        // Fallback to identity permutation
        for (int i = 0; i < Q_n; ++i) {
            Q_perm[i] = i;
        }
        return;
    } else if (metis_status == METIS_ERROR) {
        spdlog::error("METIS ERROR: General error occurred");
        // Fallback to identity permutation
        for (int i = 0; i < Q_n; ++i) {
            Q_perm[i] = i;
        }
        return;
    }
    
    spdlog::info("Successfully computed quotient graph permutation using METIS");
    spdlog::info("Quotient graph has {} nodes and {} edges", Q_n, NNZ/2);
}

void GPUOrdering::step4_compute_patch_permutation()
{
    //Create global node to with patch id mapping
    global_to_local.resize(G_n);
    for(auto& q_node: quotient_nodes) {
        for(int i = 0; i < q_node.nodes.size(); ++i) {
            global_to_local[q_node.nodes[i]] = i;
        }
    }

    //For each local patch, compute the local graph edges
    std::vector<std::vector<Eigen::Triplet<int>>> per_quotient_patch_triplets(quotient_nodes.size());
    for(int i = 0; i < G_n; ++i) {
        for(int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            int nbr_q_id = map_graph_to_quotient_node[nbr_id];
            int q_id = map_graph_to_quotient_node[i];
            if (q_id == nbr_q_id) {
                int local_id_1 = global_to_local[i];
                int local_id_2 = global_to_local[nbr_id];
                assert(local_id_1 < quotient_nodes[q_id].nodes.size());
                assert(local_id_2 < quotient_nodes[q_id].nodes.size());
                per_quotient_patch_triplets[q_id].emplace_back(local_id_1, local_id_2, 1);
            }
        }
    }
    assert(per_quotient_patch_triplets.size() == Q_n);
    // spdlog::info("Creating {} local patch graphs", per_quotient_patch_triplets.size());

    //For each local patch, compute the local graph permutation
    for(auto& q_node: quotient_nodes) {
        if (q_node.nodes.size() == 0) {//TODO: I am not sure whether it causes problem
            continue;
        }
        if(per_quotient_patch_triplets[q_node.q_id].size() == 0) {
            q_node.permuted_local_labels.resize(q_node.nodes.size());
            //No edges, use identity permutation
            for(int i = 0; i < q_node.nodes.size(); ++i) {
                q_node.permuted_local_labels[i] = i;
            }
            continue;
        }
        Eigen::SparseMatrix<int> local_graph(q_node.nodes.size(), q_node.nodes.size());
        // spdlog::info("Working on quotient node {} with {} vertices and {} edges",
            // q_node.q_id, q_node.nodes.size(), per_quotient_patch_triplets[q_node.q_id].size());
        local_graph.setFromTriplets(per_quotient_patch_triplets[q_node.q_id].begin(), per_quotient_patch_triplets[q_node.q_id].end());
        local_graph.makeCompressed();
        //Compute the permutation
        idx_t local_N = local_graph.rows();
        idx_t* local_p = local_graph.outerIndexPtr();
        idx_t* local_i = local_graph.innerIndexPtr();
        std::vector<idx_t> local_perm(local_N);
        std::vector<idx_t> local_iperm(local_N);
        idx_t* l_perm = local_perm.data();
        idx_t* l_iperm = local_iperm.data();
        // spdlog::info("Computing local permutation");
        METIS_NodeND(&local_N, local_p, local_i, NULL, NULL, l_perm, l_iperm);
        // spdlog::info("Finished computing local permutation");
        //Map the local permutation into labels
        assert(q_node.nodes.size() == local_N);
        q_node.permuted_local_labels.resize(q_node.nodes.size());
        for(int i = 0; i < q_node.nodes.size(); ++i) {
            q_node.permuted_local_labels[local_perm[i]] = i;
        }
    }
}

void GPUOrdering::step5_map_patch_permutation_to_vertex_permutation(std::vector<int>& perm)
{
    //Compute patch offsets to assemble the local permutation into global permutation
    int offset = 0;
    for(int p = 0; p < quotient_nodes.size(); ++p) {
        int q_id = Q_perm[p];
        quotient_nodes[q_id].offset = offset;
        offset += quotient_nodes[q_id].nodes.size();
    }
    //Compute new global labels
    std::vector<int> global_new_labels(G_n);
    for(int i = 0; i < G_n; ++i) {
        int q_id = map_graph_to_quotient_node[i];
        global_new_labels[i] = quotient_nodes[q_id].offset + quotient_nodes[q_id].permuted_local_labels[global_to_local[i]];
    }
    //Compute new global permutation
    perm.clear();
    perm.resize(G_n);
    for(int i = 0; i < G_n; ++i) {
        perm[global_new_labels[i]] = i;
    }   
}

void GPUOrdering::compute_permutation(std::vector<int>& perm)
{
    std::chrono::system_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();
    step1_find_boundary_vertices();
    std::chrono::system_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();
    spdlog::info("Time taken for step1: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
    start_time = end_time;
    step2_create_quotient_graph();
    end_time = std::chrono::high_resolution_clock::now();
    spdlog::info("Time taken for step2: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
    start_time = end_time;
    step3_compute_quotient_permutation();
    end_time = std::chrono::high_resolution_clock::now();
    spdlog::info("Time taken for step3: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
    start_time = end_time;
    step4_compute_patch_permutation();
    end_time = std::chrono::high_resolution_clock::now();
    spdlog::info("Time taken for step4: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
    start_time = end_time;
    step5_map_patch_permutation_to_vertex_permutation(perm);
    end_time = std::chrono::high_resolution_clock::now();
    spdlog::info("Time taken for step5: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

    // perm.resize(G_n);
    // for (int i = 0; i < G_n; ++i) {
    //     perm[i] = i;
    // }
}


}