//
// Created by behrooz on 2025-10-07.
//
#include <amd.h>
#include <metis.h>
#include <cassert>
#include <cmath>
#include <unordered_set>
#include "gpu_ordering_v3.h"
#include "min_vertex_cover_bipartite.h"
#include "rxmesh/rxmesh_static.h"
#include "spdlog/spdlog.h"

namespace RXMESH_SOLVER {

GPUOrdering_V3::GPUOrdering_V3() : Gp(nullptr), Gi(nullptr), G_n(0), G_nnz(0)
{
}

GPUOrdering_V3::~GPUOrdering_V3()
{
}

void GPUOrdering_V3::setGraph(const int* Gp, const int* Gi, int G_N, int NNZ)
{
    this->Gp    = Gp;
    this->Gi    = Gi;
    this->G_n   = G_N;
    this->G_nnz = NNZ;
#ifndef NDEBUG
    spdlog::info("Checking that the graph has no diagonal entry...");
    for (int i = 0; i < G_N; i ++) {
        for (int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; nbr_ptr++) {
            int nbr_idx = Gi[nbr_ptr];
            if (nbr_idx == i) {
                spdlog::error("setGraph: Found diagonal entry at node {}!", i);
                spdlog::error("  Gp[{}] = {}, Gp[{}] = {}", i, Gp[i], i + 1, Gp[i + 1]);
                spdlog::error("  Full adjacency list:");
                for (int ptr = Gp[i]; ptr < Gp[i + 1]; ++ptr) {
                    spdlog::error("    Gi[{}] = {}", ptr, Gi[ptr]);
                }
            }
            assert(nbr_idx != i);
        }
    }
    spdlog::info("Graph has no diagonal entry");
    
    // Also specifically check node 3697 if it exists
    if (G_N > 3697) {
        spdlog::info("Specifically checking node 3697:");
        spdlog::info("  Gp[3697] = {}, Gp[3698] = {}", Gp[3697], Gp[3698]);
        spdlog::info("  Neighbors of node 3697:");
        for (int ptr = Gp[3697]; ptr < Gp[3698]; ++ptr) {
            spdlog::info("    Gi[{}] = {}", ptr, Gi[ptr]);
        }
    }
#endif

}


void GPUOrdering_V3::setMesh(const double* V_data,
                             int           V_rows,
                             int           V_cols,
                             const int*    F_data,
                             int           F_rows,
                             int           F_cols)
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


void GPUOrdering_V3::local_permute_metis(Eigen::SparseMatrix<int>& local_graph,
                                         std::vector<int>& local_permutation)
{
    idx_t N   = local_graph.rows();
    idx_t NNZ = local_graph.nonZeros();
    local_permutation.resize(N);
    if (NNZ == 0) {
        assert(N != 0);
        for (int i = 0; i < N; i++) {
            local_permutation[i] = i;
        }
        return;
    }

    std::vector<int> tmp(N);
    METIS_NodeND(&N,
                 local_graph.outerIndexPtr(),
                 local_graph.innerIndexPtr(),
                 NULL,
                 NULL,
                 local_permutation.data(),
                 tmp.data());
}

void GPUOrdering_V3::local_permute_amd(Eigen::SparseMatrix<int>& local_graph,
                                       std::vector<int>& local_permutation)
{
    idx_t N   = local_graph.rows();
    idx_t NNZ = local_graph.nonZeros();
    local_permutation.resize(N);
    if (NNZ == 0) {
        assert(N != 0);
        for (int i = 0; i < N; i++) {
            local_permutation[i] = i;
        }
        return;
    }
    std::vector<int> tmp(N);
    amd_order(N,
              local_graph.outerIndexPtr(),
              local_graph.innerIndexPtr(),
              local_permutation.data(),
              nullptr,
              nullptr);
}

void GPUOrdering_V3::local_permute_unity(Eigen::SparseMatrix<int>& local_graph,
                                         std::vector<int>& local_permutation)
{
    local_permutation.resize(local_graph.rows());
    for (int i = 0; i < local_graph.rows(); i++) {
        local_permutation[i] = i;
    }
}

void GPUOrdering_V3::local_permute(Eigen::SparseMatrix<int>& local_graph,
                                   std::vector<int>&         local_permutation)
{
    if (this->local_permute_method == "metis") {
        local_permute_metis(local_graph, local_permutation);
    } else if (this->local_permute_method == "amd") {
        local_permute_amd(local_graph, local_permutation);
    } else if (this->local_permute_method == "unity") {
        local_permute_unity(local_graph, local_permutation);
    } else {
        spdlog::error("Invalid local permutation method: {}",
                      this->local_permute_method);
        return;
    }
}


void GPUOrdering_V3::refine_separator_with_metis_internals(
    int  N,
    int* Gp,
    int* Gi,
    int* where /* in/out {0,1,2} */)
{
    std::vector<idx_t> hmarker(N);
    for (int i = 0; i < N; ++i) {
        if (where[i] == 2) {
            hmarker[i] = -1;
        } else {
            hmarker[i] = 2;
        }
    }

    // Use METIS default imbalance tolerance: ufactor = 30  -> ratio 1.03
    const real_t ub = 1.03;

    // Run refinement (updates 'w' in place)
    METIS_NodeRefine(N,
                     Gp,
                     /*vwgt*/ nullptr,  // unit weights
                     Gi,
                     where,
                     hmarker.data(),
                     ub);  // :contentReference[oaicite:28]{index=28}
}


void GPUOrdering_V3::compute_separator(
    Eigen::SparseMatrix<int>& graph,           ///<[in] The local graph
    Eigen::SparseMatrix<int>& quotient_graph,  ///<[in] The local quotient graph
    std::vector<int>& quotient_graph_node_weights,  ///<[in] The node weights of
                                                    ///<the local quotient graph
    std::vector<int>& node_to_partition,   ///<[in] The node to partition map
    std::vector<int>& separator_nodes,     ///<[out] The separator nodes
    std::vector<int>& left_assigned_dofs,  ///<[out] The left assigned dofs
    std::vector<int>& right_assigned_dofs  ///<[out] The right assigned dofs
)
{

    // TODO: IMPLEMENT THIS FUNCTION
    // General Flow:
    //  Step 1: Use the separator computation to find the patches that are
    //  separator Step 2: refine separator nodes using METIS logic Return left
    //  and right assigned dofs with separator nodes
    std::vector<int> where(graph.rows(), 0);
    idx_t            Q_N = quotient_graph.rows();
    if (Q_N < 2) {
        // Call vertex separator on the actual graph
        spdlog::info("Computing separator for actual graph");
        std::vector<int> vweight(graph.rows(), 1);
        idx_t            N = graph.rows();
        idx_t            csp;
        int              ret = METIS_ComputeVertexSeparator(&N,
                                               graph.outerIndexPtr(),
                                               graph.innerIndexPtr(),
                                               vweight.data(),
                                               NULL,
                                               &csp,
                                               where.data());
    } else {
        spdlog::info("Computing separator for quotient graph");
        std::vector<int> q_where(Q_N, 0);
        assert(Q_N == quotient_graph_node_weights.size());
        idx_t csp;
        int   ret =
            METIS_ComputeVertexSeparator(&Q_N,
                                         quotient_graph.outerIndexPtr(),
                                         quotient_graph.innerIndexPtr(),
                                         quotient_graph_node_weights.data(),
                                         NULL,
                                         &csp,
                                         q_where.data());

        // Map coarse to fine nodes
        int  sep_size_before_refinement = 0;
        bool has_separator              = false;
        for (int i = 0; i < graph.rows(); i++) {
            where[i] = q_where[node_to_partition[i]];
            if (where[i] == 2) {
                sep_size_before_refinement++;
                has_separator = true;
            }
        }


        // Refine separator nodes using METIS logic
        spdlog::info("Refining separator nodes using METIS logic");
        if (has_separator) {
            refine_separator_with_metis_internals(graph.rows(),
                                                  graph.outerIndexPtr(),
                                                  graph.innerIndexPtr(),
                                                  where.data());
        }


        // Assign the separator nodes to the left and right assigned dofs
        spdlog::info("Assigning separator nodes to the left and right assigned dofs");
        left_assigned_dofs.clear();
        right_assigned_dofs.clear();
        separator_nodes.clear();
        for (int i = 0; i < graph.rows(); i++) {
            if (where[i] == 0) {
                left_assigned_dofs.push_back(i);
            } else if (where[i] == 1) {
                right_assigned_dofs.push_back(i);
            } else if (where[i] == 2) {
                separator_nodes.push_back(i);
            }
        }
        spdlog::info("Separator size reduced from {} -> {}",
                     sep_size_before_refinement,
                     separator_nodes.size());
    }
}


void GPUOrdering_V3::compute_sub_graph(
    std::vector<int>&         nodes,
    Eigen::SparseMatrix<int>& local_graph) const
{
    // Compute global node to local node mapping
    std::vector<int> global_to_local(G_n, -1);
    for (int i = 0; i < nodes.size(); ++i) {
        assert(global_to_local[nodes[i]] == -1);
        global_to_local[nodes[i]] = i;
    }

    // Compute triplets for the sub graph
    std::vector<Eigen::Triplet<int>> triplets;
    for (int i = 0; i < nodes.size(); ++i) {
        int node_id = nodes[i];
        for (int nbr_ptr = Gp[node_id]; nbr_ptr < Gp[node_id + 1];
             ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            if (node_id == nbr_id) {
                spdlog::info("node {} has a diagonal edge to itself", node_id);
                spdlog::info("  Gp[{}] = {}, Gp[{}] = {}", node_id, Gp[node_id], node_id + 1, Gp[node_id + 1]);
                spdlog::info("  nbr_ptr = {}, Gi[{}] = {}", nbr_ptr, nbr_ptr, nbr_id);
                spdlog::info("  Full adjacency list for node {}:", node_id);
                for (int ptr = Gp[node_id]; ptr < Gp[node_id + 1]; ++ptr) {
                    spdlog::info("    Gi[{}] = {}", ptr, Gi[ptr]);
                }
            }


            assert(node_id != nbr_id);
            if (global_to_local[nbr_id] == -1)
                continue;
            triplets.emplace_back(i, global_to_local[nbr_id], 1);
            assert(i != global_to_local[nbr_id]);
            assert(i == global_to_local[nodes[i]]);
        }
    }

    // Form the sub graph
    local_graph.resize(nodes.size(), nodes.size());
    local_graph.setFromTriplets(triplets.begin(), triplets.end());
    local_graph.makeCompressed();
}


int GPUOrdering_V3::post_order_offset_computation(int offset,
                                                  int decomposition_node_id)
{
    assert(decomposition_node_id <
           decomposition_tree.get_number_of_decomposition_nodes());
    auto& decomposition_node =
        decomposition_tree.decomposition_nodes[decomposition_node_id];
    // Compute the offset for the left and right children
    int left_node    = decomposition_node.left_node_idx;
    int right_node   = decomposition_node.right_node_idx;
    int right_offset = offset;
    if (left_node != -1) {
        right_offset = post_order_offset_computation(offset, left_node);
    }
    int separator_offset = right_offset;
    if (right_node != -1) {
        separator_offset =
            post_order_offset_computation(right_offset, right_node);
    }
    decomposition_node.offset = separator_offset;
    offset = separator_offset + decomposition_node.dofs.size();
    return offset;
}


void GPUOrdering_V3::normalize_node_to_patch(
    std::vector<std::pair<int, int>>& node_to_patch)
{
    std::set<int>      unique_ids;
    std::map<int, int> node_per_patch_count;
    std::map<int, int> old_to_new_patch_id;
    for (int i = 0; i < node_to_patch.size(); ++i) {
        unique_ids.insert(node_to_patch[i].second);
        node_per_patch_count[node_to_patch[i].second]++;
    }

    int new_patch_id = 0;
    for (auto& id : unique_ids) {
        old_to_new_patch_id[id] = new_patch_id;
        new_patch_id++;
    }

    for (int i = 0; i < node_to_patch.size(); ++i) {
        node_to_patch[i].second = old_to_new_patch_id[node_to_patch[i].second];
        assert(node_to_patch[i].second >= 0 &&
               node_to_patch[i].second < new_patch_id);
    }
}

void GPUOrdering_V3::compute_quotient_graph(
    std::vector<int>&         assigned_dofs,
    Eigen::SparseMatrix<int>& quotient_graph,
    std::vector<int>&         quotient_graph_node_weights,
    std::vector<int>&         global_to_local_patch_id)
{
    // TODO: IMPLEMENT THIS FUNCTION
    std::vector<bool> in_assigned_dofs(this->G_n, false);
    for (auto& dof : assigned_dofs) {
        in_assigned_dofs[dof] = true;
    }
    std::set<int> unique_patch_ids;
    for (auto& dof : assigned_dofs) {
        unique_patch_ids.insert(this->node_to_patch[dof]);
    }
    global_to_local_patch_id.clear();
    global_to_local_patch_id.resize(this->num_patches, -1);
    int local_patch_id = 0;
    for (auto& patch_id : unique_patch_ids) {
        global_to_local_patch_id[patch_id] = local_patch_id;
        local_patch_id++;
    }

    quotient_graph_node_weights.clear();
    quotient_graph_node_weights.resize(unique_patch_ids.size(), 0);
    // Create triplet for sparse matrix creation
    std::vector<Eigen::Triplet<int>> triplets;
    for (int i = 0; i < assigned_dofs.size(); ++i) {
        int node_id = assigned_dofs[i];
        int local_patch_id =
            global_to_local_patch_id[this->node_to_patch[node_id]];
        assert(local_patch_id != -1);
        quotient_graph_node_weights[local_patch_id]++;
        for (int nbr_ptr = Gp[node_id]; nbr_ptr < Gp[node_id + 1]; ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            if (!in_assigned_dofs[nbr_id])
                continue;
            int nbr_local_patch_id =
                global_to_local_patch_id[this->node_to_patch[nbr_id]];
            // The nbr is not part of this local graph
            if (nbr_local_patch_id == -1)
                continue;

            if (nbr_local_patch_id == local_patch_id)
                continue;  // Skip boundary vertices and self-loops
            assert(nbr_local_patch_id != -1);
            assert(local_patch_id != -1);
            triplets.emplace_back(local_patch_id, nbr_local_patch_id, 1);
        }
    }
    quotient_graph.resize(unique_patch_ids.size(), unique_patch_ids.size());
    quotient_graph.setFromTriplets(triplets.begin(), triplets.end());
    quotient_graph.makeCompressed();
}

void GPUOrdering_V3::decompose()
{
    int total_number_of_decomposition_nodes =
        this->decomposition_tree.get_number_of_decomposition_nodes();
    int wavefront_levels = std::log2(
        total_number_of_decomposition_nodes);  // The tree has levels from 0 to
                                               // wavefront_levels - 1

    // Think of this as the input of decompose function in recursive manner
    struct DecompositionInfo
    {
        int              decomposition_node_id        = -1;
        int              decomposition_node_parent_id = -1;
        std::vector<int> assigned_dofs;
    };

    // Initialize the first call for the decomposition info stack
    std::vector<DecompositionInfo> decomposition_info_stack(
        total_number_of_decomposition_nodes);
    decomposition_info_stack[0].decomposition_node_id        = 0;
    decomposition_info_stack[0].decomposition_node_parent_id = -1;
    for (int i = 0; i < this->node_to_patch.size(); ++i) {
        decomposition_info_stack[0].assigned_dofs.push_back(i);
    }

    // #pragma omp parallel
    {
        for (int l = 0; l < wavefront_levels; l++) {
            int start_level_idx = (1 << l) - 1;
            int end_level_idx   = (1 << (l + 1)) - 1;
            assert(end_level_idx < total_number_of_decomposition_nodes);
            // #pragma omp for schedule(dynamic)
            for (int node_idx = start_level_idx; node_idx < end_level_idx;
                 ++node_idx) {
                // Get the input information for the current node
                spdlog::info("*********** Working on node {} ***********", node_idx);
                int decomposition_node_id =
                    decomposition_info_stack[node_idx].decomposition_node_id;
                int decomposition_node_parent_id =
                    decomposition_info_stack[node_idx]
                        .decomposition_node_parent_id;
                std::vector<int>& assigned_dofs =
                    decomposition_info_stack[node_idx].assigned_dofs;
                auto& cur_decomposition_node =
                    decomposition_tree
                        .decomposition_nodes[decomposition_node_id];
                //+++++++++++++ If there are no patches assigned to the node
                //++++++++++++++++
                if (assigned_dofs.size() == 0)
                    continue;

                if (decomposition_node_id == -1 ||
                    decomposition_node_parent_id == -1) {
                    if (decomposition_node_id != 0) {
                        spdlog::error("The info is not initialized correctly.");
                        spdlog::error("The level is {}", l);
                        spdlog::error("The node_id {}, and parent_id {}",
                                      decomposition_node_id,
                                      decomposition_node_parent_id);
                    }
                }


                //+++++++++++++ If it is a leaf node ++++++++++++++++
                if (l == wavefront_levels - 1) {
                    // Add all the nodes in the patches to the dofs
                    assert(assigned_dofs.size() > 0 &&
                           "Leaf node should have at least one DOF");
                    cur_decomposition_node.init_node(
                        -1,
                        -1,
                        decomposition_node_id,
                        decomposition_node_parent_id,
                        l,
                        assigned_dofs);
                    continue;
                }

                //+++++++++++++ If it is not a leaf node ++++++++++++++++
                // Overall flow:
                // Step 1: Create local graph and local quotient graph
                // Step 2: Compute the separator nodes
                // Step 3: Initialize the input of the left and right
                // childeren for the next wavefront
                // Local graph
                Eigen::SparseMatrix<int> local_graph;
                this->compute_sub_graph(assigned_dofs, local_graph);

                Eigen::SparseMatrix<int> local_quotient_graph;
                std::vector<int>         local_quotient_graph_node_weights;
                std::vector<int>         global_to_local_patch_id;
                this->compute_quotient_graph(assigned_dofs,
                                             local_quotient_graph,
                                             local_quotient_graph_node_weights,
                                             global_to_local_patch_id);

                std::vector<int> local_node_to_partition(assigned_dofs.size(),
                                                         -1);
                for (int i = 0; i < assigned_dofs.size(); ++i) {
                    local_node_to_partition[i] = global_to_local_patch_id
                        [this->node_to_patch[assigned_dofs[i]]];
                }

                // Compute the separator nodes using the local graph and local
                // quotient graph
                std::vector<int> left_assigned_dofs, right_assigned_dofs;
                std::vector<int> separator_nodes;
                this->compute_separator(local_graph,
                                        local_quotient_graph,
                                        local_quotient_graph_node_weights,
                                        local_node_to_partition,
                                        separator_nodes,
                                        left_assigned_dofs,
                                        right_assigned_dofs);

                // Map the local nodes to the global nodes
                std::vector<int> global_separator_nodes;
                std::vector<int> global_left_assigned_dofs;
                std::vector<int> global_right_assigned_dofs;
                for (int i = 0; i < separator_nodes.size(); ++i) {
                    global_separator_nodes.push_back(
                        assigned_dofs[separator_nodes[i]]);
                }
                for (int i = 0; i < left_assigned_dofs.size(); ++i) {
                    global_left_assigned_dofs.push_back(
                        assigned_dofs[left_assigned_dofs[i]]);
                }
                for (int i = 0; i < right_assigned_dofs.size(); ++i) {
                    global_right_assigned_dofs.push_back(
                        assigned_dofs[right_assigned_dofs[i]]);
                }

                // Initialize the input of the left and right children for the
                // next wavefront
                int left_node_idx  = decomposition_node_id * 2 + 1;
                int right_node_idx = decomposition_node_id * 2 + 2;
                if (left_assigned_dofs.empty()) {
                    left_node_idx = -1;
                }
                if (right_assigned_dofs.empty()) {
                    right_node_idx = -1;
                }

                std::sort(global_separator_nodes.begin(),
                          global_separator_nodes.end());
                std::sort(global_left_assigned_dofs.begin(),
                          global_left_assigned_dofs.end());
                std::sort(global_right_assigned_dofs.begin(),
                          global_right_assigned_dofs.end());
                cur_decomposition_node.init_node(left_node_idx,
                                                 right_node_idx,
                                                 decomposition_node_id,
                                                 decomposition_node_parent_id,
                                                 l,
                                                 global_separator_nodes);

                if (left_node_idx != -1) {
                    assert(left_node_idx >= 0 &&
                           left_node_idx < decomposition_info_stack.size());
                    assert(!global_left_assigned_dofs.empty() &&
                           "Left child should have assigned patches");
                    decomposition_info_stack[left_node_idx]
                        .decomposition_node_id = left_node_idx;
                    decomposition_info_stack[left_node_idx]
                        .decomposition_node_parent_id = decomposition_node_id;
                    decomposition_info_stack[left_node_idx].assigned_dofs =
                        global_left_assigned_dofs;
                }

                // Initialize the right child
                if (right_node_idx != -1) {
                    assert(right_node_idx >= 0 &&
                           right_node_idx < decomposition_info_stack.size());
                    assert(!global_right_assigned_dofs.empty() &&
                           "Right child should have assigned patches");
                    decomposition_info_stack[right_node_idx]
                        .decomposition_node_id = right_node_idx;
                    decomposition_info_stack[right_node_idx]
                        .decomposition_node_parent_id = decomposition_node_id;
                    decomposition_info_stack[right_node_idx].assigned_dofs =
                        global_right_assigned_dofs;
                }
            }
        }
    }
}

void GPUOrdering_V3::init_patches()
{
    // Create RXMeshStatic from the mesh data (face-vertex connectivity)
    // Use default patch size of 512 (can be adjusted)
    rxmesh::rx_init(0);
    rxmesh::RXMeshStatic rx(fv, "", patch_size);

    spdlog::info(
        "RXMesh initialized with {} vertices, {} edges, {} faces, {} patches",
        rx.get_num_vertices(),
        rx.get_num_edges(),
        rx.get_num_faces(),
        rx.get_num_patches());

    this->node_to_patch.resize(rx.get_num_vertices());
    rx.for_each_vertex(
        rxmesh::HOST,
        [&](const rxmesh::VertexHandle vh) {
            uint32_t node_id             = rx.map_to_global(vh);
            this->node_to_patch[node_id] = static_cast<int>(vh.patch_id());
        },
        NULL,
        false);
}

void GPUOrdering_V3::step1_compute_node_to_patch()
{
    // Given node to patch, first give each separator node a unique patch ID
    // Step 1: assign patch-id -1 to each boundary vertex
    // Count the number of patch ids
    std::unordered_set<int> unique_ids;
    std::map<int, int>      node_per_patch_count;
    for (int i = 0; i < this->node_to_patch.size(); ++i) {
        unique_ids.insert(this->node_to_patch[i]);
        node_per_patch_count[this->node_to_patch[i]]++;
    }
    // Count the offset of each patch to create continuous patch ids
    std::vector<int> patch_offset(unique_ids.size(), 0);
    for (int i = 0; i < unique_ids.size(); ++i) {
        // It is an empty patch and all the patch after it should be offset by 1
        if (node_per_patch_count[i] == 0) {
            for (int j = i; j < unique_ids.size(); ++j) {
                patch_offset[j]++;
            }
        }
    }
    std::unordered_set<int> unique_ids_reduced;
    for (int i = 0; i < this->node_to_patch.size(); ++i) {
        this->node_to_patch[i] -= patch_offset[this->node_to_patch[i]];
        unique_ids_reduced.insert(this->node_to_patch[i]);
    }
    spdlog::info(
        "Number of initial patches before and after reduction: {} -> {}",
        unique_ids.size(),
        unique_ids_reduced.size());
    this->num_patches = unique_ids_reduced.size();
    assert(node_to_patch.size() == G_n);
}

void GPUOrdering_V3::step2_create_decomposition_tree()
{
    // Step 2.1: create the hierarchical partitioning
    int num_levels = std::ceil(std::log2(this->num_patches));
    int total_number_of_decomposition_nodes = (1 << (num_levels + 1)) - 1;
    decomposition_tree.init_decomposition_tree(
        total_number_of_decomposition_nodes, num_levels);
    spdlog::info("Decomposition tree creation .. ");
    spdlog::info("Number of decomposition levels: {}", num_levels);
    spdlog::info("Using local permutation {}", local_permute_method);

    decompose();
    spdlog::info("Decomposition tree is created.");


#ifndef NDEBUG
    //Check to see if all the nodes exist
    spdlog::info("Checking decomposition validation.");
    std::vector<bool> is_visited(this->G_n, false);
    for(auto& node : decomposition_tree.decomposition_nodes) {
        if (node.dofs.empty())
            continue;
        for (auto& dof : node.dofs) {
            assert(is_visited[dof] == false);
            is_visited[dof] = true;
        }
    }
    for (int i = 0; i < is_visited.size(); ++i) {
        assert(is_visited[i] == true);
    }
    spdlog::info("Decomposition contains all the nodes");

    //Check for whether separators are valid
#endif
}

void GPUOrdering_V3::step3_compute_local_permutations()
{
    spdlog::info("Computing local permutations .. ");
    for (auto& node : decomposition_tree.decomposition_nodes) {
        if (node.dofs.empty())
            continue;
        Eigen::SparseMatrix<int> graph;
        this->compute_sub_graph(node.dofs, graph);
        std::vector<int> local_permutation;
        local_permute(graph, local_permutation);
        node.set_local_permutation(local_permutation);
    }
    spdlog::info("Local permutations are computed.");
}

void GPUOrdering_V3::step4_assemble_final_permutation(std::vector<int>& perm)
{
    // Apply the offset to the decomposition nodes
    spdlog::info("Applying offset to the decomposition nodes .. ");
    post_order_offset_computation(0, 0);
    spdlog::info("Offset is applied to the decomposition nodes.");

    perm.clear();
    perm.resize(G_n, -1);
    for (auto& node : decomposition_tree.decomposition_nodes) {
        if (node.dofs.empty())
            continue;
        for (int local_node = 0; local_node < node.dofs.size(); local_node++) {
            int global_node = node.dofs[local_node];
            int perm_index  = node.local_new_labels[local_node] + node.offset;
            assert(global_node >= 0 && global_node < G_n &&
                   "Invalid global node index");
            assert(perm_index >= 0 && perm_index < perm.size() &&
                   "Permutation index out of bounds");
            assert(perm[perm_index] == -1 &&
                   "Permutation slot already filled - duplicate node!");
            perm[perm_index] = global_node;
        }
    }
}

void GPUOrdering_V3::compute_permutation(std::vector<int>& perm)
{
    if (Gp == nullptr || Gi == nullptr || G_n == 0 || G_nnz == 0) {
        spdlog::error(
            "Graph not set. Please call setGraph() before "
            "compute_permutation().");
        return;
    }
    if (fv.size() == 0) {
        spdlog::error(
            "Mesh not set. Please call setMesh() before "
            "compute_permutation().");
        return;
    }

    // Step 1: Create the quotient graph
    step1_compute_node_to_patch();

    // Step 2: Create hierarchical partitioning and compute local permutations
    step2_create_decomposition_tree();

    // Step 3: Compute the local permutations
    step3_compute_local_permutations();

    // Step 3: Assemble the final permutation
    step4_assemble_final_permutation(perm);
}

}  // namespace RXMESH_SOLVER