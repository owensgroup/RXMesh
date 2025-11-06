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

void GPUOrdering_V3::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp    = Gp;
    this->Gi    = Gi;
    this->G_n   = G_N;
    this->G_nnz = NNZ;
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

void GPUOrdering_V3::compute_separator(
    Eigen::SparseMatrix<int>& quotient_graph,
    std::vector<int>&         quotient_graph_node_weights,
    std::vector<int>&         separator_nodes,
    std::vector<std::pair<int, int>>&         left_assigned,
    std::vector<std::pair<int, int>>&         right_assigned)
{
    //TODO: IMPLEMENT THIS FUNCTION
  //General Flow:
  // Step 1: Use the separator computation to find the patches that are separator
  // Step 2: refine separator by converting the separator patches into the actual fine-grain graph
  // Step 3: Create the patches related to left and right children
  

  
}


void GPUOrdering_V3::compute_sub_graph(Eigen::SparseMatrix<int>& graph,
                                       std::vector<int>& graph_node_weights,
                                       Eigen::SparseMatrix<int>& sub_graph,
                                       std::vector<int>& local_node_weights,
                                       std::vector<int>& nodes) const
{
    // Compute global node to local node mapping
    std::vector<int> global_to_local(graph.rows(), -1);
    local_node_weights.resize(nodes.size(), 0);
    for (int i = 0; i < nodes.size(); ++i) {
        global_to_local[nodes[i]] = i;
        local_node_weights[i]     = graph_node_weights[nodes[i]];
    }

    // Compute triplets for the sub graph
    std::vector<Eigen::Triplet<int>> triplets;
    for (int i = 0; i < nodes.size(); ++i) {
        for (int nbr_ptr = graph.outerIndexPtr()[nodes[i]];
             nbr_ptr < graph.outerIndexPtr()[nodes[i] + 1];
             ++nbr_ptr) {
            int nbr_id = graph.innerIndexPtr()[nbr_ptr];
            if (global_to_local[nbr_id] == -1)
                continue;
            triplets.emplace_back(
                i, global_to_local[nbr_id], graph.valuePtr()[nbr_ptr]);
            assert(i != global_to_local[nbr_id]);
            assert(i == global_to_local[nodes[i]]);
        }
    }

    // Form the sub graph
    sub_graph.resize(nodes.size(), nodes.size());
    sub_graph.setFromTriplets(triplets.begin(), triplets.end());
    sub_graph.makeCompressed();
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


void GPUOrdering_V3::normalize_node_to_patch(std::vector<std::pair<int, int>>& node_to_patch) {
    std::set<int> unique_ids;
    std::map<int, int>      node_per_patch_count;
    std::map<int, int>      old_to_new_patch_id;
    for (int i = 0; i < node_to_patch.size(); ++i) {
        unique_ids.insert(node_to_patch[i].second);
        node_per_patch_count[node_to_patch[i].second]++;
    }

    int new_patch_id = 0;
    for(auto& id : unique_ids) {
        old_to_new_patch_id[id] = new_patch_id;
        new_patch_id++;
    }

    for(int i = 0; i < node_to_patch.size(); ++i) {
        node_to_patch[i].second = old_to_new_patch_id[node_to_patch[i].second];
        assert(node_to_patch[i].second >= 0 && node_to_patch[i].second < new_patch_id);
    }
}

void GPUOrdering_V3::compute_quotient_graph(std::vector<std::pair<int, int>>& node_to_patch,
     Eigen::SparseMatrix<int>& quotient_graph,
     std::vector<int>& quotient_graph_node_weights){
        //TODO: IMPLEMENT THIS FUNCTION
        int num_patches = 0;
        std::unordered_set<int> unique_ids;
        for(auto& node_to_patch : node_to_patch) {
            unique_ids.insert(node_to_patch.second);
        }
        num_patches = unique_ids.size();

        quotient_graph_node_weights.clear();
        quotient_graph_node_weights.resize(num_patches, 0);
        std::vector<int> global_node_to_local_patch_id(G_n, -1);
        for(int i = 0; i < node_to_patch.size(); ++i) {
            global_node_to_local_patch_id[node_to_patch[i].first] = node_to_patch[i].second;
        }
        // Create triplet for sparse matrix creation
        std::vector<Eigen::Triplet<int>> triplets;
        for (int i = 0; i < node_to_patch.size(); ++i) {
            int node_id = node_to_patch[i].first;
            int patch_id = node_to_patch[i].second;
            quotient_graph_node_weights[patch_id]++;
            for (int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; ++nbr_ptr) {
                int nbr_id    = Gi[nbr_ptr];
                int nbr_patch_id = global_node_to_local_patch_id[nbr_id];
                if (nbr_patch_id == patch_id)
                    continue;  // Skip boundary vertices and self-loops
                assert(nbr_patch_id != -1);
                triplets.emplace_back(patch_id, nbr_patch_id, 1);
            }
        }
        quotient_graph.resize(num_patches, num_patches);
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
        int              num_patches = -1;
        std::vector<std::pair<int, int>> node_to_patch;
    };

    // Initialize the first call for the decomposition info stack
    std::vector<DecompositionInfo> decomposition_info_stack(
        total_number_of_decomposition_nodes);
    decomposition_info_stack[0].decomposition_node_id        = 0;
    decomposition_info_stack[0].decomposition_node_parent_id = -1;
    decomposition_info_stack[0].num_patches = this->init_num_patches;
    for(int i = 0; i < this->initial_node_to_patch.size(); ++i) {
        decomposition_info_stack[0].node_to_patch.push_back(std::make_pair(i, this->initial_node_to_patch[i]));
    }

    // #pragma omp parallel
    {
        for (int l = 0; l < wavefront_levels; l++) {
            int start_level_idx = (1 << l) - 1;
            int end_level_idx   = (1 << (l + 1)) - 1;
            assert(end_level_idx < total_number_of_decomposition_nodes);
            // #pragma omp for schedule(dynamic)
            for (int node_idx = start_level_idx; node_idx < end_level_idx; ++node_idx) {
                // Get the input information for the current node
                int decomposition_node_id = decomposition_info_stack[node_idx].decomposition_node_id;
                int decomposition_node_parent_id = decomposition_info_stack[node_idx].decomposition_node_parent_id;
                auto& assigned_nodes = decomposition_info_stack[node_idx].node_to_patch;
                int num_patches = decomposition_info_stack[node_idx].num_patches;
                auto& cur_decomposition_node = decomposition_tree.decomposition_nodes[decomposition_node_id];
                //+++++++++++++ If there are no patches assigned to the node ++++++++++++++++
                if (assigned_nodes.size() == 0)
                    continue;

                if (decomposition_node_id == -1 ||
                    decomposition_node_parent_id == -1) {
                    if (decomposition_node_id != 0) {
                        spdlog::error("The info is not initialized correctly.");
                        spdlog::error("The level is {}", l);
                        spdlog::error("The node_id {}, and parent_id {}", decomposition_node_id, decomposition_node_parent_id);
                    }
                }


                //+++++++++++++ If it is a leaf node ++++++++++++++++
                if (l == wavefront_levels - 1) {
                    // Add all the nodes in the patches to the dofs
                    std::vector<int> dofs;
                    for (auto& node_to_patch : assigned_nodes) {
                        int node_id = node_to_patch.first;
                        int patch_id = node_to_patch.second;
                        assert(patch_id >= 0 && patch_id < num_patches &&
                               "Invalid patch_id");
                        dofs.push_back(node_id);
                    }
                    assert(dofs.size() > 0 &&
                           "Leaf node should have at least one DOF");
                    cur_decomposition_node.init_node(
                        -1,
                        -1,
                        decomposition_node_id,
                        decomposition_node_parent_id,
                        l,
                        dofs);
                    continue;
                }
                //**** If there is only one patch assigned to the node */
                if (num_patches == 1) {
                    // Add all the nodes in the patch to the dofs
                    std::vector<int> dofs;
                    for (auto& node_to_patch : assigned_nodes) {
                        int node_id = node_to_patch.first;
                        int patch_id = node_to_patch.second;
                        assert(patch_id >= 0 && patch_id < num_patches &&
                               "Invalid patch_id in single patch case");
                        dofs.push_back(node_id);
                    }
                    assert(dofs.size() > 0 &&
                           "Single patch node should have at least one DOF");
                    // Permute the dofs and initialize the decomposition node
                    cur_decomposition_node.init_node(
                        -1,
                        -1,
                        decomposition_node_id,
                        decomposition_node_parent_id,
                        l,
                        dofs);
                    continue;
                }

                //+++++++++++++ If it is not a leaf node ++++++++++++++++
                // Overall flow:
                // Step 1: Use the separator computation to find the patches that are separator
                // Step 2: refine separator by converting the separator patches into the actual fine-grain graph
                // Step 3: Create a set of new local patches after separation of the separator patches
                // Step 4: Initialize the input of the left and right
                // childeren for the next wavefront

                //=========== Step 1 and 2 and 3: Use the separator computation to find the patches that are separator
                // and refine separator by converting the separator patches into the actual fine-grain graph

                Eigen::SparseMatrix<int> quotient_graph;
                std::vector<int> quotient_graph_node_weights;
                this->compute_quotient_graph(assigned_nodes, quotient_graph, quotient_graph_node_weights);
                // Compute the bipartition of the local quotient graph
                std::vector<std::pair<int, int>> left_assigned, right_assigned;
                std::vector<int> separator_nodes;
                this->compute_separator(quotient_graph, quotient_graph_node_weights, separator_nodes, left_assigned, right_assigned);

                //=========== Step 4: Initialize the input of the left and right
                //children for the next wavefront ============
                int left_node_idx = decomposition_node_id * 2 + 1;
                int right_node_idx = decomposition_node_id * 2 + 2;
                if (left_assigned.empty()) {
                    left_node_idx = -1;
                }
                if (right_assigned.empty()) {
                    right_node_idx = -1;
                }


                cur_decomposition_node.init_node(
                    left_node_idx,
                    right_node_idx,
                    decomposition_node_id,
                    decomposition_node_parent_id,
                    l,
                    separator_nodes);

                //=========== Step 4: Initialize the input of the left and right
                //children for the next wavefront ============ Initialize the
                // left child
                if (left_node_idx != -1) {
                    assert(left_node_idx >= 0 &&
                           left_node_idx < decomposition_info_stack.size());
                    assert(!left_assigned.empty() &&
                           "Left child should have assigned patches");
                    decomposition_info_stack[left_node_idx]
                        .decomposition_node_id = left_node_idx;
                    decomposition_info_stack[left_node_idx]
                        .decomposition_node_parent_id = decomposition_node_id;
                    decomposition_info_stack[left_node_idx]
                        .node_to_patch = left_assigned;
                }

                // Initialize the right child
                if (right_node_idx != -1) {
                    assert(right_node_idx >= 0 &&
                           right_node_idx < decomposition_info_stack.size());
                    assert(!right_assigned.empty() &&
                           "Right child should have assigned patches");
                    decomposition_info_stack[right_node_idx]
                        .decomposition_node_id = right_node_idx;
                    decomposition_info_stack[right_node_idx]
                        .decomposition_node_parent_id = decomposition_node_id;
                    decomposition_info_stack[right_node_idx]
                        .node_to_patch = right_assigned;
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

    initial_node_to_patch.resize(rx.get_num_vertices());
    rx.for_each_vertex(
        rxmesh::HOST,
        [&](const rxmesh::VertexHandle vh) {
            uint32_t node_id       = rx.map_to_global(vh);
            initial_node_to_patch[node_id] = static_cast<int>(vh.patch_id());
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
    for (int i = 0; i < initial_node_to_patch.size(); ++i) {
        unique_ids.insert(initial_node_to_patch[i]);
        node_per_patch_count[initial_node_to_patch[i]]++;
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
    for (int i = 0; i < initial_node_to_patch.size(); ++i) {
        initial_node_to_patch[i] -= patch_offset[initial_node_to_patch[i]];
        unique_ids_reduced.insert(initial_node_to_patch[i]);
    }
    spdlog::info("Number of initial patches before and after reduction: {} -> {}", unique_ids.size(), unique_ids_reduced.size());
    this->init_num_patches = unique_ids_reduced.size();
}

void GPUOrdering_V3::step2_create_decomposition_tree()
{
    // Step 2.1: create the hierarchical partitioning
    this->decomposition_max_level = std::ceil(std::log2(this->init_num_patches));
    int total_number_of_decomposition_nodes =
        (1 << (this->decomposition_max_level + 1)) - 1;
    decomposition_tree.init_decomposition_tree(total_number_of_decomposition_nodes);
    spdlog::info("Decomposition tree creation .. ");
    spdlog::info("Using local permutation {}", local_permute_method);

    decompose();
    spdlog::info("Decomposition tree is created.");
}


void GPUOrdering_V3::step3_compute_local_permutations()
{
    auto compute_sub_graph = [](
        int*              Gp,
        int*              Gi,
        int               G_N,
        std::vector<int>& nodes) -> Eigen::SparseMatrix<int>
    {
        // Compute global node to local node mapping
        std::sort(nodes.begin(), nodes.end());
        std::vector<int> global_to_local(G_N, -1);
        for (int i = 0; i < nodes.size(); ++i) {
            global_to_local[nodes[i]] = i;
        }
    
        // Compute triplets for the sub graph
        std::vector<Eigen::Triplet<int>> triplets;
        for (int i = 0; i < nodes.size(); ++i) {
            for (int nbr_ptr = Gp[nodes[i]]; nbr_ptr < Gp[nodes[i] + 1];
                 ++nbr_ptr) {
                int nbr_id = Gi[nbr_ptr];
                if (global_to_local[nbr_id] == -1)
                    continue;
                assert(i != global_to_local[nbr_id]);
                assert(i == global_to_local[nodes[i]]);
                triplets.emplace_back(i, global_to_local[nbr_id], 1);
            }
        }
        // Form the sub graph
        Eigen::SparseMatrix<int> sub_graph(nodes.size(), nodes.size());
        sub_graph.setFromTriplets(triplets.begin(), triplets.end());
        sub_graph.makeCompressed();
        return sub_graph;
    };


    spdlog::info("Computing local permutations .. ");
    for(auto& node : decomposition_tree.decomposition_nodes) {
        if(node.dofs.empty())
            continue;
        Eigen::SparseMatrix<int> graph = compute_sub_graph(Gp, Gi, G_n, node.dofs);
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
    for(auto& node : decomposition_tree.decomposition_nodes) {
        if(node.dofs.empty())
            continue;
        for(auto& dof : node.dofs) {
            int perm_index = node.local_new_labels[dof] + node.offset;
            assert(perm_index >= 0 && perm_index < perm.size() &&
                   "Permutation index out of bounds");
            assert(perm[perm_index] == -1 &&
                   "Permutation slot already filled - duplicate node!");
            perm[perm_index] = dof;
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