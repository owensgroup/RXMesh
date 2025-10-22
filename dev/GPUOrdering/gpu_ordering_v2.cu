//
// Created by behrooz on 2025-10-07.
//
#include <amd.h>
#include <metis.h>
#include <cassert>
#include <cmath>
#include <unordered_set>
#include "gpu_ordering_v2.h"
#include "rxmesh/rxmesh_static.h"
#include "spdlog/spdlog.h"


namespace RXMESH_SOLVER {

GPUOrdering_V2::GPUOrdering_V2()
    : Gp(nullptr), Gi(nullptr), G_n(0), G_nnz(0)
{
}

GPUOrdering_V2::~GPUOrdering_V2()
{
}

void GPUOrdering_V2::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp    = Gp;
    this->Gi    = Gi;
    this->G_n   = G_N;
    this->G_nnz = NNZ;
}


void GPUOrdering_V2::setMesh(const double* V_data,
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


void GPUOrdering_V2::local_permute_metis(Eigen::SparseMatrix<int>& local_graph,
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

void GPUOrdering_V2::local_permute_amd(Eigen::SparseMatrix<int>& local_graph,
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


void GPUOrdering_V2::local_permute_unity(Eigen::SparseMatrix<int>& local_graph,
                                       std::vector<int>& local_permutation){
    local_permutation.resize(local_graph.rows());
    for (int i = 0; i < local_graph.rows(); i++) {
        local_permutation[i] = i;
    }
}

void GPUOrdering_V2::local_permute(Eigen::SparseMatrix<int>& local_graph,
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

void GPUOrdering_V2::compute_bipartition(
    Eigen::SparseMatrix<int>& quotient_graph,
    std::vector<int>&         quotient_graph_node_weights,
    std::vector<int>&         node_to_partition)
{
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] =
        METIS_OBJTYPE_VOL;  // Total communication volume minimization.
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_CONTIG]    = 0;
    options[METIS_OPTION_COMPRESS]  = 0;
    options[METIS_OPTION_DBGLVL]    = 0;

    idx_t   nvtxs  = quotient_graph.rows();
    idx_t   ncon   = 1;
    idx_t*  vwgt   = NULL;
    idx_t*  vsize  = NULL;
    idx_t   nparts = 2;
    real_t* tpwgts = NULL;
    real_t* ubvec  = NULL;
    idx_t   objval = 0;

    node_to_partition.resize(quotient_graph.rows(), 0);

    int metis_status = METIS_PartGraphKway(&nvtxs,
                                           &ncon,
                                           quotient_graph.outerIndexPtr(),
                                           quotient_graph.innerIndexPtr(),
                                           quotient_graph_node_weights.data(),
                                           vsize,
                                           quotient_graph.valuePtr(),
                                           &nparts,
                                           tpwgts,
                                           ubvec,
                                           options,
                                           &objval,
                                           node_to_partition.data());

    if (metis_status == METIS_ERROR_INPUT) {
        RXMESH_ERROR("METIS ERROR INPUT");
        exit(EXIT_FAILURE);
    } else if (metis_status == METIS_ERROR_MEMORY) {
        RXMESH_ERROR("\n METIS ERROR MEMORY \n");
        exit(EXIT_FAILURE);
    } else if (metis_status == METIS_ERROR) {
        RXMESH_ERROR("\n METIS ERROR\n");
        exit(EXIT_FAILURE);
    }

    // std::vector<int> part_size(nparts, 0);
    // for (int i = 0; i < part.size(); ++i) {
    //     part_size[part[i]]++;
    // }

    // // RXMESH_INFO(" Metis parts size: ");
    // // for (int i = 0; i < part_size.size(); ++i) {
    // //     RXMESH_INFO("   Parts {}= {}", i, part_size[i]);
    // // }
}

void GPUOrdering_V2::find_separator_basic(
    std::vector<int>& graph_to_partition_map,
    std::vector<int>& separator_nodes)
{
    // double partition_1_ratio = 0;
    // double partition_2_ratio = 0;
    // for (int i = 0; i < graph_to_partition_map.size(); ++i) {
    //     if (graph_to_partition_map[i] == 0)
    //         partition_1_ratio++;
    //     else if (graph_to_partition_map[i] == 1)
    //         partition_2_ratio++;
    // }
    // partition_1_ratio = partition_1_ratio / graph_to_partition_map.size();
    // partition_2_ratio = partition_2_ratio / graph_to_partition_map.size();
    // spdlog::info("Partition sizes: {} - {}", partition_1_ratio, partition_2_ratio);
    separator_nodes.clear();
    for (int i = 0; i < G_n; ++i) {
        if (graph_to_partition_map[i] == -1) continue;
        int partition_id = graph_to_partition_map[i];
        for (int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            if (graph_to_partition_map[nbr_id] == -1) continue;
            int nbr_partition_id = graph_to_partition_map[nbr_id];
            if (partition_id > nbr_partition_id) {
                separator_nodes.push_back(i);
                // separator_nodes.push_back(nbr_id);
                break;
            }
        }
    }
}

void GPUOrdering_V2::find_separator_max_degree(
    std::vector<int>& graph_to_partition_map,
    std::vector<int>& separator_nodes)
{
    separator_nodes.clear();
    
    // Edge representation: pair of node IDs (always min, max order)
    using Edge = std::pair<int, int>;
    
    // Custom hash for Edge (pair<int, int>)
    struct EdgeHash {
        std::size_t operator()(const Edge& e) const {
            return std::hash<int>()(e.first) ^ (std::hash<int>()(e.second) << 1);
        }
    };
    
    // Step 1: Identify all cross-partition edges
    std::unordered_set<Edge, EdgeHash> all_cross_edges;
    std::unordered_map<int, std::vector<Edge>> node_to_edges;
    
    assert(graph_to_partition_map.size() == G_n && "graph_to_partition_map size should match G_n");
    for (int i = 0; i < graph_to_partition_map.size(); i++) {
        if (graph_to_partition_map[i] == -1) continue;
        
        int node_partition_id = graph_to_partition_map[i];
        for (int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            if (graph_to_partition_map[nbr_id] == -1) continue;
            
            int nbr_partition_id = graph_to_partition_map[nbr_id];
            if (node_partition_id != nbr_partition_id) {
                // Create edge with consistent ordering (min, max)
                Edge edge = std::make_pair(std::min(i, nbr_id), std::max(i, nbr_id));
                
                // Add edge to global set
                if (all_cross_edges.find(edge) == all_cross_edges.end()) {
                    all_cross_edges.insert(edge);
                }
                
                // Map node to this edge
                node_to_edges[i].push_back(edge);
            }
        }
    }
    
    // If no cross-partition edges, no separator needed
    if (all_cross_edges.empty()) {
        return;
    }
    
    // Step 2: Track uncovered edges
    std::unordered_set<Edge, EdgeHash> uncovered_edges = all_cross_edges;
    
    // Step 3: Greedy selection loop
    while (!uncovered_edges.empty()) {
        int best_node = -1;
        int best_coverage = 0;
        
        // Find the node that covers the most uncovered edges
        for (const auto& kv : node_to_edges) {
            int node = kv.first;
            const auto& edges = kv.second;
            
            // Count how many uncovered edges this node would cover
            int coverage = 0;
            for (const auto& edge : edges) {
                if (uncovered_edges.find(edge) != uncovered_edges.end()) {
                    coverage++;
                }
            }
            
            if (coverage > best_coverage) {
                best_coverage = coverage;
                best_node = node;
            }
        }
        
        // If no node covers any uncovered edges, we're done
        if (best_node == -1 || best_coverage == 0) {
            break;
        }
        
        // Add the best node to separator
        separator_nodes.push_back(best_node);
        
        // Mark all its edges as covered
        const auto& edges = node_to_edges[best_node];
        for (const auto& edge : edges) {
            uncovered_edges.erase(edge);
        }
        
        // Remove this node from consideration
        node_to_edges.erase(best_node);
    }
    
    // Assert that all cross-partition edges are covered by the separator
    assert(uncovered_edges.empty() && "All cross-partition edges should be covered by the separator");
}

void GPUOrdering_V2::find_separator_metis(
    std::vector<int>& graph_to_partition_map,
    std::vector<int>& separator_nodes){

}

void GPUOrdering_V2::find_separator(
    std::vector<int>& graph_to_partition_map,
    std::vector<int>& separator_nodes)
{
    // Use a more advanced method to find the separator
    // For now, use the basic method
    if (separator_finding_method == "basic") {
        find_separator_basic(graph_to_partition_map, separator_nodes);
    } else if (separator_finding_method == "max_degree") {
        find_separator_max_degree(graph_to_partition_map, separator_nodes);
    } else {
        spdlog::error("Invalid separator_finding_method: {}", separator_finding_method);
    }

    //Apply refinement to the separator and patches
    if (separator_refinement_method == "patch_refinement") {
        separator_patch_refinement(graph_to_partition_map, separator_nodes);
    } else if (separator_refinement_method == "redundancy_removal") {
        separator_redundancy_removal(graph_to_partition_map, separator_nodes);
    } else if (separator_refinement_method == "patch_redundancy_refinement") {
        separator_patch_refinement(graph_to_partition_map, separator_nodes);
        separator_redundancy_removal(graph_to_partition_map, separator_nodes);
    } else if (separator_refinement_method =="nothing"){

    } else {
        spdlog::error("Invalid separator_refinement_method: {}", separator_refinement_method);
    }
}

void GPUOrdering_V2::separator_patch_refinement(
    std::vector<int>& graph_to_partition_map,
    std::vector<int>& separator_nodes)
{
    // METIS-style refinement: Move separator nodes to partitions when beneficial
    int initial_separator_size = separator_nodes.size();
    if (separator_nodes.empty()) {
        return;
    }
    
    // Mark separator nodes in graph_to_partition_map as partition 2
    std::unordered_set<int> separator_set(separator_nodes.begin(), separator_nodes.end());
    for (int node : separator_nodes) {
        graph_to_partition_map[node] = 2;  // 2 = separator
    }
    
    bool improvement = true;
    int iteration = 0;
    
    while (improvement) {
        improvement = false;
        iteration++;
        
        // Compute gain for each separator node
        struct NodeGain {
            int node;
            int target_partition;
            int gain;
        };
        std::vector<NodeGain> node_gains;
        
        for (int node : separator_nodes) {
            // Count neighbors in each partition
            int neighbors_in_partition[2] = {0, 0};
            
            for (int nbr_ptr = Gp[node]; nbr_ptr < Gp[node + 1]; ++nbr_ptr) {
                int nbr_id = Gi[nbr_ptr];
                int nbr_partition = graph_to_partition_map[nbr_id];
                
                if (nbr_partition == 0 || nbr_partition == 1) {
                    neighbors_in_partition[nbr_partition]++;
                }
            }
            
            // Compute gain for moving to each partition
            // Gain = 1 (separator size reduction) - (edges to OTHER partition)
            for (int target_partition = 0; target_partition < 2; target_partition++) {
                int other_partition = 1 - target_partition;
                int gain = 1 - neighbors_in_partition[other_partition];
                
                if (gain > 0) {
                    node_gains.push_back({node, target_partition, gain});
                }
            }
        }
        
        if (node_gains.empty()) {
            break;  // No positive gain moves
        }
        
        // Sort by gain (highest first)
        std::sort(node_gains.begin(), node_gains.end(),
                  [](const NodeGain& a, const NodeGain& b) {
                      return a.gain > b.gain;
                  });
        
        // Apply the best move
        const NodeGain& best_move = node_gains[0];
        
        // Move node from separator to partition
        graph_to_partition_map[best_move.node] = best_move.target_partition;
        separator_set.erase(best_move.node);
        improvement = true;
        
        // Update separator_nodes vector
        separator_nodes.clear();
        separator_nodes.insert(separator_nodes.end(), separator_set.begin(), separator_set.end());
    }
    
    if (initial_separator_size > separator_nodes.size()) {
        spdlog::info("Patch refinement reduced separator from {} to {} in {} iterations",
             initial_separator_size,
             separator_nodes.size(),
             iteration);
    }
}

void GPUOrdering_V2::separator_redundancy_removal(
    std::vector<int>& graph_to_partition_map,
    std::vector<int>& separator_nodes)
{
    // Input: separator_nodes already contains the initial separator
    int initial_separator_size = separator_nodes.size();
    if (separator_nodes.empty()) {
        return;
    }
    
    // Step 1: Identify all cross-partition edges
    using Edge = std::pair<int, int>;
    struct EdgeHash {
        std::size_t operator()(const Edge& e) const {
            return std::hash<int>()(e.first) ^ (std::hash<int>()(e.second) << 1);
        }
    };
    
    std::unordered_set<Edge, EdgeHash> all_cross_edges;
    assert(graph_to_partition_map.size() == G_n && "graph_to_partition_map size should match G_n");
    for (int i = 0; i < graph_to_partition_map.size(); i++) {
        if (graph_to_partition_map[i] == -1) continue;
        
        int node_partition_id = graph_to_partition_map[i];
        for (int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            if (graph_to_partition_map[nbr_id] == -1) continue;
            
            int nbr_partition_id = graph_to_partition_map[nbr_id];
            if (node_partition_id != nbr_partition_id) {
                Edge edge = std::make_pair(std::min(i, nbr_id), std::max(i, nbr_id));
                all_cross_edges.insert(edge);
            }
        }
    }
    
    // Step 2: Iterative refinement - remove redundant nodes
    bool nodes_removed = true;
    while (nodes_removed) {
        nodes_removed = false;
        
        // Create a set for fast lookup of current separator nodes
        std::unordered_set<int> separator_set(separator_nodes.begin(), separator_nodes.end());
        
        // Compute degrees and sort by degree (highest first)
        std::vector<std::pair<int, int>> node_degrees; // pair of (degree, node_id)
        for (int node : separator_nodes) {
            int degree = Gp[node + 1] - Gp[node];
            node_degrees.push_back({degree, node});
        }
        std::sort(node_degrees.begin(), node_degrees.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Try to remove each node in order of decreasing degree
        for (const auto& [degree, node] : node_degrees) {
            // Temporarily remove node from separator
            separator_set.erase(node);
            
            // Check if all cross-partition edges are still covered
            bool all_covered = true;
            for (const auto& edge : all_cross_edges) {
                bool covered = separator_set.count(edge.first) > 0 || 
                               separator_set.count(edge.second) > 0;
                if (!covered) {
                    all_covered = false;
                    break;
                }
            }
            
            if (all_covered) {
                // Node is redundant, keep it removed
                nodes_removed = true;
            } else {
                // Node is necessary, restore it
                separator_set.insert(node);
            }
        }
        
        // Update separator_nodes from separator_set
        if (nodes_removed) {
            separator_nodes.clear();
            separator_nodes.insert(separator_nodes.end(), separator_set.begin(), separator_set.end());
        }
    }
    
    if (initial_separator_size > separator_nodes.size()) {
        spdlog::info("Redundancy removal reduced separator from {} to {}",
             initial_separator_size,
             separator_nodes.size());
    }
}

void GPUOrdering_V2::compute_sub_graph(Eigen::SparseMatrix<int>& graph,
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

Eigen::SparseMatrix<int> GPUOrdering_V2::compute_sub_graph(
    int*              Gp,
    int*              Gi,
    int               G_N,
    std::vector<int>& nodes) const
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
}


int GPUOrdering_V2::post_order_offset_computation(int offset, int decomposition_node_id){
    assert(decomposition_node_id < max_match_tree.get_number_of_decomposition_nodes());
    auto& decomposition_node = max_match_tree.decomposition_nodes[decomposition_node_id];
    //Compute the offset for the left and right children
    int left_node = decomposition_node.left_node_idx;
    int right_node = decomposition_node.right_node_idx;
    int right_offset = offset;
    if (left_node != -1) {
        right_offset = post_order_offset_computation(offset, left_node);
    }
    int separator_offset = right_offset;
    if (right_node != -1) {
        separator_offset = post_order_offset_computation(right_offset, right_node);
    }
    decomposition_node.offset = separator_offset;
    offset = separator_offset + decomposition_node.dofs.size();
    return offset;
}

void GPUOrdering_V2::decompose()
{
    int total_number_of_decomposition_nodes =
        this->max_match_tree.get_number_of_decomposition_nodes();
    int wavefront_levels =
        std::log2(total_number_of_decomposition_nodes); // The tree has levels from 0 to wavefront_levels - 1

    // Think of this as the input of decompose function in recursive manner
    struct DecompositionInfo
    {
        int              decomposition_node_id        = -1;
        int              decomposition_node_parent_id = -1;
        std::vector<int> decomposition_node_patches;
    };

    // Initialize the first call for the decomposition info stack
    std::vector<int> total_patches(Q.rows(), -1);
    for (int i = 0; i < Q.rows(); ++i) {
        total_patches[i] = i;
    }

    std::vector<DecompositionInfo> decomposition_info_stack(
        total_number_of_decomposition_nodes);
    decomposition_info_stack[0].decomposition_node_id        = 0;
    decomposition_info_stack[0].decomposition_node_parent_id = -1;
    decomposition_info_stack[0].decomposition_node_patches   = total_patches;

    // #pragma omp parallel
    {
        for (int l = 0; l < wavefront_levels; l++) {
            int start_level_idx = (1 << l) - 1;
            int end_level_idx   = (1 << (l + 1)) - 1;
            assert(end_level_idx < total_number_of_decomposition_nodes);
            #pragma omp parallel for schedule(dynamic)
            for (int node_idx = start_level_idx; node_idx < end_level_idx;
                 node_idx++) {
                // Get the input information for the current node
                int decomposition_node_id = decomposition_info_stack[node_idx].decomposition_node_id;
                int decomposition_node_parent_id = decomposition_info_stack[node_idx].decomposition_node_parent_id;
                std::vector<int>& assigned_patches =decomposition_info_stack[node_idx].decomposition_node_patches;
                auto& cur_decomposition_node = max_match_tree.decomposition_nodes[decomposition_node_id];
                int base_offset_for_decomposition_node_offset = -1;
                //+++++++++++++ If there are no patches assigned to the node
                //++++++++++++++++
                if (assigned_patches.size() == 0)
                    continue;

                if (decomposition_node_id == -1 || decomposition_node_parent_id == -1) {
                    if (decomposition_node_id != 0) {
                        spdlog::error("The info is not initialized correctly.");
                        spdlog::error("The level is {}", l);
                        spdlog::error("The node_id {}, and parent_id {}", decomposition_node_id, decomposition_node_parent_id);
                    }
                }


                //+++++++++++++ If it is a leaf node ++++++++++++++++
                if (l == wavefront_levels - 1) {
                //Add all the nodes in the patches to the dofs
                std::vector<int> dofs;
                for (auto& patch_id : assigned_patches) {
                    assert(patch_id >= 0 && patch_id < patch_nodes.size() && "Invalid patch_id in leaf node");
                    auto& patch = patch_nodes[patch_id];
                    assert(patch.nodes.size() > 0);
                    for (int i = 0; i < patch.nodes.size(); ++i) {
                        int node_id = patch.nodes[i];
                        assert(node_id >= 0 && node_id < G_n && "Invalid node_id");
                        if (!max_match_tree.is_separator[node_id]) {
                            dofs.push_back(node_id);
                        }
                    }
                }
                assert(dofs.size() > 0 && "Leaf node should have at least one DOF");

                    // if(assigned_patches.size() > 1) {
                    //     //print a warning that in the leaf node, it is better to have only one patch
                    //     spdlog::warn("In the leaf decomposition node, it is better to not have more than one patch");
                    //     spdlog::warn("For node {} in level {}: #Assigned patches: {}", node_idx, l, assigned_patches.size());
                    // }
                    //Permute the dofs and initialize the decomposition node
                    Eigen::SparseMatrix<int> sub_graph =
                        compute_sub_graph(Gp, Gi, G_n, dofs);
                    std::vector<int> local_permutation;
                    this->local_permute(sub_graph, local_permutation);
                    cur_decomposition_node.init_node(-1,
                                                     -1,
                                                     decomposition_node_id,
                                                     decomposition_node_parent_id,
                                                     l,
                                                     dofs,
                                                     local_permutation,
                                                     assigned_patches,
                                                     base_offset_for_decomposition_node_offset);
                    continue;
                }
                //**** If there is only one patch assigned to the node */
                if (assigned_patches.size() == 1) {
                //Add all the nodes in the patch to the dofs
                std::vector<int> dofs;
                for (auto& patch_id : assigned_patches) {
                    assert(patch_id >= 0 && patch_id < patch_nodes.size() && "Invalid patch_id in single patch case");
                    auto& patch = patch_nodes[patch_id];
                    assert(patch.nodes.size() > 0);
                    for (int i = 0; i < patch.nodes.size(); ++i) {
                        int node_id = patch.nodes[i];
                        assert(node_id >= 0 && node_id < G_n && "Invalid node_id");
                        if (!max_match_tree.is_separator[node_id]) {
                            dofs.push_back(node_id);
                        }
                    }
                }
                assert(dofs.size() > 0 && "Single patch node should have at least one DOF");
                    //Permute the dofs and initialize the decomposition node
                    Eigen::SparseMatrix<int> sub_graph =
                        compute_sub_graph(Gp, Gi, G_n, dofs);
                    std::vector<int> local_permutation;
                    this->local_permute(sub_graph, local_permutation);
                    cur_decomposition_node.init_node(-1,
                                                     -1,
                                                     decomposition_node_id,
                                                     decomposition_node_parent_id,
                                                     l,
                                                     dofs,
                                                     local_permutation,
                                                     assigned_patches,
                                                     base_offset_for_decomposition_node_offset);
                    continue;
                }

                //+++++++++++++ If it is not a leaf node ++++++++++++++++
                // Overall flow:
                // Step 1: Divide the patches into two parts (left and right)
                // Step 2: Compute the sparator between these two parts
                // Step 3: Permute the separator and initialize the decompositon
                // node Step 4: Initialize the input of the left and right
                // childeren for the next wavefront

                //=========== Step 1: Divide the patches into two parts (left and right) ============
                //Compute the local quotient graph from the assigned patches
                std::vector<int>         quotient_sub_graph_node_weights;
                Eigen::SparseMatrix<int> quotient_sub_graph;
                std::sort(
                    assigned_patches.begin(),
                    assigned_patches
                        .end());  // The node_to_partition is mapped this way
                this->compute_sub_graph(Q,
                                        Q_node_weights,
                                        quotient_sub_graph,
                                        quotient_sub_graph_node_weights,
                                        assigned_patches);
                // Compute the bipartition of the local quotient graph
                std::vector<int> local_node_to_partition;
                this->compute_bipartition(quotient_sub_graph,
                                          quotient_sub_graph_node_weights,
                                          local_node_to_partition);
                // Compute the two parts of the local quotient graph for the next decomposition node
                std::vector<int> left_assigned, right_assigned;
                int skipped_invalid_patches = 0;
                for (int i = 0; i < local_node_to_partition.size(); i++) {
                    int global_patch_id = assigned_patches[i];
                    assert(global_patch_id >= 0 && global_patch_id < patch_nodes.size() && "Invalid patch ID");
                    //Check if the patch has non-separator nodes - IMPORTANT
                    bool is_valid_patch = false;
                    for (int j = 0; j < patch_nodes[global_patch_id].nodes.size(); ++j) {
                        if (!max_match_tree.is_separator[patch_nodes[global_patch_id].nodes[j]]) {
                            is_valid_patch = true;
                            break;
                        }
                    }
                    if (!is_valid_patch) {
                        skipped_invalid_patches++;
                        continue;
                    }
                    if (local_node_to_partition[i] == 0) {
                        left_assigned.push_back(global_patch_id);
                    } else {
                        right_assigned.push_back(global_patch_id);
                    }
                }
                // Assert that we didn't skip all patches
                assert((left_assigned.size() + right_assigned.size() + skipped_invalid_patches) == assigned_patches.size() &&
                       "All patches should be accounted for");

                //=========== Step 2: Compute the separator between these two parts ============
                // Compute the separator between the two parts of the local quotient graph
                std::vector<int> graph_to_partition_map(G_n, -1);
                std::vector<int> separator;
                // Creating global_to_local_quotient_graph_map
                std::vector<int> global_to_local_node(Q.rows(), -1);
                for (int i = 0; i < assigned_patches.size(); i++) {
                    auto& patch        = patch_nodes[assigned_patches[i]];
                    int   partition_id = local_node_to_partition[i];
                    for (int j = 0; j < patch.nodes.size(); j++) {
                        if (max_match_tree.is_separator[patch.nodes[j]]) continue;
                        graph_to_partition_map[patch.nodes[j]] = partition_id;
                    }
                }
                this->find_separator(graph_to_partition_map, separator);

                // Mark the separator nodes as true
                assert(max_match_tree.is_separator.size() == G_n);
                for (int i = 0; i < separator.size(); ++i) {
                    int sep_node = separator[i];
                    assert(sep_node >= 0 && sep_node < G_n && "Separator node index out of bounds");
                    assert(max_match_tree.is_separator[sep_node] == false && "Node already marked as separator");
                    max_match_tree.is_separator[sep_node] = true;
                }


                //=========== Step 3: Permute the separator and initialize the
                //decompositon node ============ Compute the permutation for the
                // separator
                std::vector<int> separator_permutation;
                if (!separator.empty()) {
                    Eigen::SparseMatrix<int> separator_graph = compute_sub_graph(Gp, Gi, G_n, separator);
                    // spdlog::info("For decomposition node {} in level {}, #Assigned patches: {}, #Separator nodes / total nodes: {}",
                    //              node_idx, l, assigned_patches.size(), separator.size() * 1.0 / G_n);
                    this->local_permute(separator_graph, separator_permutation);
                }

                // Initialize the decompositon node
                int left_node_idx = decomposition_node_id * 2 + 1;
                if (left_node_idx >= max_match_tree.get_number_of_decomposition_nodes())
                    left_node_idx = -1;

                int right_node_idx = decomposition_node_id * 2 + 2;
                if (right_node_idx >= max_match_tree.get_number_of_decomposition_nodes())
                    right_node_idx = -1;

                if (left_assigned.empty()) {
                    left_node_idx = -1;
                }
                if (right_assigned.empty()) {
                    right_node_idx = -1;
                }


                cur_decomposition_node.init_node(left_node_idx,
                                                 right_node_idx,
                                                 decomposition_node_id,
                                                 decomposition_node_parent_id,
                                                 l,
                                                 separator,
                                                 separator_permutation,
                                                 assigned_patches,
                                                 base_offset_for_decomposition_node_offset);

                //=========== Step 4: Initialize the input of the left and right children for the next wavefront ============
                //Initialize the left child
                if (left_node_idx != -1) {
                    assert(left_node_idx >= 0 && left_node_idx < decomposition_info_stack.size());
                    assert(!left_assigned.empty() && "Left child should have assigned patches");
                    decomposition_info_stack[left_node_idx].decomposition_node_id = left_node_idx;
                    decomposition_info_stack[left_node_idx].decomposition_node_parent_id = decomposition_node_id;
                    decomposition_info_stack[left_node_idx].decomposition_node_patches = left_assigned;
                }

                //Initialize the right child
                if (right_node_idx != -1) {
                    assert(right_node_idx >= 0 && right_node_idx < decomposition_info_stack.size());
                    assert(!right_assigned.empty() && "Right child should have assigned patches");
                    decomposition_info_stack[right_node_idx].decomposition_node_id = right_node_idx;
                    decomposition_info_stack[right_node_idx].decomposition_node_parent_id = decomposition_node_id;
                    decomposition_info_stack[right_node_idx].decomposition_node_patches = right_assigned;
                }
            }
        }
    }
}


void GPUOrdering_V2::init_patches()
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

    node_to_patch.resize(rx.get_num_vertices());
    rx.for_each_vertex(
        rxmesh::HOST,
        [&](const rxmesh::VertexHandle vh) {
            uint32_t node_id       = rx.map_to_global(vh);
            node_to_patch[node_id] = static_cast<int>(vh.patch_id());
        },
        NULL,
        false);
}

void GPUOrdering_V2::step1_create_quotient_graph()
{
    // Given node to patch, first give each separator node a unique patch ID
    // Step 1: assign patch-id -1 to each boundary vertex
    // Count the number of patch ids
    std::unordered_set<int> unique_ids;
    std::map<int, int>      node_per_patch_count;
    for (int i = 0; i < node_to_patch.size(); ++i) {
        unique_ids.insert(node_to_patch[i]);
        node_per_patch_count[node_to_patch[i]]++;
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
    for (int i = 0; i < node_to_patch.size(); ++i) {
        node_to_patch[i] -= patch_offset[node_to_patch[i]];
    }

    // DEBUGING
    int reduced_patches = patch_offset.back();
    spdlog::info("Reduced num of patches: {}", reduced_patches);
    std::unordered_set<int> unique_ids_reduced;
    for (auto& patch : node_to_patch) {
        unique_ids_reduced.insert(patch);
    }
    assert(unique_ids_reduced.size() == unique_ids.size() - reduced_patches);

    // Step 2: create quotient graph
    // Step 2.1: rename the vertices of the quotient graph
    int patch_n = unique_ids_reduced.size();  // Assuming patches start from 0
    // Step 2.2: create patch nodes
    patch_nodes.resize(patch_n);
    for (int i = 0; i < G_n; ++i) {
        int q_id = node_to_patch[i];
        assert(q_id < patch_nodes.size());
        patch_nodes[q_id].nodes.push_back(i);
        patch_nodes[q_id].q_id = q_id;
    }

    // Step 2.3: compute the edge and node weights
    Q_node_weights.clear();
    Q_node_weights.resize(patch_n, 0);
    // Compute the edge and node weights
    int edge_count = 0;
    // Create triplet for sparse matrix creation
    std::vector<Eigen::Triplet<int>> triplets;
    for (int i = 0; i < G_n; ++i) {
        int node_label = node_to_patch[i];
        Q_node_weights[node_label]++;
        for (int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; ++nbr_ptr) {
            int nbr_id    = Gi[nbr_ptr];
            int nbr_label = node_to_patch[nbr_id];
            if (nbr_label == node_label)
                continue;  // Skip boundary vertices and self-loops
            assert(nbr_label != -1);
            triplets.emplace_back(node_label, nbr_label, 1);
        }
    }
    spdlog::info("Found {} edges", triplets.size() / 2);

    // Step 2.4: Create the graph (note that the values are the weight of each
    // edge)
    Q = Eigen::SparseMatrix<int>(patch_n, patch_n);
    Q.setFromTriplets(triplets.begin(), triplets.end());
    Q.makeCompressed();
}

void GPUOrdering_V2::step2_create_hierarchical_partitioning_and_permute()
{
    // Step 2.1: create the hierarchical partitioning
    assert(Q.rows() > 0);
    this->decomposition_max_level = std::ceil(std::log2(this->Q.rows()));
    int total_number_of_decomposition_nodes =
        (1 << (this->decomposition_max_level + 1)) - 1;
    max_match_tree.init_max_match_tree(G_n,
                                       total_number_of_decomposition_nodes);
    spdlog::info("Max match tree creation .. ");
    spdlog::info("Using local permutation {}", local_permute_method);
    spdlog::info("Using separator finding method: {}", separator_finding_method);
    spdlog::info("Using separator refinement method: {}", separator_refinement_method);

    decompose();
    spdlog::info("Max match tree is created.");

    //Apply the offset to the decomposition nodes
    spdlog::info("Applying offset to the decomposition nodes .. ");
    post_order_offset_computation(0, 0);
    spdlog::info("Offset is applied to the decomposition nodes.");

    //Some correctness check ... Comment later (Why nvidia doesn't have NDEBUG? :()
    //Compute total node size
    int total_number_of_dofs = 0;
    std::vector<bool> node_exist(G_n, false);
    for (int i = 0; i < max_match_tree.decomposition_nodes.size(); ++i) {
        total_number_of_dofs +=
            max_match_tree.decomposition_nodes[i].dofs.size();
        for (int j = 0; j < max_match_tree.decomposition_nodes[i].dofs.size(); ++j) {
            int node_id = max_match_tree.decomposition_nodes[i].dofs[j];
            assert(node_exist[node_id] == false);
            node_exist[node_id] = true;
        }
    }
    if(!total_number_of_dofs == G_n) {
        spdlog::error("There are missing nodes after decomposition");
    }
    //Count the number of trues
    int total_number_of_nodes_exist = std::count(node_exist.begin(), node_exist.end(), true);
    if (!total_number_of_nodes_exist == G_n) {
        spdlog::error("There are repetitive nodes in decomposition.");
    }


    //Check the correctness of separator size with flagged separators
    int total_number_of_separator_nodes = 0;
    int max_decomposition_with_separator = (1 << decomposition_max_level) - 1;

    for (int i = 0; i < max_decomposition_with_separator; i++) {
        auto& decomposition_node = max_match_tree.decomposition_nodes[i];
        total_number_of_separator_nodes += decomposition_node.dofs.size();
    }
    int total_flagged_separator_nodes = 0;
    for (int i = 0; i < max_match_tree.is_separator.size(); ++i) {
        if (max_match_tree.is_separator[i]) total_flagged_separator_nodes++;
    }
    assert(total_flagged_separator_nodes == total_flagged_separator_nodes);
    spdlog::info("The ratio of separator nodes: {}",
                 total_flagged_separator_nodes * 1.0 / G_n);
    separator_ratio = total_flagged_separator_nodes * 1.0 / G_n;

    //Check the correctness of the separators

}

void GPUOrdering_V2::assemble_permutation(int decomposition_node_id, std::vector<int>& perm){
    assert(decomposition_node_id >= 0 && decomposition_node_id < max_match_tree.decomposition_nodes.size() && "Invalid decomposition_node_id");
    auto& node = max_match_tree.decomposition_nodes[decomposition_node_id];
    assert(node.dofs.size() == node.local_new_labels.size() && "DOFs and labels size mismatch");
    
    for (int local_node = 0; local_node < node.dofs.size(); local_node++) {
        int global_node = node.dofs[local_node];
        int perm_index = node.local_new_labels[local_node] + node.offset;
        assert(global_node >= 0 && global_node < G_n && "Invalid global node index");
        assert(perm_index >= 0 && perm_index < perm.size() && "Permutation index out of bounds");
        assert(perm[perm_index] == -1 && "Permutation slot already filled - duplicate node!");
        perm[perm_index] = global_node;
      }
    
      if (node.left_node_idx != -1) {
        assert(node.left_node_idx < max_match_tree.decomposition_nodes.size() && "Invalid left child index");
        assemble_permutation(node.left_node_idx, perm);
      }
      if (node.right_node_idx != -1) {
        assert(node.right_node_idx < max_match_tree.decomposition_nodes.size() && "Invalid right child index");
        assemble_permutation(node.right_node_idx, perm);
      }
}


void GPUOrdering_V2::step3_assemble_permutation(std::vector<int>& perm)
{
    perm.clear();
    perm.resize(G_n, -1);
    assemble_permutation(0, perm);

    //DEBUGING
    for (int i = 0; i < perm.size(); i++) {
        assert(perm[i] != -1);
    }
}


void GPUOrdering_V2::compute_permutation(std::vector<int>& perm)
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
    step1_create_quotient_graph();

    // Step 2: Create hierarchical partitioning and compute local permutations
    step2_create_hierarchical_partitioning_and_permute();

    // Step 3: Assemble the final permutation
    step3_assemble_permutation(perm);

    // For now, just return the identity permutation
    // perm.resize(G_n);
    // for (int i = 0; i < G_n; i++) {
    //     perm[i] = i;
    // }
}

}  // namespace RXMESH_SOLVER