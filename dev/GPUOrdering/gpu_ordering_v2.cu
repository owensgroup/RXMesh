//
// Created by behrooz on 2025-10-07.
//
#include <amd.h>
#include <metis.h>
#include <cassert>
#include <cmath>
#include <chrono>
#include <unordered_set>
#include "gpu_ordering_v2.h"
#include "min_vertex_cover_bipartite.h"
#include "rxmesh/rxmesh_static.h"
#include "spdlog/spdlog.h"

namespace RXMESH_SOLVER {

GPUOrdering_V2::GPUOrdering_V2() : _Gp(nullptr), _Gi(nullptr), _G_n(0), _G_nnz(0)
{
}

GPUOrdering_V2::~GPUOrdering_V2()
{
}

void GPUOrdering_V2::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->_Gp    = Gp;
    this->_Gi    = Gi;
    this->_G_n   = G_N;
    this->_G_nnz = NNZ;
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
                                         std::vector<int>& local_permutation)
{
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


void GPUOrdering_V2::compute_local_quotient_graph(
    int tree_node_idx,///<[in] The index of the current decomposition node
    std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
    Eigen::SparseMatrix<int>& Q,///<[out] The local quotient graph
    std::vector<int>& Q_node_weights,///<[out] The node weights of the local quotient graph
    std::vector<int>& q_local_to_global_map///<[in/out] The map from current assigned quotient nodes to tree nodes
){
    //Find the mapping between local partitions and global partitions
    std::set<int> unique_global_partitions;
    std::vector<int> Q_global_to_local_map(this->_num_patches, -1);
    for(int i = 0; i < assigned_g_nodes.size(); i++) {
        int dof = assigned_g_nodes[i];
        int q_global_node = this->_g_node_to_patch[dof];
        unique_global_partitions.insert(q_global_node);
    }

    int cnt = 0;
    for (auto& part: unique_global_partitions) {
        Q_global_to_local_map[part] = cnt++;
    }

    auto is_in_tree = [&](int g_node_id) -> bool {
        if(this->_decomposition_tree.is_separator(g_node_id)) return false;
        int q_node_id = this->_g_node_to_patch[g_node_id];
        if(this->_decomposition_tree.q_node_to_tree_node[q_node_id] == tree_node_idx)
            return true;
        return false;
    };

    q_local_to_global_map.resize(unique_global_partitions.size(), -1);
    std::copy(unique_global_partitions.begin(), unique_global_partitions.end(), q_local_to_global_map.begin());

    //Create the local Q
    Q.resize(unique_global_partitions.size(), unique_global_partitions.size());
    Q_node_weights.resize(unique_global_partitions.size(), 0);
    std::vector<Eigen::Triplet<int>> triplets;
    for(int i = 0; i < assigned_g_nodes.size(); i++) {
        int g_node = assigned_g_nodes[i];
        int node_q_id = this->_g_node_to_patch[g_node];
        assert(node_q_id != -1);
        int local_q = Q_global_to_local_map[node_q_id];
        Q_node_weights[local_q]++;
        for(int nbr_ptr = this->_Gp[g_node]; nbr_ptr < this->_Gp[g_node + 1]; nbr_ptr++) {
            int nbr_id = this->_Gi[nbr_ptr];
            if(!is_in_tree(nbr_id)) continue;
            int q_nbr_id = this->_g_node_to_patch[nbr_id];
            int local_nbr_q = Q_global_to_local_map[q_nbr_id];
            if (local_nbr_q == local_q) continue;
            triplets.push_back(Eigen::Triplet<int>(local_q, local_nbr_q, 1));
        }
    }
    Q.setFromTriplets(triplets.begin(), triplets.end());//The edge weights are the values of the edges
    Q.makeCompressed();
}

void GPUOrdering_V2::compute_bipartition(
    Eigen::SparseMatrix<int>& Q,
    std::vector<int>&         Q_node_weights,
    std::vector<int>&         Q_partition_map)
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

    idx_t   nvtxs  = Q.rows();
    idx_t   ncon   = 1;
    idx_t*  vwgt   = NULL;
    idx_t*  vsize  = NULL;
    idx_t   nparts = 2;
    real_t* tpwgts = NULL;
    real_t* ubvec  = NULL;
    idx_t   objval = 0;

    Q_partition_map.resize(Q.rows(), 0);

    int metis_status = METIS_PartGraphKway(&nvtxs,
                                           &ncon,
                                           Q.outerIndexPtr(),
                                           Q.innerIndexPtr(),
                                           Q_node_weights.data(),
                                           vsize,
                                           Q.valuePtr(),
                                           &nparts,
                                           tpwgts,
                                           ubvec,
                                           options,
                                           &objval,
                                           Q_partition_map.data());

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
}


void GPUOrdering_V2::two_way_Q_partition(
    int tree_node_idx,///<[in] The index of the current decomposition node
    std::vector<int>& assigned_g_nodes///<[in] Assigned G nodes for current decomposition
){
    //Compute local quotient graph
    Eigen::SparseMatrix<int> local_Q;
    std::vector<int> local_Q_node_to_global_Q_node;
    std::vector<int> local_Q_node_weights;
    compute_local_quotient_graph(tree_node_idx,
        assigned_g_nodes,
        local_Q,
        local_Q_node_weights,
        local_Q_node_to_global_Q_node);

    std::vector<int> two_way_q_partition_map;
    compute_bipartition(local_Q, local_Q_node_weights, two_way_q_partition_map);

    //Converting the q map to node map
    for(int i = 0; i < local_Q_node_to_global_Q_node.size(); i++) {
        int q_global_node = local_Q_node_to_global_Q_node[i];
        //Assign the q to left and right nodes
        this->_decomposition_tree.q_node_to_tree_node[q_global_node] = tree_node_idx * 2 + two_way_q_partition_map[i] + 1;
    }
}

void GPUOrdering_V2::find_separator_superset(
    std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
    std::vector<int>& separator_superset///<[out] The superset of separator nodes
){

    auto get_partition_id = [&](int g_node_id) -> int {
        int q_node_id = this->_g_node_to_patch[g_node_id];
        return this->_decomposition_tree.q_node_to_tree_node[q_node_id];
    };

    separator_superset.clear();
    for(int i = 0; i < assigned_g_nodes.size(); i++) {
        int g_node = assigned_g_nodes[i];
        int partition_id = get_partition_id(g_node);
        for (int nbr_ptr = this->_Gp[g_node]; nbr_ptr < this->_Gp[g_node + 1]; ++nbr_ptr) {
            int nbr_id = this->_Gi[nbr_ptr];
            if (g_node >= nbr_id)
                continue;
            if(this->_decomposition_tree.is_separator(nbr_id)) continue;
            int nbr_partition_id = get_partition_id(nbr_id);
            if(partition_id == nbr_partition_id) continue;
            separator_superset.push_back(g_node);
            separator_superset.push_back(nbr_id);
        }
    }
    //Erase repetitive nodes
    std::sort(separator_superset.begin(), separator_superset.end());
    separator_superset.erase(std::unique(separator_superset.begin(), separator_superset.end()), separator_superset.end());
}


void GPUOrdering_V2::refine_bipartate_separator(
    int parent_node_id,
    std::vector<int>& separator_superset)
{
    // The boundary nodes to be used for extracting bipartite graph
    if (separator_superset.size() == 0) {
        return;
    }
    // Extract the bipartite graph from the separator nodes
    // Overall flow:
    // Step 1: Extract the bipartite graph from the separator nodes
    // Step 2: Use the min_vertex_cover_bipartite to compute the max matching
    // and thus min vertex cover Step 4: Update the separator with min vertex
    // cover

#ifdef DEBUG
    //Make sure that there is only two partitions
    std::unordered_set<int> partition_unique_ids;
    for(int i = 0; i < separator_superset.size(); i++) {
        int partition_id = this->_g_node_to_patch[separator_superset[i]];
        int tree_node_id = this->_decomposition_tree.q_node_to_tree_node[partition_id];
        partition_unique_ids.insert(tree_node_id);
    }
    assert(partition_unique_ids.size() == 2 && "There should be only two partitions");
#endif

    //Step 1: Extract the bipartite graph from the separator nodes
    Eigen::SparseMatrix<int>         bipartite_graph;
    std::vector<Eigen::Triplet<int>> triplets;
    // mapping from separator nodes to local nodes
    std::map<int, int> separator_node_to_local_node;
    for (int i = 0; i < separator_superset.size(); i++) {
        separator_node_to_local_node[separator_superset[i]] = i;
    }

    auto get_tree_id = [&](int g_node) -> int {
        int partition_id = this->_g_node_to_patch[g_node];
        int tree_node_id = this->_decomposition_tree.q_node_to_tree_node[partition_id];
        return tree_node_id;
    };

    // Extract the bipartite graph
    for (auto& g_node : separator_superset) {
        assert(!this->_decomposition_tree.is_separator(g_node));
        int g_node_tree_id = get_tree_id(g_node);
        for (int nbr_ptr = this->_Gp[g_node]; nbr_ptr < this->_Gp[g_node + 1]; nbr_ptr++) {
            int nbr_id = this->_Gi[nbr_ptr];
            //Not in the separator set
            if(separator_node_to_local_node.find(nbr_id) == separator_node_to_local_node.end()) continue;
            //They should not be in the same partition
            int nbr_node_tree_id = get_tree_id(nbr_id);
            if (g_node_tree_id == nbr_node_tree_id) continue;;
            triplets.push_back(Eigen::Triplet<int>(separator_node_to_local_node[g_node], separator_node_to_local_node[nbr_id], 1));
            triplets.push_back(Eigen::Triplet<int>(separator_node_to_local_node[nbr_id], separator_node_to_local_node[g_node], 1));
        }
    }
    bipartite_graph.resize(separator_superset.size(), separator_superset.size());
    bipartite_graph.setFromTriplets(triplets.begin(), triplets.end());
    bipartite_graph.makeCompressed();

    //=========== Step 3: Use the min_vertex_cover_bipartite to compute the max
    //matching and thus min vertex cover ============
    std::vector<int> local_graph_to_partition(separator_superset.size(), -1);
    for(int i = 0; i < separator_superset.size(); i++) {
        int tree_node_id = get_tree_id(separator_superset[i]);
        if (parent_node_id * 2 + 1 == tree_node_id) {
            local_graph_to_partition[i] = 0;
        } else {
            local_graph_to_partition[i] = 1;
            assert(parent_node_id * 2 + 2 == tree_node_id);
        }
    }

#ifdef DEBUG
    for (int node = 0; node < bipartite_graph.rows(); node++) {
        for (int nbr_ptr = bipartite_graph.outerIndexPtr()[node];
             nbr_ptr < bipartite_graph.outerIndexPtr()[node + 1];
             ++nbr_ptr) {
            int nbr_id = bipartite_graph.innerIndexPtr()[nbr_ptr];
            int partition_id = local_graph_to_partition[node];
            int nbr_partition_id = local_graph_to_partition[nbr_id];
            if (partition_id == nbr_partition_id) {
                spdlog::error("The graph is not bipartite");
                assert(false);
            }
        }
    }
#endif


    RXMESH_SOLVER::MinVertexCoverBipartite solver(
        bipartite_graph.rows(),
        bipartite_graph.outerIndexPtr(),
        bipartite_graph.innerIndexPtr(),
        local_graph_to_partition);
    std::vector<int> min_vertex_cover = solver.compute_min_vertex_cover();
    //=========== Step 4: Update the separator nodes ============
    int initial_separator_size = separator_superset.size();
    std::vector<int> tmp = separator_superset;
    separator_superset.clear();
    //Converting the local min_vertex_cover to global one
    for (int i = 0; i < min_vertex_cover.size(); i++) {
        separator_superset.push_back(tmp[min_vertex_cover[i]]);
    }
    int final_separator_size = separator_superset.size();
#ifdef DEBUG
    spdlog::info("Bipartite graph refinement reduced separator from {} to {}",
                 initial_separator_size,
                 final_separator_size);
    //Check whether the separator nodes are repetitive or not
    std::vector<bool> visited(_G_n, false);
    for (int i = 0; i < separator_superset.size(); i++) {
        assert(visited[i] == false);
        visited[i] = true;
    }
#endif
}


void GPUOrdering_V2::three_way_G_partition(
    int tree_node_idx,
    std::vector<int>& assigned_g_nodes,
    std::vector<int>& separator_g_nodes)
{
    find_separator_superset(assigned_g_nodes, separator_g_nodes);
    refine_bipartate_separator(tree_node_idx, separator_g_nodes);
}


int GPUOrdering_V2::post_order_offset_computation(int offset,
                                                  int decomposition_node_id)
{
    assert(decomposition_node_id <
           this->_decomposition_tree.get_number_of_decomposition_nodes());
    auto& decomposition_node =
        this->_decomposition_tree.decomposition_nodes[decomposition_node_id];
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
    offset = separator_offset + decomposition_node.assigned_g_nodes.size();
    return offset;
}

void GPUOrdering_V2::decompose()
{
    // Think of this as the input of decompose function in recursive manner
    struct DecompositionInfo
    {
        int              decomposition_node_id        = -1;
        int              decomposition_node_parent_id = -1;
        std::vector<int> assigned_g_nodes;
    };

    // Initialize the first call for the decomposition info stack
    std::vector<DecompositionInfo> decomposition_info_stack(
        this->_decomposition_tree.get_number_of_decomposition_nodes());
    decomposition_info_stack[0].decomposition_node_id        = 0;
    decomposition_info_stack[0].decomposition_node_parent_id = -1;
    decomposition_info_stack[0].assigned_g_nodes.resize(_G_n);
    for(int i = 0; i < _G_n; i++) {
        decomposition_info_stack[0].assigned_g_nodes[i] = i;
    }

    //Show where patches are in the tree during decomposition
    _decomposition_tree.q_node_to_tree_node.resize(this->_num_patches, 0);
    _decomposition_tree.g_node_to_tree_node.resize(this->_G_n, 0);
    #pragma omp parallel
    {
        for (int l = 0; l < this->_decomposition_tree.decomposition_level; l++) {
            int start_level_idx = (1 << l) - 1;
            int end_level_idx   = (1 << (l + 1)) - 1;
            assert(end_level_idx < this->_decomposition_tree.get_number_of_decomposition_nodes());
            #pragma omp for schedule(dynamic)
            for (int node_idx = start_level_idx; node_idx < end_level_idx;
                 ++node_idx) {
                // Get the input information for the current node
#ifndef NDEBUG
                spdlog::info("*********** Working on node {} ***********", node_idx);
#endif
                int decomposition_node_id = decomposition_info_stack[node_idx].decomposition_node_id;
                int decomposition_node_parent_id = decomposition_info_stack[node_idx].decomposition_node_parent_id;
                std::vector<int>& assigned_g_nodes = decomposition_info_stack[node_idx].assigned_g_nodes;
                auto& cur_decomposition_node =this->_decomposition_tree.decomposition_nodes[decomposition_node_id];

                //++++++ if it is not decomposable, skip it ******
                if(assigned_g_nodes.size() == 0) continue;

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
                if (l == this->_decomposition_tree.decomposition_level - 1) {
                    // Add all the nodes in the patches to the dofs
                    cur_decomposition_node.init_node(
                        -1,
                        -1,
                        decomposition_node_id,
                        decomposition_node_parent_id,
                        l,
                        assigned_g_nodes);
                    continue;
                }

                //+++++++++++++ If it is not a leaf node ++++++++++++++++
                // Overall flow:
                // Step 1: Compute two equal size partitions from the assigned dofs
                // Step 2: Find the separator nodes of the two partitions
                // Step 3: Initialize the input of the left and right

                // Step 1: Compute two equal size partitions from the assigned dofs

                std::vector<int> two_way_dof_partition_map;
                this->two_way_Q_partition(node_idx, assigned_g_nodes);
                // Step 2: Find the separator nodes of the two partitions
                std::vector<int> left_assigned_g_nodes, right_assigned_g_nodes;
                std::vector<int> separator_g_nodes;
                this->three_way_G_partition(node_idx, assigned_g_nodes, separator_g_nodes);

                left_assigned_g_nodes.clear();
                right_assigned_g_nodes.clear();

                //flag the separator nodes in the decomposition tree
                for(int i = 0; i < separator_g_nodes.size(); i++) {
                    assert(this->_decomposition_tree.is_separator(separator_g_nodes[i]) == false);
                    this->_decomposition_tree.set_separator(separator_g_nodes[i]);
                    this->_decomposition_tree.g_node_to_tree_node[separator_g_nodes[i]] = node_idx;
                }

                //Assign the nodes to the left and right assigned dofs
                for(int i = 0; i < assigned_g_nodes.size(); i++) {
                    int g_node = assigned_g_nodes[i];
                    int q_node = this->_g_node_to_patch[g_node];
                    if(this->_decomposition_tree.is_separator(g_node)) continue;
                    int q_partition = this->_decomposition_tree.q_node_to_tree_node[q_node];
                    if(q_partition == node_idx * 2 + 1) {
                        left_assigned_g_nodes.push_back(g_node);
                    } else {
                        right_assigned_g_nodes.push_back(g_node);
                        assert(q_partition == node_idx * 2 + 2);
                    }
                }

#ifndef NDEBUG
                spdlog::info("The left size is: {}, the right size is: {} and the separator size is {}: ",
                    left_assigned_g_nodes.size(),
                    right_assigned_g_nodes.size(),
                    separator_g_nodes.size());
#endif

                // Initialize the input of the left and right children for the
                // next wavefront
                int left_node_idx  = decomposition_node_id * 2 + 1;
                int right_node_idx = decomposition_node_id * 2 + 2;
                if (left_assigned_g_nodes.empty()) {
                    left_node_idx = -1;
                }
                if (right_assigned_g_nodes.empty()) {
                    right_node_idx = -1;
                }

                cur_decomposition_node.init_node(left_node_idx,
                                                 right_node_idx,
                                                 decomposition_node_id,
                                                 decomposition_node_parent_id,
                                                 l,
                                                 separator_g_nodes);

                if (left_node_idx != -1) {
                    assert(left_node_idx >= 0 &&
                           left_node_idx < decomposition_info_stack.size());
                    assert(!left_assigned_g_nodes.empty() &&
                           "Left child should have assigned patches");
                    decomposition_info_stack[left_node_idx]
                        .decomposition_node_id = left_node_idx;
                    decomposition_info_stack[left_node_idx]
                        .decomposition_node_parent_id = decomposition_node_id;
                    decomposition_info_stack[left_node_idx].assigned_g_nodes =
                        left_assigned_g_nodes;
                }

                // Initialize the right child
                if (right_node_idx != -1) {
                    assert(right_node_idx >= 0 &&
                           right_node_idx < decomposition_info_stack.size());
                    assert(!right_assigned_g_nodes.empty() &&
                           "Right child should have assigned patches");
                    decomposition_info_stack[right_node_idx]
                        .decomposition_node_id = right_node_idx;
                    decomposition_info_stack[right_node_idx]
                        .decomposition_node_parent_id = decomposition_node_id;
                    decomposition_info_stack[right_node_idx].assigned_g_nodes =
                        right_assigned_g_nodes;
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
    rxmesh::RXMeshStatic rx(this->fv, "", this->_patch_size);

    spdlog::info(
        "RXMesh initialized with {} vertices, {} edges, {} faces, {} patches",
        rx.get_num_vertices(),
        rx.get_num_edges(),
        rx.get_num_faces(),
        rx.get_num_patches());

    this->_g_node_to_patch.resize(rx.get_num_vertices());
    rx.for_each_vertex(
        rxmesh::HOST,
        [&](const rxmesh::VertexHandle vh) {
            uint32_t node_id       = rx.map_to_global(vh);
            this->_g_node_to_patch[node_id] = static_cast<int>(vh.patch_id());
        },
        NULL,
        false);
}


void GPUOrdering_V2::compute_sub_graph(
    std::vector<int>&         nodes,
    Eigen::SparseMatrix<int>& local_graph) const
{
    // Compute global node to local node mapping
    std::vector<int> global_to_local(this->_G_n, -1);
    for (int i = 0; i < nodes.size(); ++i) {
        assert(global_to_local[nodes[i]] == -1);
        global_to_local[nodes[i]] = i;
    }

    // Compute triplets for the sub graph
    std::vector<Eigen::Triplet<int>> triplets;
    for (int i = 0; i < nodes.size(); ++i) {
        int node_id = nodes[i];
        for (int nbr_ptr = this->_Gp[node_id]; nbr_ptr < this->_Gp[node_id + 1];
             ++nbr_ptr) {
            int nbr_id = this->_Gi[nbr_ptr];
            if (node_id == nbr_id) {
                spdlog::info("node {} has a diagonal edge to itself", node_id);
                spdlog::info("  Gp[{}] = {}, Gp[{}] = {}", node_id, this->_Gp[node_id], node_id + 1, this->_Gp[node_id + 1]);
                spdlog::info("  nbr_ptr = {}, Gi[{}] = {}", nbr_ptr, nbr_ptr, nbr_id);
                spdlog::info("  Full adjacency list for node {}:", node_id);
                for (int ptr = this->_Gp[node_id]; ptr < this->_Gp[node_id + 1]; ++ptr) {
                    spdlog::info("    Gi[{}] = {}", ptr, this->_Gi[ptr]);
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


void GPUOrdering_V2::step1_compute_node_to_patch()
{
    // Given node to patch, first give each separator node a unique patch ID
    // Step 1: assign patch-id -1 to each boundary vertex
    // Count the number of patch ids
    std::unordered_set<int> unique_ids;
    std::map<int, int>      node_per_patch_count;
    for (int i = 0; i < this->_g_node_to_patch.size(); ++i) {
        unique_ids.insert(this->_g_node_to_patch[i]);
        node_per_patch_count[this->_g_node_to_patch[i]]++;
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
    for (int i = 0; i < this->_g_node_to_patch.size(); ++i) {
        this->_g_node_to_patch[i] -= patch_offset[this->_g_node_to_patch[i]];
        unique_ids_reduced.insert(this->_g_node_to_patch[i]);
    }
    spdlog::info(
        "Number of initial patches before and after reduction: {} -> {}",
        unique_ids.size(),
        unique_ids_reduced.size());
    this->_num_patches = unique_ids_reduced.size();
    assert(this->_g_node_to_patch.size() == this->_G_n);
}

void GPUOrdering_V2::step2_create_decomposition_tree()
{
    // Step 2.1: create the hierarchical partitioning
    int num_levels = std::ceil(std::log2(this->_num_patches));
    int total_number_of_decomposition_nodes = (1 << (num_levels + 1)) - 1;
    this->_decomposition_tree.init_decomposition_tree(
        total_number_of_decomposition_nodes, num_levels, this->_G_n);
    spdlog::info("Decomposition tree creation .. ");
    spdlog::info("Number of decomposition levels: {}", num_levels);
    spdlog::info("Using local permutation {}", this->local_permute_method);

    this->decompose();
    spdlog::info("Decomposition tree is created.");

#ifndef NDEBUG
    //Check to see if all the nodes exist
    spdlog::info("Checking decomposition validation.");
    std::vector<bool> is_visited(this->_G_n, false);
    for(auto& node : this->_decomposition_tree.decomposition_nodes) {
        if (node.assigned_g_nodes.empty())
            continue;
        for (auto& g_node : node.assigned_g_nodes) {
            assert(is_visited[g_node] == false);
            is_visited[g_node] = true;
        }
    }
    for (int i = 0; i < is_visited.size(); ++i) {
        assert(is_visited[i] == true);
    }
    spdlog::info("Decomposition contains all the nodes");

    //Check for whether separators are valid
#endif
}

void GPUOrdering_V2::step3_CPU_compute_local_permutations()
{
    spdlog::info("Computing local permutations .. ");
    #pragma omp parallel for
    for (int i = 0; i < this->_decomposition_tree.decomposition_nodes.size(); i++) {
        auto& node = this->_decomposition_tree.decomposition_nodes[i];
        if (node.assigned_g_nodes.empty())
            continue;
        Eigen::SparseMatrix<int> graph;
        this->compute_sub_graph(node.assigned_g_nodes, graph);
        std::vector<int> local_permutation;
        local_permute(graph, local_permutation);
        node.set_local_permutation(local_permutation);
    }
    spdlog::info("Local permutations are computed.");
}


void GPUOrdering_V2::step3_GPU_compute_local_permutations()
{

}

void GPUOrdering_V2::step4_assemble_final_permutation(std::vector<int>& perm)
{
    // Apply the offset to the decomposition nodes
    spdlog::info("Applying offset to the decomposition nodes .. ");
    post_order_offset_computation(0, 0);
    spdlog::info("Offset is applied to the decomposition nodes.");
    spdlog::info("Assembling the final permutation .. ");
    perm.clear();
    perm.resize(this->_G_n, -1);
    for (auto& node : this->_decomposition_tree.decomposition_nodes) {
        if (node.assigned_g_nodes.empty())
            continue;
        for (int local_node = 0; local_node < node.assigned_g_nodes.size(); local_node++) {
            int global_node = node.assigned_g_nodes[local_node];
            int perm_index  = node.local_new_labels[local_node] + node.offset;
            assert(global_node >= 0 && global_node < this->_G_n &&
                   "Invalid global node index");
            assert(perm_index >= 0 && perm_index < perm.size() &&
                   "Permutation index out of bounds");
            assert(perm[perm_index] == -1 &&
                   "Permutation slot already filled - duplicate node!");
            perm[perm_index] = global_node;
        }
    }
    spdlog::info("Final permutation is assembled.");
}


void GPUOrdering_V2::compute_permutation(std::vector<int>& perm)
{
    if (this->_Gp == nullptr || this->_Gi == nullptr || this->_G_n == 0 || this->_G_nnz == 0) {
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

    // Step 1: Compute node to patch map
    auto start_time = std::chrono::high_resolution_clock::now();
    step1_compute_node_to_patch();
    auto end_time = std::chrono::high_resolution_clock::now();
    node_to_patch_time = std::chrono::duration<double>(end_time - start_time).count();
    spdlog::info("Step 1 (node to patch) completed in {:.6f} seconds", node_to_patch_time);

    // Step 2: Create decomposition tree
    start_time = std::chrono::high_resolution_clock::now();
    step2_create_decomposition_tree();
    end_time = std::chrono::high_resolution_clock::now();
    decompose_time = std::chrono::duration<double>(end_time - start_time).count();
    spdlog::info("Step 2 (decomposition tree) completed in {:.6f} seconds", decompose_time);

    // Step 3: Compute the local permutations
    start_time = std::chrono::high_resolution_clock::now();
    step3_CPU_compute_local_permutations();
    end_time = std::chrono::high_resolution_clock::now();
    local_permute_time = std::chrono::duration<double>(end_time - start_time).count();
    spdlog::info("Step 3 (local permutations) completed in {:.6f} seconds", local_permute_time);

    // Step 4: Assemble the final permutation
    start_time = std::chrono::high_resolution_clock::now();
    step4_assemble_final_permutation(perm);
    end_time = std::chrono::high_resolution_clock::now();
    assemble_time = std::chrono::duration<double>(end_time - start_time).count();
    spdlog::info("Step 4 (assemble permutation) completed in {:.6f} seconds", assemble_time);
}


}  // namespace RXMESH_SOLVER