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

void GPUOrdering_V2::local_permute(Eigen::SparseMatrix<int>& local_graph,
                                   std::vector<int>&         local_permutation)
{
    if (this->local_permute_method == "metis") {
        local_permute_metis(local_graph, local_permutation);
    } else if (this->local_permute_method == "amd") {
        local_permute_amd(local_graph, local_permutation);
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
    double partition_1_ratio = 0;
    double partition_2_ratio = 0;
    for (int i = 0; i < graph_to_partition_map.size(); ++i) {
        if (graph_to_partition_map[i] == 0)
            partition_1_ratio++;
        else if (graph_to_partition_map[i] == 1)
            partition_2_ratio++;
    }
    partition_1_ratio = partition_1_ratio / graph_to_partition_map.size();
    partition_2_ratio = partition_2_ratio / graph_to_partition_map.size();
    spdlog::info("Partition sizes: {} - {}", partition_1_ratio, partition_2_ratio);
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
                break;
            }
        }
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
        int              base_offset_for_decomposition_node_offset = -1;
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
    decomposition_info_stack[0].base_offset_for_decomposition_node_offset    = 0; //This offset will be the starting point for the offset

    // #pragma omp parallel
    {
        for (int l = 0; l < wavefront_levels; l++) {
            int start_level_idx = (1 << l) - 1;
            int end_level_idx   = (1 << (l + 1)) - 1;
            assert(end_level_idx < total_number_of_decomposition_nodes);
            // #pragma omp for schedule(dynamic)
            for (int node_idx = start_level_idx; node_idx < end_level_idx;
                 node_idx++) {

                // Get the input information for the current node
                int decomposition_node_id = decomposition_info_stack[node_idx].decomposition_node_id;
                int decomposition_node_parent_id = decomposition_info_stack[node_idx].decomposition_node_parent_id;
                std::vector<int>& assigned_patches =decomposition_info_stack[node_idx].decomposition_node_patches;
                auto& cur_decomposition_node = max_match_tree.decomposition_nodes[decomposition_node_id];
                int base_offset_for_decomposition_node_offset = decomposition_info_stack[node_idx].base_offset_for_decomposition_node_offset;
                //+++++++++++++ If there are no patches assigned to the node
                //++++++++++++++++
                if (assigned_patches.size() == 0)
                    continue;

                //+++++++++++++ If it is a leaf node ++++++++++++++++
                if (l == wavefront_levels - 1) {
                    //Add all the nodes in the patches to the dofs
                    std::vector<int> dofs;
                    for (auto& patch_id : assigned_patches) {
                        auto& patch = patch_nodes[patch_id];
                        assert(patch.nodes.size() > 0);
                        for (int i = 0; i < patch.nodes.size(); ++i) {
                            if (!max_match_tree.is_separator[patch.nodes[i]]) {
                                dofs.push_back(patch.nodes[i]);
                            }
                        }
                    }

                    if(assigned_patches.size() > 1) {
                        //print a warning that in the leaf node, it is better to have only one patch
                        spdlog::warn("In the leaf decomposition node, it is better to not have more than one patch");
                        spdlog::warn("For node {} in level {}: #Assigned patches: {}", node_idx, l, assigned_patches.size());
                    }
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
                        auto& patch = patch_nodes[patch_id];
                        assert(patch.nodes.size() > 0);
                        for (int i = 0; i < patch.nodes.size(); ++i) {
                            if (!max_match_tree.is_separator[patch.nodes[i]]) {
                                dofs.push_back(patch.nodes[i]);
                            }
                        }
                    }
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
                for (int i = 0; i < local_node_to_partition.size(); i++) {
                    int global_patch_id = assigned_patches[i];
                    //Check if the patch has non-separator nodes - IMPORTANT
                    bool is_valid_patch = false;
                    for (int j = 0; j < patch_nodes[global_patch_id].nodes.size(); ++j) {
                        if (!max_match_tree.is_separator[patch_nodes[global_patch_id].nodes[j]]) {
                            is_valid_patch = true;
                            break;
                        }
                    }
                    if (!is_valid_patch) continue;
                    if (local_node_to_partition[i] == 0) {
                        left_assigned.push_back(global_patch_id);
                    } else {
                        right_assigned.push_back(global_patch_id);
                    }
                }

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
                this->find_separator_basic(graph_to_partition_map, separator);

                // Mark the separator nodes as true
                assert(max_match_tree.is_separator.size() == G_n);
                for (int i = 0; i < separator.size(); ++i) {
                    assert(max_match_tree.is_separator[separator[i]] == false);
                    max_match_tree.is_separator[separator[i]] = true;
                }


                //=========== Step 3: Permute the separator and initialize the
                //decompositon node ============ Compute the permutation for the
                // separator
                Eigen::SparseMatrix<int> separator_graph =
                    compute_sub_graph(Gp, Gi, G_n, separator);
                spdlog::info("For decomposition node {} in level {}, #Assigned patches: {}, #Separator nodes / total nodes: {}",
                             node_idx, l, assigned_patches.size(), separator.size() * 1.0 / G_n);
                std::vector<int> separator_permutation;
                this->local_permute(separator_graph, separator_permutation);
                // Initialize the decompositon node
                int left_node_idx = decomposition_node_id * 2 + 1;
                if (left_node_idx >= max_match_tree.get_number_of_decomposition_nodes())
                    left_node_idx = -1;

                int right_node_idx = decomposition_node_id * 2 + 2;
                if (right_node_idx >= max_match_tree.get_number_of_decomposition_nodes())
                    right_node_idx = -1;

                //It will not destroy correctness, but I believe that it should hold true
                //We can delete it later if it is not needed
                assert(left_assigned.size() > 0);
                assert(right_assigned.size() > 0);

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
                decomposition_info_stack[left_node_idx].decomposition_node_id = left_node_idx;
                decomposition_info_stack[left_node_idx].decomposition_node_parent_id = decomposition_node_id;
                decomposition_info_stack[left_node_idx].decomposition_node_patches = left_assigned;
                decomposition_info_stack[left_node_idx].base_offset_for_decomposition_node_offset = base_offset_for_decomposition_node_offset;
                //Initialize the right child
                decomposition_info_stack[right_node_idx].decomposition_node_id = right_node_idx;
                decomposition_info_stack[right_node_idx].decomposition_node_parent_id = decomposition_node_id;
                decomposition_info_stack[right_node_idx].decomposition_node_patches = right_assigned;
                decomposition_info_stack[right_node_idx].base_offset_for_decomposition_node_offset = base_offset_for_decomposition_node_offset + left_assigned.size();
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
        std::pow(2, this->decomposition_max_level + 1) - 1;
    max_match_tree.init_max_match_tree(G_n,
                                       total_number_of_decomposition_nodes);
    spdlog::info("Max match tree creation .. ");
    decompose();
    spdlog::info("Max match tree is created.");

    //Some correctness check ... Comment later (Why nvidia doesn't have NDEBUG? :()
    //Compute total node size
    int total_number_of_dofs = 0;
    for (int i = 0; i < max_match_tree.decomposition_nodes.size(); ++i) {
        total_number_of_dofs +=
            max_match_tree.decomposition_nodes[i].dofs.size();
    }
    assert(total_number_of_dofs == G_n);

    //Check the correctness of separator size with flagged separators

    //Check the correctness of the separators

}

void GPUOrdering_V2::step3_assemble_permutation()
{
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
    step3_assemble_permutation();

    // For now, just return the identity permutation
    perm.resize(G_n);
    for (int i = 0; i < G_n; i++) {
        perm[i] = i;
    }
}

}  // namespace RXMESH_SOLVER