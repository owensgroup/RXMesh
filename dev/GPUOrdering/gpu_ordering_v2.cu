//
// Created by behrooz on 2025-10-07.
//
#include "gpu_ordering_v2.h"
#include "spdlog/spdlog.h"
#include "rxmesh/rxmesh_static.h"
#include <unordered_set>
#include <metis.h>
#include <amd.h>



namespace RXMESH_SOLVER {

GPUOrdering_V2::GPUOrdering_V2()
    : Gp(nullptr), Gi(nullptr), G_n(0), G_nnz(0), Q_n(0)
{
}

GPUOrdering_V2::~GPUOrdering_V2()
{
}

void GPUOrdering_V2::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp = Gp;
    this->Gi = Gi;
    this->G_n = G_N;
    this->G_nnz = NNZ;
}


void GPUOrdering_V2::setMesh(const double* V_data, int V_rows, int V_cols,
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


void GPUOrdering_V2::local_permute_metis(Eigen::SparseMatrix<int>& local_graph, std::vector<int> & local_permutation){
    idx_t N = local_graph.rows();
    idx_t NNZ = local_graph.nonZeros();
    local_permutation.resize(N);
    if (NNZ == 0) {
      assert(M_n != 0);
      for (int i = 0; i < N; i++) {
        local_permutation[i] = i;
      }
      return;
    }
    
    std::vector<int> tmp(N);
    METIS_NodeND(&N, local_graph.outerIndexPtr(), local_graph.innerIndexPtr(), NULL, NULL, local_permutation.data(), tmp.data());
}

void GPUOrdering_V2::local_permute_amd(Eigen::SparseMatrix<int>& local_graph, std::vector<int> & local_permutation){
    idx_t N = local_graph.rows();
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
    amd_order(N, local_graph.outerIndexPtr(), local_graph.innerIndexPtr(), local_permutation.data(), nullptr, nullptr);
}

void GPUOrdering_V2::local_permute(Eigen::SparseMatrix<int>& local_graph, std::vector<int> & local_permutation){
    if(this->local_permute_method == "metis") {
        local_permute_metis(local_graph, local_permutation);
    } else if(this->local_permute_method == "amd") {
        local_permute_amd(local_graph, local_permutation);
    } else {
        spdlog::error("Invalid local permutation method: {}", this->local_permute_method);
        return;
    }
}

void GPUOrdering_V2::compute_bipartition(Eigen::SparseMatrix<int>& quotient_graph, std::vector<int>& quotient_graph_node_weights,
    std::vector<int>& node_to_partition){
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

void GPUOrdering_V2::compute_sub_graph(Eigen::SparseMatrix<int>& graph,
    std::vector<int>& graph_node_weights,
    Eigen::SparseMatrix<int>& sub_graph,
    std::vector<int>& local_node_weights,
    std::vector<int>& nodes) const{
    //Compute global node to local node mapping
    std::vector<int> global_to_local(graph.rows(), -1);
    local_node_weights.resize(nodes.size(), 0);
    for(int i = 0; i < nodes.size(); ++i) {
        global_to_local[nodes[i]] = i;
        local_node_weights[i] = graph_node_weights[nodes[i]];
    }

    //Compute triplets for the sub graph
    std::vector<Eigen::Triplet<int>> triplets;
    for(int i = 0; i < nodes.size(); ++i) {
        for(int nbr_ptr = graph.outerIndexPtr()[nodes[i]]; nbr_ptr < graph.outerIndexPtr()[nodes[i] + 1]; ++nbr_ptr) {
            int nbr_id = graph.innerIndexPtr()[nbr_ptr];
            if(global_to_local[nbr_id] == -1) continue;
            triplets.emplace_back(i, global_to_local[nbr_id], graph.valuePtr()[nbr_ptr]);
            assert(i != global_to_local[nbr_id]);
            assert(i == global_to_local[i]);
        }
    }

    //Form the sub graph
    sub_graph.resize(nodes.size(), nodes.size());
    sub_graph.setFromTriplets(triplets.begin(), triplets.end());
    sub_graph.makeCompressed();
}

Eigen::SparseMatrix<int> GPUOrdering_V2::compute_sub_graph(int* Gp, int* Gi, int G_N, int NNZ, std::vector<int>& nodes) const{
    //Compute global node to local node mapping
    std::sort(nodes.begin(), nodes.end());
    std::vector<int> global_to_local(G_N, -1);
    for(int i = 0; i < nodes.size(); ++i) {
        global_to_local[nodes[i]] = i;
    }
    
    //Compute triplets for the sub graph
    std::vector<Eigen::Triplet<int>> triplets;
    for(int i = 0; i < nodes.size(); ++i) {
        for(int nbr_ptr = Gp[nodes[i]]; nbr_ptr < Gp[nodes[i] + 1]; ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            if(global_to_local[nbr_id] == -1) continue;
            assert(i != global_to_local[nbr_id]);
            assert(i == global_to_local[i]);
            triplets.emplace_back(i, global_to_local[nbr_id], 1);
        }
    }
    //Form the sub graph
    Eigen::SparseMatrix<int> sub_graph(nodes.size(), nodes.size());
    sub_graph.setFromTriplets(triplets.begin(), triplets.end());
    sub_graph.makeCompressed();
    return sub_graph;
}


void GPUOrdering_V2::decompose(
    int decomposition_node_id,   ///[in] HMD node id
    int decomposition_node_parent_id, /// <[in] HMD node's parent id
    int decomposition_level,     ///<[in] The current level that we are dissecting
    std::vector<int> & decomposition_node_patches) {
  assert(decomposition_node_patches.size() != 0);

  auto &cur_node = max_match_tree.decomposition_nodes[decomposition_node_id];
  //+++++++++++++ When we get to the leaf level ++++++++++++++++
  if (decomposition_level == this->decomposition_max_level - 1) { 
    // Init the assigned nodes
    if(decomposition_node_patches.empty()){
        std::vector<int> local_permutation;
        cur_node.init_node(-1, -1, decomposition_node_id,
            decomposition_node_parent_id, decomposition_level,
            decomposition_node_patches, local_permutation, -1); //Empty patch
        return;
    } else if(decomposition_node_patches.size() == 1){
        //Compute the permutation for the nodes within a single patch
        auto& patch = patch_nodes[decomposition_node_patches[0]];
        std::vector<int> dofs;
        for(int i = 0; i < patch.nodes.size(); ++i) {
            if(is_separator[patch.nodes[i]]) {
                dofs.push_back(patch.nodes[i]);
            }
        }
        Eigen::SparseMatrix<int> sub_graph = compute_sub_graph(Gp, Gi, G_n, G_nnz, dofs);
        std::vector<int> local_permutation;
        this->local_permute(sub_graph, local_permutation);
        cur_node.init_node(-1, -1, decomposition_node_id,
            decomposition_node_parent_id, decomposition_level,
            dofs, local_permutation, patch.q_id);
        return;
    } else {
        spdlog::error("Decomposition node at last level ({}) has more than one patch ({})", decomposition_level, decomposition_node_patches.size());
        return;
    }
  }

  // Prepare the Wavefront parallelism arrays and data
  int total_number_of_sub_tree_nodes =
      std::pow(2, (decomposition_max_level - decomposition_level) + 1) - 1;
  // Levels are zero based. So for num_levels = 1 we have two levels 0 and 1,
  // and we need an array of size 3 to define wavefront parallelism
  int wavefront_levels = decomposition_max_level - decomposition_level + 1;

  std::vector<int> level_ptr(wavefront_levels + 1);
  std::vector<int> tree_node_ids_set(total_number_of_sub_tree_nodes);
  std::vector<int> tree_node_ids_set_inv(total_number_of_sub_tree_nodes);
  std::vector<int> parent_node_ids_set(total_number_of_sub_tree_nodes);
  // These variables should be computed on the fly
  std::vector<int> offset_set(total_number_of_sub_tree_nodes);

  int size_per_level = 1;
  level_ptr[0] = 0;
  for (int l = 1; l < wavefront_levels + 1; l++) {
    level_ptr[l] = size_per_level + level_ptr[l - 1];
    size_per_level = size_per_level * 2;
  }

  tree_node_ids_set[0] = decomposition_node_id;
  parent_node_ids_set[0] = decomposition_node_parent_id;
  for (int l = 0; l < wavefront_levels - 1; l++) {
    int next_level_idx = level_ptr[l + 1];
    for (int node_ptr = level_ptr[l]; node_ptr < level_ptr[l + 1]; node_ptr++) {
      int node_idx = tree_node_ids_set[node_ptr];
      // Left node
      parent_node_ids_set[next_level_idx] = node_idx;
      tree_node_ids_set[next_level_idx++] = node_idx * 2 + 1;
      // Right node
      parent_node_ids_set[next_level_idx] = node_idx;
      tree_node_ids_set[next_level_idx++] = node_idx * 2 + 2;
    }
  }

  for (int i = 0; i < total_number_of_sub_tree_nodes; i++) {
    tree_node_ids_set_inv[tree_node_ids_set[i]] = i;
  }

  //Think of this as the input of decompose function in recursive manner
  struct DecompositionInfo {
    int decomposition_node_id = -1;
    int decomposition_node_parent_id = -1;
    int decomposition_level = -1;
    bool is_init = false;
    std::vector<int> decomposition_node_patches;
  };

  //Initialize the first call for the decomposition info stack
  std::vector<DecompositionInfo> decomposition_info_stack(total_number_of_sub_tree_nodes);
  decomposition_info_stack[0].decomposition_node_id = decomposition_node_id;
  decomposition_info_stack[0].decomposition_node_parent_id = decomposition_node_parent_id;
  decomposition_info_stack[0].decomposition_level = decomposition_level;
  decomposition_info_stack[0].decomposition_node_patches = decomposition_node_patches;
  decomposition_info_stack[0].is_init = true;

// #pragma omp parallel
  {
    for (int l = 0; l < wavefront_levels; l++) {
// #pragma omp for schedule(dynamic)
      for (int node_ptr = level_ptr[l]; node_ptr < level_ptr[l + 1];
           node_ptr++) {

        //Get the input information for the current node
        assert(decomposition_info_stack[node_ptr].is_init);
         int id = decomposition_info_stack[node_ptr].decomposition_node_id;
         int parent_id = decomposition_info_stack[node_ptr].decomposition_node_parent_id;
         int current_level = decomposition_info_stack[node_ptr].decomposition_level + decomposition_level;
         std::vector<int> &assigned_patches = decomposition_info_stack[node_ptr].decomposition_node_patches;
        auto &cur_decomposition_node = max_match_tree.decomposition_nodes[id];
        //+++++++++++++ If it is a leaf node ++++++++++++++++
        if (current_level == this->decomposition_max_level - 1) { 
            // Init the decomposition node
            if(assigned_patches.empty()){
                std::vector<int> dofs;
                std::vector<int> local_permutation;
                cur_decomposition_node.init_node(-1, -1, id,
                    parent_id, current_level,
                    dofs, local_permutation, -1); //Empty patch
                return;
            } else if(assigned_patches.size() == 1){
                //Compute the permutation for the nodes within a single patch
                auto& patch = patch_nodes[assigned_patches[0]];
                std::vector<int> dofs;
                for(int i = 0; i < patch.nodes.size(); ++i) {
                    if(is_separator[patch.nodes[i]]) {
                        dofs.push_back(patch.nodes[i]);
                    }
                }
                Eigen::SparseMatrix<int> sub_graph = compute_sub_graph(Gp, Gi, G_n, G_nnz, dofs);
                std::vector<int> local_permutation;
                this->local_permute(sub_graph, local_permutation);
                cur_decomposition_node.init_node(-1, -1, id,
                    parent_id, current_level,
                    dofs, local_permutation, patch.q_id);
                return;
            } else {
                spdlog::error("Decomposition node at last level ({}) has more than one patch ({})", decomposition_level, decomposition_node_patches.size());
                return;
            }
        }

        //+++++++++++++ If it is not a leaf node ++++++++++++++++
        //Overall flow:
        //Step 1: Divide the patches into two parts (left and right)
        //Step 2: Compute the sparator between these two parts
        //Step 3: Permute the separator and initialize the decompositon node
        //Step 4: Initialize the input of the left and right childeren for the next wavefront

        //=========== Step 1: Divide the patches into two parts (left and right) ============
        //Compute the quotient graph from the assigned patches
        std::vector<int> quotient_sub_graph_node_weights;
          Eigen::SparseMatrix<int> quotient_sub_graph;
        std::sort(assigned_patches.begin(), assigned_patches.end());//The node_to_partition is mapped this way
        this->compute_sub_graph(Q, Q_node_weights,
            quotient_sub_graph, quotient_sub_graph_node_weights, assigned_patches);
        //Compute the bipartition of the quotient graph
        std::vector<int> node_to_partition;
        this->compute_bipartition(quotient_sub_graph, quotient_sub_graph_node_weights, node_to_partition);
        //Compute the two parts of the quotient graph
        std::vector<int> left_assigned, right_assigned;
        for(int i = 0; i < node_to_partition.size(); i++) {
            int global_patch_id = assigned_patches[i];
            if(node_to_partition[i] == 0) {
                left_assigned.push_back(global_patch_id);
            } else {
                right_assigned.push_back(global_patch_id);
        }

        // Assign nodes
        auto &left_assigned =
            sub_mesh_assigned_nodes[tree_node_ids_set_inv[id * 2 + 1]];
        auto &right_assigned =
            sub_mesh_assigned_nodes[tree_node_ids_set_inv[id * 2 + 2]];
        auto &sep_assigned = current_node.DOFs;
        left_assigned.reserve(left_region.M_n);
        right_assigned.reserve(right_region.M_n);
        sep_assigned.reserve(sep_region.M_n);

        for (int i = 0; i < local_nodes_regions.size(); i++) {
          if (local_nodes_regions[i] == 0) { // Left assigned
            left_assigned.emplace_back(assigned_nodes_par[i]);
          } else if (local_nodes_regions[i] == 1) {
            right_assigned.emplace_back(assigned_nodes_par[i]);
          } else if (local_nodes_regions[i] == 2) {
            sep_assigned.emplace_back(assigned_nodes_par[i]);
          } else {
            std::cerr << "There are more than 3 regions in here" << std::endl;
          }
        }

        // Assign the permuted labels to the sep nodes
        sep_region.perm.resize(sep_region.M_n);
        if (sep_region.M_n != 0) {
          Permute(sep_region.M_n, sep_region.Mp.data(), sep_region.Mi.data(),
                  sep_region.perm.data(), nullptr);
        }

        if (left_region.M_n == 0 && right_region.M_n == 0) {
          current_node.setIdentifier(-1, -1, id, parent_id,
                                     offset_par + left_region.M_n +
                                         right_region.M_n,
                                     current_level, current_node.DOFs.size());
        } else if (left_region.M_n == 0) {
          current_node.setIdentifier(-1, id * 2 + 2, id, parent_id,
                                     offset_par + left_region.M_n +
                                         right_region.M_n,
                                     current_level, current_node.DOFs.size());
        } else if (right_region.M_n == 0) {
          current_node.setIdentifier(id * 2 + 1, -1, id, parent_id,
                                     offset_par + left_region.M_n +
                                         right_region.M_n,
                                     current_level, current_node.DOFs.size());
        } else {
          current_node.setIdentifier(id * 2 + 1, id * 2 + 2, id, parent_id,
                                     offset_par + left_region.M_n +
                                         right_region.M_n,
                                     current_level, current_node.DOFs.size());
        }

        current_node.setInitFlag();
        current_node.setPermutedNewLabel(sep_region.perm);

        // assign the regions to the global node regions
        for (auto &sep_node : sep_assigned) {
          this->DOF_to_HMD_node[sep_node] = id;
        }

        // Left offset
        offset_set[tree_node_ids_set_inv[id * 2 + 1]] = offset_par;
        // Right offset
        offset_set[tree_node_ids_set_inv[id * 2 + 2]] =
            offset_par + left_region.M_n;

        // Clear the vectors values to release memory
        sub_mesh_stack[node_ptr].clear();
        sub_mesh_assigned_nodes[node_ptr].clear();
      }
    }
  };
}


void GPUOrdering_V2::init_patches()
{
    // Create RXMeshStatic from the mesh data (face-vertex connectivity)
    // Use default patch size of 512 (can be adjusted)
    rxmesh::rx_init(0);
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

void GPUOrdering_V2::step1_create_quotient_graph(){
    // Given node to patch, first give each separator node a unique patch ID
    //Step 1: assign patch-id -1 to each boundary vertex
    //Count the number of patch ids
    std::unordered_set<int> unique_ids;
    std::map<int, int> node_per_patch_count;
    for (int i = 0; i < node_to_patch.size(); ++i) {
        unique_ids.insert(node_to_patch[i]);
        node_per_patch_count[node_to_patch[i]]++;
    }
    //Count the offset of each patch to create continuous patch ids
    std::vector<int> patch_offset(unique_ids.size(), 0);
    for(int i = 0; i < unique_ids.size(); ++i) {
        // It is an empty patch and all the patch after it should be offset by 1
        if(node_per_patch_count[i] == 0) {
            for(int j = i; j < unique_ids.size(); ++j) {
                patch_offset[j]++;
            }
        }
    }
    for(int i = 0; i < node_to_patch.size(); ++i) {
        node_to_patch[i] -= patch_offset[node_to_patch[i]];
    }

    //DEBUGING
    int reduced_patches = patch_offset.back();
    spdlog::info("Reduced num of patches: {}", reduced_patches);
    std::unordered_set<int> unique_ids_reduced;
    for(auto& patch: node_to_patch) {
        unique_ids_reduced.insert(patch);
    }
    assert(unique_ids_reduced.size() == reduced_patches);
    
    //Step 2: create quotient graph
    //Step 2.1: rename the vertices of the quotient graph
    int patch_n = unique_ids_reduced.size();//Assuming patches start from 0
    //Step 2.2: create patch nodes
    patch_nodes.resize(patch_n);
    for (int i = 0; i < G_n; ++i) {
        int q_id = node_to_patch[i];
        assert(q_id < patch_nodes.size());
        patch_nodes[q_id].nodes.push_back(i);
        patch_nodes[q_id].q_id = q_id;
    }

    //Step 2.3: compute the edge and node weights
    Q_n = patch_n;
    Q_node_weights.clear();
    Q_node_weights.resize(Q_n, 0);
    //Compute the edge and node weights
    int edge_count = 0;
    //Create triplet for sparse matrix creation
    std::vector<Eigen::Triplet<int>> triplets;
    for (int i = 0; i < G_n; ++i) {
        int node_label = node_to_patch[i];
        Q_node_weights[node_label]++;
        for(int nbr_ptr = Gp[i]; nbr_ptr < Gp[i + 1]; ++nbr_ptr) {
            int nbr_id = Gi[nbr_ptr];
            int nbr_label = node_to_patch[nbr_id];
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

void GPUOrdering_V2::step2_create_hierarchical_partitioning_and_permute(){
    //Step 2.1: create the hierarchical partitioning
    this->decomposition_max_level = std::ceil(std::log2(this->num_patches));
    int total_number_of_decomposition_nodes = std::pow(2, this->decomposition_max_level) - 1;
    max_match_tree.init_max_match_tree(total_number_of_decomposition_nodes);
    is_separator.resize(G_n, false);//The decomposition assign value to is_separator vector
    
}

void GPUOrdering_V2::step3_assemble_permutation(){
}


void GPUOrdering_V2::compute_permutation(std::vector<int>& perm)
{
    if (Gp == nullptr || Gi == nullptr || G_n == 0 || G_nnz == 0) {
        spdlog::error("Graph not set. Please call setGraph() before compute_permutation().");
        return;
    }
    if (fv.size() == 0) {
        spdlog::error("Mesh not set. Please call setMesh() before compute_permutation().");
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

}