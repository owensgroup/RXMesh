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
#include "cuda_error_handler.h"
#include "cuda_profiler_api.h"

#ifdef USE_PROFILE
#include "nvtx_helper.h"
#endif

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
    if (_use_gpu) {
        this->_d_Gp.resize(_G_n + 1);
        thrust::copy(Gp, Gp + _G_n + 1, _d_Gp.begin());
        this->_d_Gi.resize(NNZ);
        thrust::copy(Gi, Gi + NNZ, _d_Gi.begin());
    }

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

void GPUOrdering_V2::local_permute_metis(int G_n, int* Gp, int* Gi,
                                         std::vector<int>& local_permutation)
{
    idx_t N   = G_n;
    idx_t NNZ = Gp[G_n];
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
                 Gp,
                 Gi,
                 NULL,
                 NULL,
                 local_permutation.data(),
                 tmp.data());
}

void GPUOrdering_V2::local_permute_amd(int G_n, int* Gp, int* Gi,
                                       std::vector<int>& local_permutation)
{
    idx_t N   = G_n;
    idx_t NNZ = Gp[G_n];
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
              Gp,
              Gi,
              local_permutation.data(),
              nullptr,
              nullptr);
}

void GPUOrdering_V2::local_permute_unity(int G_n, int* Gp, int* Gi,
                                         std::vector<int>& local_permutation)
{
    local_permutation.resize(G_n);
    for (int i = 0; i < G_n; i++) {
        local_permutation[i] = i;
    }
}

void GPUOrdering_V2::local_permute(int G_n, int* Gp, int* Gi,
                                   std::vector<int>&         local_permutation)
{
    if (this->local_permute_method == "metis") {
        local_permute_metis(G_n, Gp, Gi, local_permutation);
    } else if (this->local_permute_method == "amd") {
        local_permute_amd(G_n, Gp, Gi, local_permutation);
    } else if (this->local_permute_method == "unity") {
        local_permute_unity(G_n, Gp, Gi, local_permutation);
    } else {
        spdlog::error("Invalid local permutation method: {}",
                      this->local_permute_method);
        return;
    }
}

void GPUOrdering_V2::compute_local_quotient_graph(
    int tree_node_idx,///<[in] The index of the current decomposition node
    int& local_Q_n,
    std::vector<int>& local_Qp,
    std::vector<int>& local_Qi,
    std::vector<int>& local_Q_node_weights,///<[out] The node weights of the local quotient graph
    std::vector<int>& q_local_to_global_map///<[in/out] The map from current assigned quotient nodes to tree nodes
){
    local_Qp.clear();
    local_Qi.clear();
    local_Q_node_weights.clear();
    q_local_to_global_map.clear();
    //Find the Qs in this node and create the mapping
    q_local_to_global_map.clear();
    q_local_to_global_map.reserve(this->_num_patches);
    std::vector<int> global_q_to_local_map(this->_num_patches, -1);
    for(int q_node = 0; q_node < this->_quotient_graph._Q_n; q_node++) {
        if(this->_decomposition_tree.q_node_to_tree_node[q_node] == tree_node_idx){
            q_local_to_global_map.push_back(q_node);
            global_q_to_local_map[q_node] = q_local_to_global_map.size() - 1;
        }
    }

    //Create the local Q
    local_Q_n = q_local_to_global_map.size();
    local_Qp.resize(local_Q_n + 1, 0);
    local_Qi.reserve(this->_quotient_graph._Qp[this->_quotient_graph._Q_n]);
    local_Q_node_weights.resize(q_local_to_global_map.size(), 0);
    int cnt = 0;
    for(int local_q = 0; local_q < q_local_to_global_map.size(); local_q++) {
        int global_q = q_local_to_global_map[local_q];
        local_Q_node_weights[local_q] = this->_quotient_graph._Q_node_weights[global_q];
        for(int nbr_ptr = this->_quotient_graph._Qp[global_q]; nbr_ptr < this->_quotient_graph._Qp[global_q + 1]; nbr_ptr++) {
            int nbr_q_global = this->_quotient_graph._Qi[nbr_ptr];
            int nbr_q_local = global_q_to_local_map[nbr_q_global];
            if(nbr_q_local == -1) continue;
            local_Qi.push_back(nbr_q_local);
            cnt++;
        }
        local_Qp[local_q + 1] = cnt;
    }
    assert(local_Qp.back() == local_Qi.size());
}

void GPUOrdering_V2::compute_bipartition(
    int Q_n,
    int* Qp,
    int* Qi,
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

    idx_t   nvtxs  = Q_n;
    idx_t   ncon   = 1;
    idx_t*  vwgt   = NULL;
    idx_t*  vsize  = NULL;
    idx_t   nparts = 2;
    real_t* tpwgts = NULL;
    real_t* ubvec  = NULL;
    idx_t   objval = 0;

    Q_partition_map.resize(Q_n, 0);

    int metis_status = METIS_PartGraphKway(&nvtxs,
                                           &ncon,
                                           Qp,
                                           Qi,
                                           Q_node_weights.data(),
                                           vsize,
                                           nullptr,//Q.valuePtr(),//Edge weights are not used for Q
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
    std::vector<int> two_way_q_partition_map;
    if (tree_node_idx != 0) {
        int local_Q_n;
        std::vector<int> local_Qp;
        std::vector<int> local_Qi;
        std::vector<int> local_Q_node_to_global_Q_node;
        std::vector<int> local_Q_node_weights;
        compute_local_quotient_graph(tree_node_idx,
            local_Q_n,
            local_Qp,
            local_Qi,
            local_Q_node_weights,
            local_Q_node_to_global_Q_node);

        compute_bipartition(local_Q_n, local_Qp.data(), local_Qi.data(), local_Q_node_weights, two_way_q_partition_map);

        //Converting the q map to node map
        for(int i = 0; i < local_Q_node_to_global_Q_node.size(); i++) {
            int q_global_node = local_Q_node_to_global_Q_node[i];
            //Assign the q to left and right nodes
            this->_decomposition_tree.q_node_to_tree_node[q_global_node] = tree_node_idx * 2 + two_way_q_partition_map[i] + 1;
        }
    } else {
        compute_bipartition(_quotient_graph._Q_n, _quotient_graph._Qp.data(), _quotient_graph._Qi.data(), _quotient_graph._Q_node_weights, two_way_q_partition_map);
        //Converting the q map to node map
        for(int i = 0; i < _quotient_graph._Q_n; i++) {
            //Assign the q to left and right nodes
            this->_decomposition_tree.q_node_to_tree_node[i] = tree_node_idx * 2 + two_way_q_partition_map[i] + 1;
        }
    }

}

// Functor must be defined outside the function for CUDA compatibility
struct is_node_separator_functor {
    int _n;
    int* _p;
    int* _i;
    char *_is_sep;
    int* _g_node_to_patch;
    int* _q_node_to_tree_node;

    __host__ __device__
    is_node_separator_functor(int* p, int* i, int n, char* is_sep, int* g_node_to_patch, int* q_node_to_tree_node) {
        this->_p = p;
        this->_i = i;
        this->_n = n;
        this->_is_sep = is_sep;
        this->_g_node_to_patch = g_node_to_patch;
        this->_q_node_to_tree_node = q_node_to_tree_node;
    }

    __host__ __device__
    int get_tree_node_id(int node) {
        int q_node_id = this->_g_node_to_patch[node];
        return this->_q_node_to_tree_node[q_node_id];
    }

    __host__ __device__
    int operator()(int node) {
        int tree_node_id = get_tree_node_id(node);
        for (int nbr_ptr = this->_p[node]; nbr_ptr < this->_p[node + 1]; ++nbr_ptr) {
            const int nbr_id = this->_i[nbr_ptr];
            if(this->_is_sep[nbr_id] == 1) continue;
            int nbr_partition_id = get_tree_node_id(nbr_id);
            if(tree_node_id == nbr_partition_id) continue;
            return node;
        }
        return -1;
    }
};

void GPUOrdering_V2::find_separator_superset_gpu(
    std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
    std::vector<int>& separator_superset///<[out] The superset of separator nodes
)
{
    thrust::device_vector<int> d_assigned_g_nodes(assigned_g_nodes.size());
    thrust::copy(assigned_g_nodes.begin(), assigned_g_nodes.end(), d_assigned_g_nodes.begin());

    auto sep_operator_obj = is_node_separator_functor(
        thrust::raw_pointer_cast(this->_d_Gp.data()),
        thrust::raw_pointer_cast(this->_d_Gi.data()),
        this->_G_n,
        thrust::raw_pointer_cast(this->_decomposition_tree.d_is_sep.data()),
        thrust::raw_pointer_cast(this->_d_g_node_to_patch.data()),
        thrust::raw_pointer_cast(this->_decomposition_tree.d_q_node_to_tree_node.data()));

    auto start = thrust::make_transform_iterator(d_assigned_g_nodes.begin(), sep_operator_obj);
    auto end = thrust::make_transform_iterator(d_assigned_g_nodes.end(), sep_operator_obj);
    thrust::device_vector<int> d_separator_superset(assigned_g_nodes.size(), -1);
    auto end_it = thrust::copy_if(
        start,
        end,
        d_separator_superset.begin(),
        [] __device__ (int flag) { return flag != -1; });

    separator_superset.resize(end_it - d_separator_superset.begin());
    thrust::copy(d_separator_superset.begin(), end_it, separator_superset.begin());
}

void GPUOrdering_V2::find_separator_superset_cpu(
    std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
    std::vector<int>& separator_superset///<[out] The superset of separator nodes
)
{

    auto get_partition_id = [&](int g_node_id) -> int {
        int q_node_id = this->_g_node_to_patch[g_node_id];
        return this->_decomposition_tree.q_node_to_tree_node[q_node_id];
    };

    separator_superset.clear();
    for(int g_node : assigned_g_nodes) {
        assert(g_node < this->_G_n);
        int partition_id = get_partition_id(g_node);
        for (int nbr_ptr = this->_Gp[g_node]; nbr_ptr < this->_Gp[g_node + 1]; ++nbr_ptr) {
            int nbr_id = this->_Gi[nbr_ptr];
            assert(nbr_id < _decomposition_tree.is_sep.size());
            if(this->_decomposition_tree.is_sep[nbr_id] == 1) continue;
            int nbr_partition_id = get_partition_id(nbr_id);
            if(partition_id == nbr_partition_id) continue;
            separator_superset.push_back(g_node);
            break;
        }
    }
    //Erase repetitive nodes
    // std::sort(separator_superset.begin(), separator_superset.end());
    #ifndef NDEBUG
    int prev_size = separator_superset.size();
    separator_superset.erase(std::unique(separator_superset.begin(), separator_superset.end()), separator_superset.end());
    int after_size = separator_superset.size();
    assert(prev_size == after_size);
    #endif
}

void GPUOrdering_V2::refine_bipartate_separator(
    int parent_node_id,
    std::vector<int>& separator_superset)
{
    // The boundary nodes to be used for extracting bipartite graph
    if (separator_superset.empty()) {
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
    std::vector<int> bipart_p(separator_superset.size() + 1, 0);
    std::vector<int> bipart_i;
    int bipartition_nnz = 0;
    for (auto& g_node : separator_superset) {
        assert(!this->_decomposition_tree.is_separator(g_node));
        int g_node_tree_id = get_tree_id(g_node);
        int bipart_node_id = separator_node_to_local_node[g_node];
        for (int nbr_ptr = this->_Gp[g_node]; nbr_ptr < this->_Gp[g_node + 1]; nbr_ptr++) {
            int nbr_id = this->_Gi[nbr_ptr];
            //Not in the separator set
            if(separator_node_to_local_node.find(nbr_id) == separator_node_to_local_node.end()) continue;
            //They should not be in the same partition
            int nbr_node_tree_id = get_tree_id(nbr_id);
            if (g_node_tree_id == nbr_node_tree_id) continue;
            int bipart_nbr_id = separator_node_to_local_node[nbr_id];
            // triplets.emplace_back(separator_node_to_local_node[g_node], separator_node_to_local_node[nbr_id], 1);
            // triplets.emplace_back(separator_node_to_local_node[nbr_id], separator_node_to_local_node[g_node], 1);
            bipart_i.push_back(bipart_nbr_id);
            bipartition_nnz++;
        }
        bipart_p[bipart_node_id + 1] = bipartition_nnz;
    }


    //=========== Step : assign bipartition for max match
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
    //Check for symmetric
    for (int i = 0; i < separator_superset.size(); i++) {
        for (int j_ptr = bipart_p[i]; j_ptr < bipart_p[i + 1]; ++j_ptr) {
            int j = bipart_i[j_ptr];
            // Check if edge (j, i) exists
            bool found_reverse_edge = false;
            for (int i_ptr = bipart_p[j]; i_ptr < bipart_p[j + 1]; ++i_ptr) {
                if (bipart_i[i_ptr] == i) {
                    found_reverse_edge = true;
                    break;
                }
            }
            if (!found_reverse_edge) {
                spdlog::error("The matrix is not symmetric: edge ({}, {}) exists but ({}, {}) does not", i, j, j, i);
                assert(false);
            }
        }
    }


    //Check for bipartite
    for (int node = 0; node < separator_superset.size(); node++) {
        for (int nbr_ptr = bipart_p[node];
             nbr_ptr < bipart_p[node + 1];
             ++nbr_ptr) {
            int nbr_id = bipart_i[nbr_ptr];
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
        separator_superset.size(),
        bipart_p.data(),
        bipart_i.data(),
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
    if(_use_gpu) {
        find_separator_superset_gpu(assigned_g_nodes, separator_g_nodes);
    } else {
        find_separator_superset_cpu(assigned_g_nodes, separator_g_nodes);
    }
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
    #ifdef USE_PROFILE
    NVTX_RANGE_COLOR("decompose", 0xFF00FF00);  // Green
    #endif
    // Initialize the first call for the decomposition info stack
    std::vector<DecompositionInfo> decomposition_info_stack(
        this->_decomposition_tree.get_number_of_decomposition_nodes());
    decomposition_info_stack[0].decomposition_node_parent_id = -1;
    decomposition_info_stack[0].assigned_g_nodes.resize(_G_n);
    for(int i = 0; i < _G_n; i++) {
        decomposition_info_stack[0].assigned_g_nodes[i] = i;
    }

    // auto omp_parallel_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        for (int l = 0; l < this->_decomposition_tree.decomposition_level; l++) {
            int start_level_idx = (1 << l) - 1;
            int end_level_idx   = (1 << (l + 1)) - 1;
            assert(end_level_idx < this->_decomposition_tree.get_number_of_decomposition_nodes());
            #pragma omp for schedule(dynamic)
            for (int tree_node_id = start_level_idx; tree_node_id < end_level_idx;
                 ++tree_node_id) {

                // Get the input information for the current node
#ifndef NDEBUG
                spdlog::info("*********** Working on node {} ***********", tree_node_id);
#endif
                // auto node_start = std::chrono::high_resolution_clock::now();
                int tree_node_parent_id = decomposition_info_stack[tree_node_id].decomposition_node_parent_id;
                std::vector<int>&

                    assigned_g_nodes = decomposition_info_stack[tree_node_id].assigned_g_nodes;
                auto& cur_decomposition_node =this->_decomposition_tree.decomposition_nodes[tree_node_id];

                //++++++ if it is not decomposable, skip it ******
                if(assigned_g_nodes.empty()) continue;


                if (tree_node_id == -1 ||
                    tree_node_parent_id == -1) {
                    if (tree_node_id != 0) {
                        spdlog::error("The info is not initialized correctly.");
                        spdlog::error("The level is {}", l);
                        spdlog::error("The node_id {}, and parent_id {}",
                                      tree_node_id,
                                      tree_node_parent_id);
                    }
                }


                //+++++++++++++ If it is a leaf node ++++++++++++++++
                if (l == this->_decomposition_tree.decomposition_level - 1) {
                    // Add all the nodes in the patches to the dofs
                    cur_decomposition_node.init_node(
                        -1,
                        -1,
                        tree_node_id,
                        tree_node_parent_id,
                        l,
                        assigned_g_nodes);
                    //Assign nodes to tree array
                    for (const auto& node: assigned_g_nodes) {
                        this->_decomposition_tree.g_node_to_tree_node[node] = tree_node_id;
                    }
                    continue;
                }

                //+++++++++++++ If it is not a leaf node ++++++++++++++++
                // Overall flow:
                // Step 1: Compute two equal size partitions from the assigned dofs
                // Step 2: Find the separator nodes of the two partitions
                // Step 3: Initialize the input of the left and right

                // Step 1: Compute two equal size partitions from the assigned dofs
                // auto two_way_start = std::chrono::high_resolution_clock::now();
                #ifdef USE_PROFILE
                {
                    NVTX_RANGE_COLOR("two_way_Q_partition", 0xFFFF0000);  // Red
                #endif
                    this->two_way_Q_partition(tree_node_id,
                        assigned_g_nodes);
                #ifdef USE_PROFILE
                }
                #endif

                //Copy the q to tree_node
                if(_use_gpu) {
                    #ifdef USE_PROFILE
                    {
                            NVTX_RANGE_COLOR("gpu_copy_q", 0xFF0000FF);  // Blue
                    #endif
                        THRUST_CALL(thrust::copy(this->_decomposition_tree.q_node_to_tree_node.begin(),
                            this->_decomposition_tree.q_node_to_tree_node.end(),
                            this->_decomposition_tree.d_q_node_to_tree_node.begin()));
                    #ifdef USE_PROFILE
                    }
                    #endif
                }

                // Step 2: Find the separator nodes of the two partitions
                std::vector<int> separator_g_nodes;
                auto three_way_start = std::chrono::high_resolution_clock::now();
                #ifdef USE_PROFILE
                {
                    NVTX_RANGE_COLOR("three_way_G_partition", 0xFFFFFF00);  // Yellow
                #endif
                    this->three_way_G_partition(tree_node_id, assigned_g_nodes, separator_g_nodes);
                #ifdef USE_PROFILE
                }
                #endif
                auto three_way_end = std::chrono::high_resolution_clock::now();
                auto three_way_duration = std::chrono::duration_cast<std::chrono::milliseconds>(three_way_end - three_way_start).count();
                #ifdef USE_PROFILE
                {
                    NVTX_RANGE_COLOR("assign_separator", 0xFFFFFF00);
                #endif
                    this->_decomposition_tree.assign_nodes_to_tree(separator_g_nodes, tree_node_id);
                #ifdef USE_PROFILE
                }
                #endif
                
                // spdlog::info("two_way_Q_partition time: {} ms", two_way_duration);
                // spdlog::info("three_way_G_partition time: {} ms", three_way_duration);
                #ifdef USE_PROFILE
                {
                    NVTX_RANGE_COLOR("init_next_level", 0xFFFFFF00);
                #endif
                std::vector<int> left_assigned_g_nodes, right_assigned_g_nodes;

                //Assign the nodes to the left and right assigned dofs
                left_assigned_g_nodes.reserve(assigned_g_nodes.size());
                right_assigned_g_nodes.reserve(assigned_g_nodes.size());
                for(int i = 0; i < assigned_g_nodes.size(); i++) {
                    int g_node = assigned_g_nodes[i];
                    int q_node = this->_g_node_to_patch[g_node];
                    if(this->_decomposition_tree.is_separator(g_node)) continue;
                    int q_partition = this->_decomposition_tree.q_node_to_tree_node[q_node];
                    if(q_partition == tree_node_id * 2 + 1) {
                        left_assigned_g_nodes.push_back(g_node);
                    } else {
                        right_assigned_g_nodes.push_back(g_node);
                        assert(q_partition == tree_node_id * 2 + 2);
                    }
                }
                //Compress
                left_assigned_g_nodes.shrink_to_fit();
                right_assigned_g_nodes.shrink_to_fit();

#ifndef NDEBUG
                spdlog::info("The left size is: {}, the right size is: {} and the separator size is {}: ",
                    left_assigned_g_nodes.size(),
                    right_assigned_g_nodes.size(),
                    separator_g_nodes.size());
#endif

                // Initialize the input of the left and right children for the
                // next wavefront
                int left_node_idx  = tree_node_id * 2 + 1;
                int right_node_idx = tree_node_id * 2 + 2;
                if (left_assigned_g_nodes.empty()) {
                    left_node_idx = -1;
                }
                if (right_assigned_g_nodes.empty()) {
                    right_node_idx = -1;
                }


                //Update the quotient graph by removing the effect of separator nodes
                for(int i = 0; i < separator_g_nodes.size(); i++) {
                    int g_node = separator_g_nodes[i];
                    int q_node = this->_g_node_to_patch[g_node];
                    this->_quotient_graph._Q_node_weights[q_node]--;
                    //TODO: Remove the edge weights next if the performance degrades
                }

                cur_decomposition_node.init_node(left_node_idx,
                                                 right_node_idx,
                                                 tree_node_id,
                                                 tree_node_parent_id,
                                                 l,
                                                 separator_g_nodes);

                if (left_node_idx != -1) {
                    assert(left_node_idx >= 0 &&
                           left_node_idx < decomposition_info_stack.size());
                    assert(!left_assigned_g_nodes.empty() &&
                           "Left child should have assigned patches");
                    decomposition_info_stack[left_node_idx]
                        .decomposition_node_parent_id = tree_node_id;
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
                        .decomposition_node_parent_id = tree_node_id;
                    decomposition_info_stack[right_node_idx].assigned_g_nodes =
                        right_assigned_g_nodes;
                }

                #ifdef USE_PROFILE
                }
                #endif
                
                // auto node_end = std::chrono::high_resolution_clock::now();
                // auto node_duration = std::chrono::duration_cast<std::chrono::milliseconds>(node_end - node_start).count();
                // spdlog::info("Node {} total time: {} ms", node_idx, node_duration);
            }
        }
    }
    // auto omp_parallel_end = std::chrono::high_resolution_clock::now();
    // auto omp_parallel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(omp_parallel_end - omp_parallel_start).count();
    // spdlog::info("OpenMP parallel region wall clock time: {} ms", omp_parallel_duration);
}


void GPUOrdering_V2::init_patches()
{
    // Create RXMeshStatic from the mesh data (face-vertex connectivity)
    // Use default patch size of 512 (can be adjusted)
    rxmesh::rx_init(0);
    _rxmesh = std::make_unique<rxmesh::RXMeshStatic>(this->fv, "", this->_patch_size);

    spdlog::info(
        "RXMesh initialized with {} vertices, {} edges, {} faces, {} patches",
        _rxmesh->get_num_vertices(),
        _rxmesh->get_num_edges(),
        _rxmesh->get_num_faces(),
        _rxmesh->get_num_patches());

    this->_g_node_to_patch.resize(_rxmesh->get_num_vertices());
    this->_num_patches = _rxmesh->get_num_patches();
    _rxmesh->for_each_vertex(
        rxmesh::HOST,
        [&](const rxmesh::VertexHandle vh) {
            uint32_t node_id       = _rxmesh->map_to_global(vh);
            this->_g_node_to_patch[node_id] = static_cast<int>(vh.patch_id());
        },
        NULL,
        false);
    // Init the hirerchical tree memory
    int num_levels = std::ceil(std::log2(this->_num_patches));
    num_levels--;//Based on my experience, normally, the last level is empty
    int total_number_of_decomposition_nodes = (1 << (num_levels + 1)) - 1;
    this->_decomposition_tree.init_decomposition_tree(
        total_number_of_decomposition_nodes,
        num_levels, _rxmesh->get_num_vertices(), this->_num_patches, _use_gpu);
    if(_use_gpu) {
        _d_g_node_to_patch = _g_node_to_patch;
    }
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


void GPUOrdering_V2::step1_compute_quotient_graph()
{
    // Given node to patch, first give each separator node a unique patch ID
    // Step 1: assign patch-id -1 to each boundary vertex
    // Count the number of patch ids

    // auto patch_fix_start = std::chrono::high_resolution_clock::now();
    std::vector<int> patch_offset(_num_patches, 1);
    // auto patch_fix_end = std::chrono::high_resolution_clock::now();
    // auto patch_time = std::chrono::duration_cast<std::chrono::milliseconds>(patch_fix_end - patch_fix_start).count();
    // spdlog::info("Patch fix time: {} ms for vector of size {}", patch_time, _num_patches);
    for (int i = 0; i < _g_node_to_patch.size(); i++) {
        int patch_id = _g_node_to_patch[i];
        patch_offset[patch_id] = 0;
    }


    //prefix scan
    for (int i = 1; i < _num_patches; ++i) {
        patch_offset[i] += patch_offset[i - 1];
    }

    for (int i = 0; i < _g_node_to_patch.size(); i++) {
        int patch_id = _g_node_to_patch[i];
        _g_node_to_patch[i] -= patch_offset[patch_id];
    }
    this->_num_patches -= patch_offset.back();

    assert(this->_g_node_to_patch.size() == this->_G_n);

    
    //Create the local quotient graph
    // auto quotient_start = std::chrono::high_resolution_clock::now();
    
    // 1. Node weights (patch sizes) and mapping nodes -> patches
    _quotient_graph._Q_node_weights.assign(_num_patches, 0);

    // Optional but very useful: nodes of each patch
    std::vector<std::vector<int>> patch_nodes(_num_patches);
    patch_nodes.reserve(_num_patches);

    for (int g = 0; g < _G_n; ++g) {
        int p = this->_g_node_to_patch[g];
        assert(p >= 0 && p < _num_patches);
        _quotient_graph._Q_node_weights[p]++;
        patch_nodes[p].push_back(g);
    }

    // 2. Build adjacency lists of quotient graph (per patch)
    std::vector<std::vector<int>> Q_adj(_num_patches);
    std::vector<int> mark(_num_patches, -1);  // mark[q] == p means q already added as neighbor of p

    for (int p = 0; p < _num_patches; ++p) {
        auto &nodes = patch_nodes[p];
        if (nodes.empty()) continue;

        for (int g : nodes) {
            for (int nbr_ptr = this->_Gp[g]; nbr_ptr < this->_Gp[g + 1]; ++nbr_ptr) {
                int nbr_id = this->_Gi[nbr_ptr];
                int q = this->_g_node_to_patch[nbr_id];
                if (q == p) continue;  // stay inside patch -> no quotient edge

                // Deduplicate neighbors for this p using "mark"
                if (mark[q] == p) continue;
                mark[q] = p;
                Q_adj[p].push_back(q);
            }
        }
    }

    int q_nnz = 0;
    for (auto& nbrs: Q_adj) {
        std::sort(nbrs.begin(), nbrs.end());
        q_nnz += nbrs.size();
    }
    _quotient_graph._Q_n = this->_num_patches;
    _quotient_graph._Qp.resize(this->_num_patches + 1, 0);
    _quotient_graph._Qi.resize(q_nnz, 0);
    int cnt = 0;
    for(int q_id = 0; q_id < this->_num_patches; q_id++) {
        auto& nbrs = Q_adj[q_id];
        for (const auto&  nbr: nbrs) {
            _quotient_graph._Qi[cnt] = nbr;
            cnt++;
        }
        _quotient_graph._Qp[q_id + 1] = cnt;
    }
    assert(cnt == q_nnz);
    #ifdef DEBUG
    // Check if the quotient graph CSR matrix is symmetric
    bool is_symmetric = true;
    for (int i = 0; i < _quotient_graph._Q_n; ++i) {
        for (int idx = _quotient_graph._Qp[i]; idx < _quotient_graph._Qp[i + 1]; ++idx) {
            int j = _quotient_graph._Qi[idx];
            
            // Check if i exists in row j's neighbors
            bool found = false;
            for (int jdx = _quotient_graph._Qp[j]; jdx < _quotient_graph._Qp[j + 1]; ++jdx) {
                if (_quotient_graph._Qi[jdx] == i) {
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                is_symmetric = false;
                spdlog::warn("CSR matrix is NOT symmetric: edge ({}, {}) exists but ({}, {}) does not", i, j, j, i);
                break;
            }
        }
        if (!is_symmetric) break;
    }
    assert(is_symmetric);
    #endif
}

void GPUOrdering_V2::step2_create_decomposition_tree()
{
    #ifdef USE_PROFILE
    cudaProfilerStart();
    #endif
    this->decompose();
    #ifdef USE_PROFILE
    cudaProfilerStop();
    #endif
#ifndef NDEBUG
    spdlog::info("Decomposition tree is created.");
#endif

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

void GPUOrdering_V2::compute_sub_graphs(std::vector<SubGraph>& sub_graphs){
    // This loop creates the global-to-local mapping as well as number of nodes per group
    std::vector<int> global_to_local(this->_G_n, -1);
    sub_graphs.clear();
    sub_graphs.resize(this->_decomposition_tree.decomposition_nodes.size());

    for (auto& node : this->_decomposition_tree.decomposition_nodes) {
        if (node.assigned_g_nodes.empty()) continue;

        // assign local ids 0..k-1 within this subgraph
        for (int i = 0; i < (int)node.assigned_g_nodes.size(); i++) {
            int g_node = node.assigned_g_nodes[i];
            assert(global_to_local[g_node] == -1);
            global_to_local[g_node] = i;
        }

        sub_graphs[node.node_id]._num_nodes = (int)node.assigned_g_nodes.size();
        sub_graphs[node.node_id]._Gp.assign(node.assigned_g_nodes.size() + 1, 0);
    }

    // This loop finds the number of edges per group (total, for allocation sanity)
    std::vector<int> edge_per_group(sub_graphs.size(), 0);
    for (int i = 0; i < this->_G_n; i++) {
        int group_id = this->_decomposition_tree.g_node_to_tree_node[i];
        if (group_id < 0) continue; // in case some nodes are unassigned

        for (int nbr_ptr = _Gp[i]; nbr_ptr < _Gp[i + 1]; nbr_ptr++) {
            int nbr_id       = _Gi[nbr_ptr];
            int nbr_group_id = this->_decomposition_tree.g_node_to_tree_node[nbr_id];
            if (nbr_group_id == group_id) {
                edge_per_group[group_id]++;
            }
        }
    }
    for (int i = 0; i < (int)sub_graphs.size(); i++) {
        if (sub_graphs[i]._num_nodes == 0) continue;
        sub_graphs[i]._Gi.resize(edge_per_group[i], 0);
    }

    // Now build the sub graphs
    // Pass 1: count degree per local node within its group to build _Gp (row pointers)
    for (int g_node = 0; g_node < this->_G_n; g_node++) {
        int group_id = this->_decomposition_tree.g_node_to_tree_node[g_node];
        assert(group_id >= 0 && group_id < sub_graphs.size());
        int local_i = global_to_local[g_node];
        assert(local_i >= 0 && local_i < sub_graphs[group_id]._num_nodes);

        SubGraph& sg = sub_graphs[group_id];
        if (sg._num_nodes == 0) continue;

        for (int nbr_ptr = _Gp[g_node]; nbr_ptr < _Gp[g_node + 1]; nbr_ptr++) {
            int nbr_id       = _Gi[nbr_ptr];
            int nbr_group_id = this->_decomposition_tree.g_node_to_tree_node[nbr_id];
            if (nbr_group_id != group_id) continue;

            int local_j = global_to_local[nbr_id];
            assert(local_j >= 0);
            // Count one edge from local_i
            sg._Gp[local_i + 1]++;
        }
    }

    // Prefix-sum to convert degrees into CSR row pointers
    for (int gid = 0; gid < (int)sub_graphs.size(); gid++) {
        SubGraph& sg = sub_graphs[gid];
        if (sg._num_nodes == 0) continue;

        for (int r = 0; r < sg._num_nodes; r++) {
            sg._Gp[r + 1] += sg._Gp[r];
        }

        // Optional sanity check: total edges match allocation
        assert((int)sg._Gi.size() == sg._Gp[sg._num_nodes]);
    }

    // Pass 2: fill _Gi using a set of write cursors, one per group
    std::vector<std::vector<int>> write_pos(sub_graphs.size());
    for (int gid = 0; gid < sub_graphs.size(); gid++) {
        if (sub_graphs[gid]._num_nodes == 0) continue;
        write_pos[gid] = sub_graphs[gid]._Gp; // start at row starts
    }

    for (int g_node = 0; g_node < this->_G_n; g_node++) {
        int group_id = this->_decomposition_tree.g_node_to_tree_node[g_node];
        assert(group_id >= 0 && group_id < sub_graphs.size());
        int local_i = global_to_local[g_node];

        SubGraph& sg = sub_graphs[group_id];
        if (sg._num_nodes == 0) continue;

        for (int nbr_ptr = _Gp[g_node]; nbr_ptr < _Gp[g_node + 1]; nbr_ptr++) {
            int nbr_id       = _Gi[nbr_ptr];
            int nbr_group_id = this->_decomposition_tree.g_node_to_tree_node[nbr_id];
            if (nbr_group_id != group_id) continue;

            int local_j = global_to_local[nbr_id];
            assert(local_j >= 0);

            int& pos = write_pos[group_id][local_i];
            sg._Gi[pos] = local_j;
            pos++;
        }
    }
}

void GPUOrdering_V2::step3_CPU_compute_local_permutations()
{
    spdlog::info("Computing local subgraphs .. ");
    std::vector<SubGraph> sub_graphs;

    auto sub_graph_start = std::chrono::high_resolution_clock::now();
    this->compute_sub_graphs(sub_graphs);
    auto sub_graph_end   = std::chrono::high_resolution_clock::now();
    double sub_graph_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            sub_graph_end - sub_graph_start)
            .count();
    spdlog::info("Sub-graph construction completed in {:.6f} seconds",
                 sub_graph_time);

    //Compute the local permutations
    auto local_perm_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for 
    for (int i = 0; i < this->_decomposition_tree.decomposition_nodes.size(); i++) {
        std::vector<int> local_permutation;
        if (sub_graphs[i]._num_nodes == 0) continue;
        local_permute(sub_graphs[i]._num_nodes, sub_graphs[i]._Gp.data(), sub_graphs[i]._Gi.data(), local_permutation);
        this->_decomposition_tree.decomposition_nodes[i].set_local_permutation(local_permutation);
    }
    auto local_perm_end = std::chrono::high_resolution_clock::now();
    double local_perm_openmp_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            local_perm_end - local_perm_start)
            .count();
    spdlog::info("Local permutation (OpenMP) block completed in {:.6f} seconds",
                 local_perm_openmp_time);
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

    spdlog::info("Using GPU: {}", _use_gpu);
    // Step 1: Compute node to patch map
    auto start_time = std::chrono::high_resolution_clock::now();
    step1_compute_quotient_graph();
    auto end_time = std::chrono::high_resolution_clock::now();
    node_to_patch_time = std::chrono::duration<double>(end_time - start_time).count();
    spdlog::info("Step 1 (compute quotient graph) completed in {:.6f} seconds", node_to_patch_time);

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