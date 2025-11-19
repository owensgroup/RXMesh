//
// Created by behrooz on 2025-10-07.
//
#pragma once
#include "Eigen/Core"
#include "Eigen/Sparse"
#include "rxmesh/rxmesh_static.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include "cuda_error_handler.h"

namespace RXMESH_SOLVER {

    
    struct assign_gpu_sep{
        int* _is_sep;
        int* assigned_nodes;
        int tree_node_id;
        assign_gpu_sep(int* is_sep, int* assigned_nodes, int tree_node_id) {
            this->_is_sep = is_sep;
            this->assigned_nodes = assigned_nodes;
            this->tree_node_id = tree_node_id;
        }
        __host__ __device__
        void operator()(int node){
            _is_sep[node] = tree_node_id;
        }
    };

class GPUOrdering_V2
{
public:
    GPUOrdering_V2();
    ~GPUOrdering_V2();
    // Think of this as the input of decompose function in recursive manner
    struct DecompositionInfo
    {
        int              decomposition_node_parent_id = -1;
        std::vector<int> assigned_g_nodes;
    };

    struct DecompositionNode
    {
        int left_node_idx = -1;
        int right_node_idx = -1;
        int node_id = -1;
        int parent_idx = -1;
        int level = -1;
        int offset = -1;
        std::vector<int> assigned_g_nodes;
        std::vector<int> local_new_labels;
        bool is_initialized = false;

        bool isLeaf() const
        {
            if (left_node_idx == -1 && right_node_idx == -1)
                return true;
            return false;
        }

        
        void init_node(int left_node_idx, int right_node_idx,
            int node_id, int parent_idx, int level, std::vector<int> & assigned_g_nodes)
        {
            this->left_node_idx = left_node_idx;
            this->right_node_idx = right_node_idx;
            this->parent_idx = parent_idx;
            this->node_id = node_id;
            this->level = level;
            this->assigned_g_nodes = assigned_g_nodes;
            this->is_initialized = true;
        }

        void set_local_permutation(std::vector<int> & local_permutation) {
            this->local_new_labels.resize(assigned_g_nodes.size());
            for(int i = 0; i < local_permutation.size(); ++i) {
                assert(local_permutation[i] < local_permutation.size());
                this->local_new_labels[local_permutation[i]] = i;
            }
        }

        void set_offset(int offset) {
            this->offset = offset;
        }
    };

    struct DecompositionTree {
        int decomposition_level = -1;
        std::vector<DecompositionNode> decomposition_nodes; // All the nodes of the tree
        std::vector<int> decomposition_node_offset; // The offset of the nodes in the tree for permutation
        std::vector<char> is_sep;
        std::vector<int> g_node_to_tree_node;
        std::vector<int> q_node_to_tree_node;
        thrust::device_vector<int> d_q_node_to_tree_node;
        thrust::device_vector<char> d_is_sep;
        bool _use_gpu = false;


        void init_decomposition_tree(int num_decomposition_nodes,
            int decomposition_level, int total_nodes, int num_patches, bool use_gpu) {
#ifndef NDEBUG
            spdlog::info("Decomposition tree creation .. ");
            spdlog::info("Number of decomposition levels: {}", decomposition_level);
#endif
            this->decomposition_nodes.resize(num_decomposition_nodes);
            this->decomposition_level = decomposition_level;
            this->q_node_to_tree_node.resize(num_patches, 0);
            this->g_node_to_tree_node.resize(total_nodes, 0);
            this->is_sep.clear();
            this->is_sep.resize(total_nodes, 0);

            this->_use_gpu = use_gpu;
            if(this->_use_gpu) {
                this->d_q_node_to_tree_node.resize(num_patches);
                this->d_is_sep.resize(total_nodes, 0);
            }
        }
        inline int get_number_of_decomposition_nodes() const {
            return decomposition_nodes.size();
        }

        inline bool is_separator(const int node_id) const
        {
            return this->is_sep[node_id];
        }

        inline void assign_nodes_to_tree(
            const std::vector<int>& separator_g_nodes, int tree_node_id) {
            //flag the separator nodes in the decomposition tree
            for(int separator_g_node : separator_g_nodes) {
                assert(separator_g_node < g_node_to_tree_node.size());
                assert(is_sep.size() == g_node_to_tree_node.size());
                assert(!this->is_separator(separator_g_node));
                this->is_sep[separator_g_node] = 1;
                this->g_node_to_tree_node[separator_g_node] = tree_node_id;
            }
            if(this->_use_gpu) {
                THRUST_CALL(thrust::copy(this->is_sep.begin(), this->is_sep.end(), this->d_is_sep.begin()));
            }
        }
    };

    struct QuotientGraph {
        int _Q_n;
        std::vector<int> _Q_edge_weights;
        std::vector<int> _Q_node_weights;
        std::vector<int> _Qp;
        std::vector<int> _Qi;
    };    

    struct SubGraph {
        int _num_nodes;
        std::vector<int> _Gp;
        std::vector<int> _Gi;
    };




    QuotientGraph _quotient_graph;

    std::string local_permute_method = "amd";
    // std::string separator_refinement_method = "nothing";
    DecompositionTree _decomposition_tree;
    int _decomposition_max_level;
    int _patch_size = 512;
    int _num_patches = -1;
    std::vector<int> _g_node_to_patch;
    std::vector<int> _g_node_to_tree_node;
    thrust::device_vector<int> _d_g_node_to_patch;
    int _G_n, _G_nnz;
    int* _Gp, *_Gi;
    thrust::device_vector<int> _d_Gp;
    thrust::device_vector<int> _d_Gi;

    double node_to_patch_time = 0;
    double decompose_time = 0;
    double local_permute_time = 0;
    double assemble_time = 0;
    double quotient_graph_creation_time = 0;

    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>> vertices;

    bool _use_gpu = true;
    std::unique_ptr<rxmesh::RXMeshStatic> _rxmesh;
    double _separator_ratio = 0.0;

    void setGraph(int* Gp, int* Gi, int G_N, int NNZ);
    void setMesh(const double* V_data, int V_rows, int V_cols,
                 const int* F_data, int F_rows, int F_cols);

    void init_patches();

    //This function updates the q_node_to_tree_node map for the current decomposition node
    //The left and right q nodes are mark as tree_node_idx * 2 + 1 and tree_node_idx * 2 + 2 respectively
    void two_way_Q_partition(int tree_node_idx,///<[in] The index of the current decomposition node
        std::vector<int>& assigned_g_nodes///<[in] Assigned G nodes for current decomposition
    );
  
    void compute_bipartition(
        int Q_n,
        int* Qp,
        int* Qi,
        std::vector<int>&         Q_node_weights,///<[in] The node weights of the graph
        std::vector<int>&         Q_partition_map///<[out] The partition map of the graph
    );


    void find_separator_superset_cpu(
        std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
        std::vector<int>& separator_superset///<[out] The superset of separator nodes
    );

    void find_separator_superset_gpu(
        std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
        std::vector<int>& separator_superset///<[out] The superset of separator nodes
    );
    //Given a set of separator superset, from G, extracts the bipartite graph and refines the separator nodes
    //Based on the Hopcroft-Karp algorithm
    void refine_bipartate_separator(
        int tree_node_idx,///<[in] The tree decomposition node
        std::vector<int>& separator_superset///<[in] The separator superset
        );

    //Given a subset of nodes (filtered nodes are marked with -1), partition the graph into three parts: left, right, and separator
    void three_way_G_partition(int tree_node_idx,///<[in] The index of the current decomposition node
        std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
        std::vector<int>& separator_g_nodes///<[out] The separator nodes
    );

    void decompose();

    void compute_local_quotient_graph(
        int tree_node_idx,///<[in] The index of the current decomposition node
        int& local_Q_n,
        std::vector<int>& local_Qp,
        std::vector<int>& local_Qi,
        std::vector<int>& Q_node_weights,///<[out] The node weights of the local quotient graph
        std::vector<int>& Q_node_to_global_Q_node///<[out] The local Q node to global Q node map
    );

    void local_permute_metis(int G_n, int* Gp, int* Gi,
        std::vector<int> & local_permutation);

    void local_permute_amd(int G_n, int* Gp, int* Gi,
        std::vector<int> & local_permutation);

    void local_permute_unity(int G_n, int* Gp, int* Gi,
        std::vector<int>& local_permutation);

    //Apply fill-in reducing ordering to each node in the decomposition tree
    void local_permute(int G_n, int* Gp, int* Gi,
        std::vector<int> & local_permutation);

    int post_order_offset_computation(int offset, int decomposition_node_id);
    void compute_sub_graph(
        std::vector<int>&         nodes,
        Eigen::SparseMatrix<int>& local_graph) const;

    void compute_sub_graphs(std::vector<SubGraph>& sub_graphs);

    void step1_compute_quotient_graph();
    void step2_create_decomposition_tree();
    void step3_CPU_compute_local_permutations();
    void step3_GPU_compute_local_permutations();
    void step4_assemble_final_permutation(std::vector<int>& perm);

    void compute_permutation(std::vector<int>& perm);
};
}