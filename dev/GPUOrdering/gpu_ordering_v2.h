//
// Created by behrooz on 2025-10-07.
//
#include "rxmesh/rxmesh_static.h"
#include "Eigen/Core"
#include "Eigen/Sparse"

namespace RXMESH_SOLVER {
class GPUOrdering_V2
{
public:
    GPUOrdering_V2();
    ~GPUOrdering_V2();

    struct DecompositionNode
    {
        int left_node_idx = -1;
        int right_node_idx = -1;
        int node_id = -1;
        int parent_idx = -1;
        int level = -1;

        std::vector<int> dofs;
        std::vector<int> local_new_labels;
        bool is_initialized = false;
        int patch_id = -1; //Only the leaves have a single patch or empty patch assigned to them

        bool isLeaf() const
        {
            if (left_node_idx == -1 && right_node_idx == -1)
                return true;
            return false;
        }

        void init_node(int left_node_idx, int right_node_idx,
            int node_id, int parent_idx, int level,
            std::vector<int> & dofs, std::vector<int> & local_permutation, int patch_id)
        {
            this->left_node_idx = left_node_idx;
            this->right_node_idx = right_node_idx;
            this->node_id = node_id;
            this->parent_idx = parent_idx;
            this->level = level;
            this->dofs = dofs;
            assert(local_permutation.size() == dofs.size());
            this->local_new_labels.resize(local_permutation.size());
            for(int i = 0; i < local_permutation.size(); ++i) {
                this->local_new_labels[local_permutation[i]] = i;
            }
            this->is_initialized = true;
            this->patch_id = patch_id;
        }
    };

    struct PatchNode
    {
        int q_id;
        std::vector<int> nodes;
    };


    struct MaxMatchTree {
        std::vector<DecompositionNode> decomposition_nodes; // All the nodes of the tree
        std::vector<int> decomposition_node_offset; // The offset of the nodes in the tree for permutation
        std::vector<bool> is_separator; // The separator of the nodes in the tree

        void init_max_match_tree(int num_tree_nodes) {
            decomposition_nodes.resize(num_tree_nodes);
        }

        void get_graph_node_to_tree_node_map_per_level(std::vector<int>& graph_node_to_tree_node_map, int level) {
        }

        void get_patch_node_to_tree_node_map_per_level(std::vector<int>& patch_node_to_tree_node_map, int level) {
        }
    };

    int num_patches;
    std::string local_permute_method = "metis";
    std::vector<PatchNode> patch_nodes;
    std::vector<bool> is_separator;
    MaxMatchTree max_match_tree;
    int decomposition_max_level;
    int patch_size = 512;
    std::vector<int> node_to_patch;
    int G_n, G_nnz;
    int* Gp, *Gi;

    int Q_n;
    Eigen::SparseMatrix<int> Q;
    std::vector<int> Q_node_weights;
    std::vector<int> Q_perm;

    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>> vertices;

    void setGraph(int* Gp, int* Gi, int G_N, int NNZ);
    void setMesh(const double* V_data, int V_rows, int V_cols,
                 const int* F_data, int F_rows, int F_cols);

    void init_patches();
    void compute_permutation(std::vector<int>& perm);

    void refine_separator(std::vector<int>& part1_nodes,
        std::vector<int>& part2_nodes);

    void find_separator_basic(std::vector<int>& part1_nodes,
        std::vector<int>& part2_nodes,
        std::vector<int>& separator_nodes);

    void decompose(int decomposition_node_id,   ///[in] decomposition node id
        int decomposition_node_parent_id, /// <[in] decomposition node's parent id
        int decomposition_level,     ///<[in] The current level that we are dissecting
        std::vector<int> & decomposition_node_patches
    );

    void GPUOrdering_V2::compute_sub_graph(Eigen::SparseMatrix<int>& graph,
        std::vector<int>& graph_node_weights,
        Eigen::SparseMatrix<int>& sub_graph,
        std::vector<int>& local_node_weights,
        std::vector<int>& nodes) const;
    Eigen::SparseMatrix<int> compute_sub_graph(int* Gp, int* Gi, int G_N, int NNZ, std::vector<int>& nodes) const;

    void local_permute_metis(Eigen::SparseMatrix<int>& local_graph,
        std::vector<int> & local_permutation);

    void local_permute_amd(Eigen::SparseMatrix<int>& local_graph,
        std::vector<int> & local_permutation);

    void local_permute(Eigen::SparseMatrix<int>& local_graph,
        std::vector<int> & local_permutation);

    void compute_bipartition(Eigen::SparseMatrix<int>& quotient_graph, std::vector<int>& quotient_graph_node_weights,
        std::vector<int>& node_to_partition);


    void step1_create_quotient_graph();
    void step2_create_hierarchical_partitioning_and_permute();
    void step3_assemble_permutation();
};
}