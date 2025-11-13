//
// Created by behrooz on 2025-10-07.
//
#include "rxmesh/rxmesh_static.h"
#include "Eigen/Core"
#include "Eigen/Sparse"

namespace RXMESH_SOLVER {
class GPUOrdering_V3
{
public:
    GPUOrdering_V3();
    ~GPUOrdering_V3();

    struct DecompositionNode
    {
        int left_node_idx = -1;
        int right_node_idx = -1;
        int node_id = -1;
        int parent_idx = -1;
        int level = -1;
        int offset = -1;

        std::vector<int> dofs;
        std::vector<int> local_new_labels;
        bool is_initialized = false;

        bool isLeaf() const
        {
            if (left_node_idx == -1 && right_node_idx == -1)
                return true;
            return false;
        }

        void init_node(int left_node_idx, int right_node_idx,
            int node_id, int parent_idx, int level,
            std::vector<int> & dofs)
        {
            this->left_node_idx = left_node_idx;
            this->right_node_idx = right_node_idx;
            this->node_id = node_id;
            this->parent_idx = parent_idx;
            this->level = level;
            this->dofs = dofs;
            this->is_initialized = true;
        }

        void set_local_permutation(std::vector<int> & local_permutation) {
            assert(local_permutation.size() == dofs.size());
#ifndef NDEBUG
            std::vector<bool> is_visited(dofs.size(), false);
            for (int i = 0; i < local_permutation.size(); i++) {
                assert(is_visited[local_permutation[i]] == false);
                is_visited[local_permutation[i]] = true;
            }
#endif
            this->local_new_labels.resize(local_permutation.size());
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

        void init_decomposition_tree(int num_decomposition_nodes, int decomposition_level) {
            decomposition_nodes.resize(num_decomposition_nodes);
            this->decomposition_level = decomposition_level;
        }
        int get_number_of_decomposition_nodes() {
            return decomposition_nodes.size();
        }
    };

    std::string local_permute_method = "amd";
    std::vector<int> node_to_patch;
    int num_patches = -1;
    DecompositionTree decomposition_tree;
    int patch_size = 512;
    int G_n, G_nnz;
    const int* Gp;
    const int* Gi;

    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>> vertices;

    double separator_ratio = 0.0;

    void setGraph(const int* Gp, const int* Gi, int G_N, int NNZ);
    void setMesh(const double* V_data, int V_rows, int V_cols,
                 const int* F_data, int F_rows, int F_cols);

    void init_patches();
    void compute_permutation(std::vector<int>& perm);

    void decompose();


    //This function changes the patch ids so that they are localized to the range [0, num_patches-1]
    void normalize_node_to_patch(std::vector<std::pair<int, int>>& node_to_patch);

    //This function computes the quotient graph from the node to patch map
    void compute_quotient_graph(std::vector<int>& assigned_dofs,///<[in] The assigned dofs
        Eigen::SparseMatrix<int>& quotient_graph,///<[out] The quotient graph
        std::vector<int>& quotient_graph_node_weights,///<[out] The node weights of the quotient graph
        std::vector<int>& global_to_local_patch_id///<[out] The global to local patch id map
    );

    //This function computes the sub graph from the nodes
    //The precondition is that the nodes are sorted
    void compute_sub_graph(std::vector<int>& nodes,
        Eigen::SparseMatrix<int>& local_graph) const;

    //Local permutation functions                                   
    void local_permute_metis(Eigen::SparseMatrix<int>& local_graph,
        std::vector<int> & local_permutation);

    void local_permute_amd(Eigen::SparseMatrix<int>& local_graph,
        std::vector<int> & local_permutation);

    void local_permute_unity(Eigen::SparseMatrix<int>& local_graph,
        std::vector<int>& local_permutation);

    void local_permute(Eigen::SparseMatrix<int>& local_graph,
        std::vector<int> & local_permutation);


    void refine_separator_with_metis_internals(
        int N, int* Gp, int* Gi, int* where /* in/out {0,1,2} */);

    //Main funciton for computing nodes
    void compute_separator(Eigen::SparseMatrix<int>& graph,///<[in] The local graph
        Eigen::SparseMatrix<int>& quotient_graph,///<[in] The local quotient graph
        std::vector<int>&         quotient_graph_node_weights,///<[in] The node weights of the local quotient graph
        std::vector<int>&         node_to_partition,///<[in] The node to partition map
        std::vector<int>&         separator_nodes,///<[out] The separator nodes
        std::vector<int>&         left_assigned_dofs,///<[out] The left assigned dofs
        std::vector<int>&         right_assigned_dofs///<[out] The right assigned dofs
    );

    //Helper function for assembling final permutation
    int post_order_offset_computation(int offset, int decomposition_node_id);
    
    

    //Main Steps
    void step1_compute_node_to_patch();
    void step2_create_decomposition_tree();
    void step3_compute_local_permutations();
    void step4_assemble_final_permutation(std::vector<int>& perm);
};
}