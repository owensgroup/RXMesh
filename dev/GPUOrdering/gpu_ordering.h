//
// Created by behrooz on 2025-10-07.
//
#include "rxmesh/rxmesh_static.h"
#include "Eigen/Core"
#include "Eigen/Sparse"

namespace RXMESH_SOLVER {
class GPUOrdering
{
public:
    GPUOrdering();
    ~GPUOrdering();

    struct QuotientNode
    {
        int q_id;
        int offset;
        std::vector<int> nodes;
        std::vector<int> permuted_local_labels;

    };

    std::vector<QuotientNode> quotient_nodes;
    std::vector<int> map_graph_to_quotient_node;
    std::vector<int> global_to_local;

    int num_patches = 128;
    std::vector<int> node_to_patch;
    std::vector<bool> is_boundary_vertex;
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

    void step1_find_boundary_vertices();
    void step2_create_quotient_graph();
    void step3_compute_quotient_permutation();
    void step4_compute_patch_permutation();
    void step5_map_patch_permutation_to_vertex_permutation(std::vector<int>& perm);
};
}