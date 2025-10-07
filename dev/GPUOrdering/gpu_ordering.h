//
// Created by behrooz on 2025-10-07.
//
#include "rxmesh/rxmesh_static.h"
#include "Eigen/Core"

namespace RXMESH_SOLVER {
class GPUOrdering
{
public:
    GPUOrdering();
    ~GPUOrdering();

    int num_patches = 128;
    std::vector<int> node_to_patch;
    std::vector<bool> is_boundary_vertex;
    int G_n, G_nnz;
    int* Gp, *Gi;

    int Q_n;
    std::vector<int> Qp, Qi;

    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>> vertices;

    std::vector<std::vector<int>> patch_perm;

    void setGraph(int* Gp, int* Gi, int G_N, int NNZ);
    void setMesh(const double* V_data, int V_rows, int V_cols,
                 const int* F_data, int F_rows, int F_cols);

    void init_patches();
    void compute_permutation(std::vector<int>& perm);

    void step1_find_boundary_vertices();
    void step2_create_quotient_graph();
    void step3_compute_quotient_permutation();
    void step4_compute_patch_permutation();
    void step5_map_patch_permutation_to_vertex_permutation();
};
}