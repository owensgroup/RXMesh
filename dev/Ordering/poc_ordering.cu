//
// Created by behrooz on 2025-09-29.
//

#include "poc_ordering.h"



namespace RXMESH_SOLVER {

POCOrdering::~POCOrdering()
{
}

void POCOrdering::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp = Gp;
    this->Gi = Gi;
    this->G_N = G_N;
    this->G_NNZ = NNZ;
    // Also set the graph in the base GPUOrdering class
    gpu_order.setGraph(Gp, Gi, G_N, NNZ);
}

void POCOrdering::setMesh(const double* V_data, int V_rows, int V_cols,
                          const int* F_data, int F_rows, int F_cols)
{
    m_has_mesh = true;
    gpu_order.setMesh(V_data, V_rows, V_cols, F_data, F_rows, F_cols);
}

bool POCOrdering::needsMesh() const
{
    return true;
}

void POCOrdering::compute_permutation(std::vector<int>& perm)
{
    assert(m_has_mesh);
    gpu_order.init_patches();
    gpu_order.compute_permutation(perm);
}


RXMESH_Ordering_Type POCOrdering::type() const
{
    return RXMESH_Ordering_Type::POC_ND;
}

std::string POCOrdering::typeStr() const
{
    return "POC_ND";
}

}