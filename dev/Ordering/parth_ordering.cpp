//
// Created by behrooz on 2025-09-29.
//

#include "parth_ordering.h"

#include "ordering.h"
#include "spdlog/spdlog.h"

namespace RXMESH_SOLVER {

ParthOrdering::~ParthOrdering()
{
}

void ParthOrdering::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    parth.setNDLevels(7);
    parth.setMesh(G_N, Gp, Gi);
}

void ParthOrdering::compute_permutation(std::vector<int>& perm)
{
    parth.computePermutation(perm);

    //Some analysis - should be commented for performance meaures
    computeRatioOfBoundaryVertices();
    computeTheStatisticsOfPatches();
}


RXMESH_Ordering_Type ParthOrdering::type() const
{
    return  RXMESH_Ordering_Type::METIS;
}
std::string ParthOrdering::typeStr() const
{
    return "METIS";
}

void ParthOrdering::computeRatioOfBoundaryVertices()
{
    int boundary_vertices = 0;
    for (auto& node: parth.hmd.HMD_tree) {
        if (!node.isLeaf()) {
            boundary_vertices+= node.DOFs.size();
        }
    }
    spdlog::info("Total number of boundary vertices: {}", boundary_vertices);
    spdlog::info("The ratio of boundary vertices to total vertices is {:.2f}%",
                 (boundary_vertices * 100.0) / parth.M_n);

}

void ParthOrdering::computeTheStatisticsOfPatches()
{
    int max_patch_size = 0;
    int min_patch_size = 1e9;
    double avg_patch_size = 0;
    for (auto& patch: parth.hmd.HMD_tree) {
        if (patch.isLeaf()) {
            int patch_size = patch.DOFs.size();
            if (patch_size > max_patch_size) {
                max_patch_size = patch_size;
            }
            if (patch_size < min_patch_size) {
                min_patch_size = patch_size;
            }
            avg_patch_size += patch_size;
        }
    }
    spdlog::info("Total number of patches including separators: {}", parth.hmd.HMD_tree.size());
    spdlog::info("The max patch size is {}", max_patch_size);
    spdlog::info("The min patch size is {}", min_patch_size);
    spdlog::info("The avg patch size is {:.2f}", avg_patch_size / parth.hmd.HMD_tree.size());
}


}