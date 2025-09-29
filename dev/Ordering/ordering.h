//
//  LinSysSolver.hpp
//  IPC
//
//  Created by Minchen Li on 6/30/18.
//
#pragma once


#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cholmod.h>

namespace RXMESH_SOLVER {

enum class RXMESH_Ordering_Type
{
    METIS,
    AMD,
    PARTH_METIS,
    PARTH_AMD,
    RXMESH_ND,
};

class Ordering
{
public:
    std::vector<int> perm;
    int* Gp = nullptr;
    int* Gi = nullptr;
    int  G_N = 0;
    int  G_NNZ = 0;

public:
    virtual ~Ordering(void) {};

    static Ordering* create(const RXMESH_Ordering_Type type);

    virtual RXMESH_Ordering_Type type() const = 0;

    virtual void setGraph(int*               Gp,
                           int*              Gi,
                           int               G_N,
                           int               NNZ) = 0;

    virtual void compute_permutation(std::vector<int>& perm) = 0;
};

}  // namespace PARTH_SOLVER
