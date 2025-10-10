//
//  LinSysSolver.hpp
//  IPC
//
//  Created by Minchen Li on 6/30/18.
//
#pragma once


#include <Eigen/Core>
#include <parth//parth.h>
#include "ordering.h"

namespace RXMESH_SOLVER {


class ParthOrdering: public Ordering
{
public:
    PARTH::ParthAPI parth;
    virtual ~ParthOrdering(void);

    virtual RXMESH_Ordering_Type type() const override;
    virtual std::string typeStr() const override;

    virtual void setGraph(int*              Gp,
                           int*              Gi,
                           int               G_N,
                           int               NNZ) override;

    virtual void compute_permutation(std::vector<int>& perm) override;


    void computeRatioOfBoundaryVertices();
    void computeTheStatisticsOfPatches();
};

}  // namespace PARTH_SOLVER
