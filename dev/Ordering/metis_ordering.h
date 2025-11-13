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
#include "ordering.h"

namespace RXMESH_SOLVER {


class MetisOrdering: public Ordering
{
public:
    virtual ~MetisOrdering(void);

    virtual RXMESH_Ordering_Type type() const override;
    virtual std::string typeStr() const override;

    virtual void setGraph(int*              Gp,
                           int*              Gi,
                           int               G_N,
                           int               NNZ) override;

    virtual void compute_permutation(std::vector<int>& perm) override;
    virtual void add_record(std::string save_address, std::map<std::string, double> extra_info, std::string mesh_name) override;
};

}  // namespace PARTH_SOLVER
