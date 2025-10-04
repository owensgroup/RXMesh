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


class RXMeshOrdering: public Ordering
{
private:
    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>> vertices;
    bool m_has_mesh = false;
    
    
public:
    virtual ~RXMeshOrdering(void);

    static RXMeshOrdering* create(const RXMESH_Ordering_Type type);

    virtual RXMESH_Ordering_Type type() const override;
    virtual std::string typeStr() const override;

    virtual void setGraph(int*              Gp,
                           int*              Gi,
                           int               G_N,
                           int               NNZ) override;

    virtual void setMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) override;
    
    virtual bool needsMesh() const override;

    virtual void compute_permutation(std::vector<int>& perm) override;
};

}  // namespace PARTH_SOLVER
