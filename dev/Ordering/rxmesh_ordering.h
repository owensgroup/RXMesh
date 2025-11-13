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

    virtual void setMesh(const double* V_data, int V_rows, int V_cols,
                        const int* F_data, int F_rows, int F_cols) override;
    
    virtual bool needsMesh() const override;

    virtual void compute_permutation(std::vector<int>& perm) override;
    virtual void add_record(std::string save_address, std::map<std::string, double> extra_info, std::string mesh_name) override;
};

}  // namespace PARTH_SOLVER
