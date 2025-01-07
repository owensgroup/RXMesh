#pragma once

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/attribute.h"
#include "rxmesh/reduce_handle.h"

#include "rxmesh/diff/diff_query_kernel.cuh"


namespace rxmesh {

struct Term
{
};

template <typename HandleT,
          uint32_t blockThreads,
          Op       op,
          typename ScalarT,
          bool ProjectHess,
          int  VariableDim,
          typename LambdaT>
struct TemplatedTerm : public Term
{
    using T = typename ScalarT::PassiveType;

    TemplatedTerm(RXMeshStatic& rx, LambdaT t, bool oreinted) : term(t)
    {
        obj_func = rx.add_attribute<T, HandleT>("ObjFunc", 1);

        reducer = std::make_shared<ReduceHandle<T, HandleT>>(*obj_func);

        rx.prepare_launch_box({op},
                              lb_active,
                              (void*)detail::diff_kernel<blockThreads,
                                                         HandleT,
                                                         op,
                                                         ScalarT,
                                                         true,
                                                         ProjectHess,
                                                         VariableDim,
                                                         LambdaT>,
                              oreinted);

        rx.prepare_launch_box({op},
                              lb_passive,
                              (void*)detail::diff_kernel<blockThreads,
                                                         HandleT,
                                                         op,
                                                         ScalarT,
                                                         false,
                                                         ProjectHess,
                                                         VariableDim,
                                                         LambdaT>,
                              oreinted);
    }

    LambdaT term;

    std::shared_ptr<Attribute<T, HandleT>>    obj_func;
    std::shared_ptr<ReduceHandle<T, HandleT>> reducer;
    LaunchBox<blockThreads>                   lb_active;
    LaunchBox<blockThreads>                   lb_passive;
};
}  // namespace rxmesh