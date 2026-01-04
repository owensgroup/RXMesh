#pragma once

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/diff/candidate_pairs.h"

namespace rxmesh {
namespace detail {
template <typename ProblemT>
void add_ev_diamond_interaction(RXMeshStatic& rx, ProblemT& problem)
{
    int expected_vv_candidate_pairs = rx.get_num_edges();

    if (problem.vv_pairs.capacity() < expected_vv_candidate_pairs) {
        problem.vv_pairs.release();
        using HessMatT = typename ProblemT::HessMatT;

        problem.vv_pairs = CandidatePairsVV<HessMatT>(
            expected_vv_candidate_pairs, *problem.hess, rx.get_context());
    }

    problem.vv_pairs.reset();

    auto vv_pairs = problem.vv_pairs;

    rx.run_query_kernel<Op::EVDiamond, 256>(
        [=] __device__(const EdgeHandle      eh,
                       const VertexIterator& iter) mutable {
            if (iter[1].is_valid() && iter[3].is_valid()) {
                bool inserted = vv_pairs.insert(iter[1], iter[3]);
                assert(inserted);
            }
        });

    problem.update_hessian();
}
}  // namespace detail
}  // namespace rxmesh