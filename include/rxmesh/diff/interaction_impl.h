#pragma once

#include "rxmesh/diff/candidate_pairs.h"
#include "rxmesh/rxmesh_static.h"

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

template <typename ProblemT>
void add_vf_pairs_to_vv_pairs(
    RXMeshStatic&                                  rx,
    ProblemT&                                      problem,
    CandidatePairsVF<typename ProblemT::HessMatT>& vf_pairs,
    CandidatePairsVV<typename ProblemT::HessMatT>& vv_pairs,
    FaceAttribute<VertexHandle>&                   face_interact_vertex)
{
    // interaction between F-V pairs leads to adding new blocks in the Hessian
    // where every vertex of the face F (i.e., v0, v1, v2) will be interacting
    // with V. So, we add blocks to the Hessian corresponding to the VV
    // interaction v0-V, v1-V, and v2-V

    // 1) every face store the vertex it interact with it
    int num_vf_pairs = vf_pairs.num_pairs();

    face_interact_vertex.reset(VertexHandle(), DEVICE);

    constexpr uint32_t blockThreads = 256;

    uint32_t blocks = DIVIDE_UP(num_vf_pairs, blockThreads);

    for_each_item<<<blocks, blockThreads>>>(
        num_vf_pairs, [=] __device__(int i) mutable {
            auto pair                         = vf_pairs.get_pair(i);
            face_interact_vertex(pair.second) = pair.first;
        });

    // 2) stage the new VV interaction vertices in vv_pairs
    rx.run_query_kernel<Op::FV, blockThreads>(
        [=] __device__(const FaceHandle      fh,
                       const VertexIterator& iter) mutable {
            VertexHandle interact_v = face_interact_vertex(fh);
            if (interact_v.is_valid()) {
                bool inserted = vv_pairs.insert(interact_v, iter[0]);
                assert(inserted);

                inserted = vv_pairs.insert(interact_v, iter[1]);
                assert(inserted);

                inserted = vv_pairs.insert(interact_v, iter[2]);
                assert(inserted);
            }
        });
}


}  // namespace detail
}  // namespace rxmesh