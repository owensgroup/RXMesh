#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"
/**
 * initliaze the attribute for the degree related, could be mered with the
 * matching function
 */
template <typename T, uint32_t blockThreads>
__global__ static void init_attribute(const rxmesh::Context      context,
                                      rxmesh::VertexAttribute<T> nedges,
                                      rxmesh::VertexAttribute<T> adjwgt,
                                      rxmesh::EdgeAttribute<T>   ewgt)
{
    using namespace rxmesh;

    auto nd_lambda = [&](VertexHandle vh, EdgeIterator& ve_iter) {
        T ewgt_sum = 0;
        for (uint32_t e = 0; e < ve_iter.size(); ++e) {
            ewgt_sum += ewgt(ve_iter[e]);
        }

        nedges(vh) = ve_iter.size();
        adjwgt(vh) = ewgt_sum;
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VE>(block, shrd_alloc, nd_lambda);
}

template <uint32_t blockThreads, typename T>
__global__ static void heavy_edge_matching(const rxmesh::Context      context,
                                           rxmesh::VertexAttribute<T> vpair,
                                           rxmesh::EdgeAttribute<T>   ewgt)
{
    // For each vertex, loop over its edges. For each incident, compute its
    // length and add it to the vertex
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    // initiate the secondary query (EV)
    Query<blockThreads> ev_query(context);
    ev_query.prologue<Op::EV>(block, shrd_alloc);


    auto sum_edges = [&](const VertexHandle& vertex,
                         const EdgeIterator& eiter) {
        uint32_t vertex_patch = vertex.patch_id();

        for (uint16_t t = 0; t < eiter.size(); ++t) {
            // find the max weight vertex
            VertexHandle paired_v;
            uint32_t     paired_wgt = 0;
            uint32_t     curr_wgt   = 0;
            for (uint16_t i = 0; i < eiter.size(); ++i) {
                EdgeHandle curr_e   = eiter[i];
                uint32_t   curr_wgt = ewgt(curr_e);

                if (vpair(vertex) != 1 << 10) {
                    continue;
                }

                //  access the edge's (eiter[i]) two end vertices
                VertexHandle curr_v;
                VertexIterator viter =
                    ev_query.template get_iterator<VertexIterator>(
                        eiter.local(i));
                // note that  we use local index from the iterator ^^^^^^^ why?

                assert(viter.size() == 2);

                const VertexHandle vh0(viter[0]), vh1(viter[1]);

                // sanity check: one of the two end vertices should be the same
                // as vertex
                assert(vh0 == vertex || vh1 == vertex);

                // get the other end vertex
                if (vertex != vh0) {
                    curr_v = vh0;
                } else {
                    curr_v = vh1;
                }

                uint16_t    curr_patch = curr_v.patch_id();

                if (curr_patch != vertex_patch) {
                    continue;
                }

                // greedy, get the largest weight neighbor
                if (curr_wgt > paired_wgt)
                {
                    paired_wgt = curr_wgt;
                }
            }

            // atomicCAS(int* address, int compare, int val) --> (old ==
            // compare ? val : old)
            atomicCAS(&vpair(paired_v), 1 << 10, vertex);

            if (vpair(paired_v) == vertex) {
                // write success
                if (vpair(vertex) == paired_v || vpair(vertex) == 1 << 10) {
                    // pick each other or not in the middle of a chain
                    vpair(vertex) = paired_v;
                    break;
                } else {
                    // write success but in the middle of a chain
                    //  TODO: need to be purged
                }
            } else {
                // write fail
                // accept the incoming scuuessful write
                if (vpair(vertex) != 1 << 10) {
                    paired_v        = vpair(vertex);
                    vpair(paired_v) = vertex;
                }
            }
        }
    };

    // the primary query
    Query<blockThreads> ve_query(context);
    ve_query.dispatch<Op::VE>(block, shrd_alloc, sum_edges);
}