#pragma once
#include "../Remesh/link_condition.cuh"
#include "rxmesh/cavity_manager.cuh"

#include "secp_pair.h"

template <typename T, uint32_t blockThreads>
__global__ static void secp(rxmesh::Context             context,
                            rxmesh::VertexAttribute<T>  coords,
                            rxmesh::EdgeAttribute<bool> to_collapse)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager<blockThreads, CavityOp::EV> cavity(
        block, context, shrd_alloc, true);

    const uint32_t pid = cavity.patch_id();

    if (pid == INVALID32) {
        return;
    }

    // we first use this mask to set the edge we want to collapse (and then
    // filter them). Then after cavity.prologue, we reuse this bitmask to mark
    // the newly added edges
    Bitmask edge_mask(cavity.patch_info().edges_capacity[0], shrd_alloc);
    edge_mask.reset(block);

    // we use this bitmask to mark the other end of to-be-collapse edge during
    // checking for the link condition
    Bitmask v0_mask(cavity.patch_info().num_vertices[0], shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().num_vertices[0], shrd_alloc);


    // Precompute EV
    Query<blockThreads> ev_query(context, pid);
    ev_query.prologue<Op::EV>(block, shrd_alloc);
    block.sync();

    // 1a) mark edge we want to collapse given to_collapse
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);

        // edge_mask.set(eh.local_id(), to_collapse(eh));
        if (to_collapse(eh)) {
            edge_mask.set(eh.local_id(), true);
        }
    });
    block.sync();

    // 2a) check edge link condition.
    link_condition(block,
                   cavity.patch_info(),
                   ev_query,
                   edge_mask,
                   v0_mask,
                   v1_mask,
                   0,
                   1);
    block.sync();

    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);
        if (edge_mask(eh.local_id())) {
            cavity.create(eh);
        }
    });
    block.sync();

    ev_query.epilogue(block, shrd_alloc);

    // create the cavity
    if (cavity.prologue(block, shrd_alloc, coords, to_collapse)) {
        edge_mask.reset(block);
        block.sync();

        // fill in the cavities
        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);

            // TODO handle boundary edges

            VertexHandle v0, v1;

            cavity.get_vertices(src, v0, v1);

            const VertexHandle new_v = cavity.add_vertex();

            if (new_v.is_valid()) {

                coords(new_v, 0) = (coords(v0, 0) + coords(v1, 0)) * T(0.5);
                coords(new_v, 1) = (coords(v0, 1) + coords(v1, 1)) * T(0.5);
                coords(new_v, 2) = (coords(v0, 2) + coords(v1, 2)) * T(0.5);


                DEdgeHandle e0 =
                    cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));

                if (e0.is_valid()) {
                    edge_mask.set(e0.local_id(), true);

                    const DEdgeHandle e_init = e0;

                    for (uint16_t i = 0; i < size; ++i) {
                        const DEdgeHandle e = cavity.get_cavity_edge(c, i);

                        const VertexHandle v_end =
                            cavity.get_cavity_vertex(c, (i + 1) % size);

                        const DEdgeHandle e1 =
                            (i == size - 1) ?
                                e_init.get_flip_dedge() :
                                cavity.add_edge(
                                    cavity.get_cavity_vertex(c, i + 1), new_v);

                        if (!e1.is_valid()) {
                            break;
                        }

                        if (i != size - 1) {
                            edge_mask.set(e1.local_id(), true);
                        }

                        const FaceHandle new_f = cavity.add_face(e0, e, e1);

                        if (!new_f.is_valid()) {
                            break;
                        }
                        e0 = e1.get_flip_dedge();
                    }
                }
            }
        });
    }

    cavity.epilogue(block);
    block.sync();
}

// template <typename View, typename InputIt>
template <typename T, uint32_t blockThreads>
__global__ static void compute_edge_priorities(
    rxmesh::Context                  context,
    const rxmesh::VertexAttribute<T> coords,
    PQViewT                          pq_view,
    size_t                           pq_num_bytes)
{
    using namespace rxmesh;
    namespace cg       = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();
    ShmemAllocator   shrd_alloc;

    Query<blockThreads> query(context);

    PriorityPairT* s_pairs =
        shrd_alloc.alloc<PriorityPairT>(query.get_patch_info().num_edges[0]);
    __shared__ int pair_counter;
    pair_counter = 0;

    auto edge_len = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        const VertexHandle v0 = iter[0];
        const VertexHandle v1 = iter[1];

        const vec3<T> p0 = coords.to_glm<3>(v0);
        const vec3<T> p1 = coords.to_glm<3>(v1);

        const T len2 = glm::distance2(p0, p1);

        assert(eh.patch_id() < (1 << 16));

        // repack the EdgeHandle into smaller 32 bits for
        // use with priority queue. Need to check elsewhere
        // that there are less than 2^16 patches.
        const uint32_t id32 =
            unique_id32(eh.local_id(), (uint16_t)eh.patch_id());

        const PriorityPairT p{len2, id32};

        int val_counter = atomicAdd(&pair_counter, 1);

        s_pairs[val_counter] = p;
    };

    auto block = cooperative_groups::this_thread_block();
    query.dispatch<Op::EV>(block, shrd_alloc, edge_len);
    block.sync();

    char* pq_shrd_mem = shrd_alloc.alloc(pq_num_bytes);
    pq_view.push(block, s_pairs, s_pairs + pair_counter, pq_shrd_mem);
}

template <uint32_t blockThreads>
__global__ static void pop_and_mark_edges_to_collapse(
    PQViewT                     pq_view,
    rxmesh::EdgeAttribute<bool> to_collapse,
    uint32_t                    pop_num_edges)
{
    // setup shared memory array to store the popped pairs
    //
    // device api pop pairs
    namespace cg = cooperative_groups;
    using namespace rxmesh;
    ShmemAllocator shrd_alloc;

    PriorityPairT* s_pairs = shrd_alloc.alloc<PriorityPairT>(blockThreads);

    char* pq_shrd_mem = shrd_alloc.alloc(pq_view.get_shmem_size(blockThreads));

    cg::thread_block g = cg::this_thread_block();

    pq_view.pop(g, s_pairs, s_pairs + blockThreads, pq_shrd_mem);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the index is within bounds
    if (tid < pop_num_edges) {
        // unpack the uid to get the patch and edge ids
        auto [patch_id, local_id] = unpack32(s_pairs[threadIdx.x].second);

        EdgeHandle eh(patch_id, LocalEdgeT(local_id));

        // use the eh to index into a passed in edge attribute
        to_collapse(eh) = true;
    }
}
