#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"

#include "link_condition.cuh"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "rxmesh/kernels/debug.cuh"

template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;

template <typename T, uint32_t blockThreads>
__global__ static void stats_kernel(const rxmesh::Context            context,
                                    const rxmesh::VertexAttribute<T> coords,
                                    rxmesh::EdgeAttribute<T>         edge_len,
                                    rxmesh::VertexAttribute<int> vertex_valence)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    auto compute_edge_len = [&](const EdgeHandle eh, const VertexIterator& ev) {
        const Vec3<T> v0(coords(ev[0], 0), coords(ev[0], 1), coords(ev[0], 2));
        const Vec3<T> v1(coords(ev[1], 0), coords(ev[1], 1), coords(ev[1], 2));

        T len = glm::distance(v0, v1);

        edge_len(eh) = len;
    };

    Query<blockThreads> query(context);
    query.compute_vertex_valence(block, shrd_alloc);
    query.dispatch<Op::EV>(block, shrd_alloc, compute_edge_len);

    for_each_vertex(query.get_patch_info(), [&](const VertexHandle vh) {
        vertex_valence(vh) = query.vertex_valence(vh);
    });
}


template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    edge_split(rxmesh::Context                   context,
               const rxmesh::VertexAttribute<T>  coords,
               rxmesh::EdgeAttribute<EdgeStatus> edge_status,
               const T                           high_edge_len_sq,
               int*                              d_buffer)
{
    // EV for calc edge len

    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, true);


    //__shared__ int s_num_splits;
    // if (threadIdx.x == 0) {
    //    s_num_splits = 0;
    //}
    if (cavity.patch_id() == INVALID32) {
        return;
    }
    Bitmask is_updated(cavity.patch_info().edges_capacity[0], shrd_alloc);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    auto should_split = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        if (edge_status(eh) == UPDATE) {
            cavity.create(eh);
        } else if (edge_status(eh) == UNSEEN) {
            const Vec3<T> v0(
                coords(iter[0], 0), coords(iter[0], 1), coords(iter[0], 2));
            const Vec3<T> v1(
                coords(iter[1], 0), coords(iter[1], 1), coords(iter[1], 2));

            const T edge_len = glm::distance2(v0, v1);

            if (edge_len > high_edge_len_sq) {
                edge_status(eh) = UPDATE;
                cavity.create(eh);
            } else {
                edge_status(eh) = OKAY;
            }
        }
    };

    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EV>(block, shrd_alloc, should_split);
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);


    if (cavity.prologue(block, shrd_alloc, coords, edge_status)) {

        is_updated.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            const VertexHandle v0 = cavity.get_cavity_vertex(c, 0);
            const VertexHandle v1 = cavity.get_cavity_vertex(c, 2);

            const VertexHandle new_v = cavity.add_vertex();

            if (new_v.is_valid()) {

                coords(new_v, 0) = (coords(v0, 0) + coords(v1, 0)) / 2.0f;
                coords(new_v, 1) = (coords(v0, 1) + coords(v1, 1)) / 2.0f;
                coords(new_v, 2) = (coords(v0, 2) + coords(v1, 2)) / 2.0f;

                DEdgeHandle e0 =
                    cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));
                const DEdgeHandle e_init = e0;

                if (e0.is_valid()) {
                    is_updated.set(e0.local_id(), true);
                    //::atomicAdd(&s_num_splits, 1);
                    for (uint16_t i = 0; i < size; ++i) {
                        const DEdgeHandle e = cavity.get_cavity_edge(c, i);

                        // is_updated.set(e.local_id(), true);

                        const DEdgeHandle e1 =
                            (i == size - 1) ?
                                e_init.get_flip_dedge() :
                                cavity.add_edge(
                                    cavity.get_cavity_vertex(c, i + 1), new_v);
                        if (!e1.is_valid()) {
                            break;
                        }
                        if (i != size - 1) {
                            is_updated.set(e1.local_id(), true);
                        }

                        const FaceHandle f = cavity.add_face(e0, e, e1);
                        if (!f.is_valid()) {
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

    if (cavity.is_successful()) {
        // if (threadIdx.x == 0) {
        //    ::atomicAdd(d_buffer, s_num_splits);
        //}
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (is_updated(eh.local_id())) {
                edge_status(eh) = ADDED;
            }
        });
    }
}


template <uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    compute_valence(rxmesh::Context                        context,
                    const rxmesh::VertexAttribute<uint8_t> v_valence)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    Query<blockThreads> query(context);
    query.compute_vertex_valence(block, shrd_alloc);
    block.sync();

    for_each_vertex(query.get_patch_info(), [&](VertexHandle vh) {
        v_valence(vh) = query.vertex_valence(vh);
    });
}


template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    edge_collapse(rxmesh::Context                   context,
                  const rxmesh::VertexAttribute<T>  coords,
                  rxmesh::EdgeAttribute<EdgeStatus> edge_status,
                  const T                           low_edge_len_sq,
                  const T                           high_edge_len_sq,
                  int*                              d_buffer)
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

    // a bitmask that indicates which edge we want to flip
    // we also use it to mark updated edges (for edge_status)
    Bitmask edge_mask(cavity.patch_info().edges_capacity[0], shrd_alloc);
    edge_mask.reset(block);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // we use this bitmask to mark the other end of to-be-collapse edge during
    // checking for the link condition
    Bitmask v0_mask(cavity.patch_info().num_vertices[0], shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().num_vertices[0], shrd_alloc);

    // Precompute EV
    Query<blockThreads> query(context, pid);
    query.prologue<Op::EV>(block, shrd_alloc);
    block.sync();

    // 1. mark edge that we want to collapse based on the edge length
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);

        if (edge_status(eh) == UNSEEN) {
            const VertexIterator iter =
                query.template get_iterator<VertexIterator>(eh.local_id());

            const VertexHandle v0(iter[0]), v1(iter[1]);

            const Vec3<T> p0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
            const Vec3<T> p1(coords(v1, 0), coords(v1, 1), coords(v1, 2));
            const T       edge_len_sq = glm::distance2(p0, p1);

            if (edge_len_sq < low_edge_len_sq) {
                edge_mask.set(eh.local_id(), true);
            }
        }
    });
    block.sync();


    // 2. check link condition
    link_condition(
        block, cavity.patch_info(), query, edge_mask, v0_mask, v1_mask, 0, 1);
    block.sync();


    // 3. create cavity for the surviving edges
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (edge_mask(eh.local_id())) {
            cavity.create(eh);
        } else {
            edge_status(eh) = OKAY;
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    // create the cavity
    if (cavity.prologue(block, shrd_alloc, coords, edge_status)) {

        edge_mask.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            //::atomicAdd(&s_num_collapses, 1);
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);

            VertexHandle v0, v1;

            cavity.get_vertices(src, v0, v1);

            const Vec3<T> p0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
            const Vec3<T> p1(coords(v1, 0), coords(v1, 1), coords(v1, 2));

            const Vec3<T> new_p((p0[0] + p1[0]) * T(0.5),
                                (p0[1] + p1[1]) * T(0.5),
                                (p0[2] + p1[2]) * T(0.5));

            // check if we will create a long edge
            bool long_edge = false;

            for (uint16_t i = 0; i < size; ++i) {


                const VertexHandle vvv = cavity.get_cavity_vertex(c, i);

                const Vec3<T> vp(
                    coords(vvv, 0), coords(vvv, 1), coords(vvv, 2));

                const T edge_len_sq = glm::distance2(vp, new_p);

                if (edge_len_sq >= low_edge_len_sq) {
                    // long_edge = true;
                    break;
                }
            }

            if (long_edge) {
                // roll back
                cavity.recover(src);

                // mark this edge as OKAY because 1) if all cavities in this
                // patch are successful, then we want to indicate that this
                // edge is okay and should not be attempted again
                // 2) if we have to rollback all changes in this patch, we still
                // don't want to attempt this edge since we know that it creates
                // short edges
                edge_status(src) = OKAY;
            } else {

                const VertexHandle new_v = cavity.add_vertex();

                if (new_v.is_valid()) {

                    coords(new_v, 0) = new_p[0];
                    coords(new_v, 1) = new_p[1];
                    coords(new_v, 2) = new_p[2];

                    DEdgeHandle e0 =
                        cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));

                    if (e0.is_valid()) {
                        edge_mask.set(e0.local_id(), true);

                        const DEdgeHandle e_init = e0;

                        for (uint16_t i = 0; i < size; ++i) {
                            const DEdgeHandle e = cavity.get_cavity_edge(c, i);

                            // edge_mask.set(e.local_id(), true);

                            const VertexHandle v_end =
                                cavity.get_cavity_vertex(c, (i + 1) % size);

                            const DEdgeHandle e1 =
                                (i == size - 1) ?
                                    e_init.get_flip_dedge() :
                                    cavity.add_edge(
                                        cavity.get_cavity_vertex(c, i + 1),
                                        new_v);

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
            }
        });
    }
    block.sync();

    cavity.epilogue(block);
    block.sync();

    if (cavity.is_successful()) {
        // if (threadIdx.x == 0) {
        //    ::atomicAdd(d_buffer, s_num_collapses);
        //}
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (edge_mask(eh.local_id()) || cavity.is_recovered(eh)) {
                edge_status(eh) = ADDED;
            }
        });
    }
}

template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    edge_flip(rxmesh::Context                        context,
              const rxmesh::VertexAttribute<T>       coords,
              const rxmesh::VertexAttribute<uint8_t> v_valence,
              rxmesh::EdgeAttribute<EdgeStatus>      edge_status,
              int*                                   d_buffer)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, false, false);


    if (cavity.patch_id() == INVALID32) {
        return;
    }

    // a bitmask that indicates which edge we want to flip
    // we also used it to mark the new edges
    Bitmask edge_mask(cavity.patch_info().edges_capacity[0], shrd_alloc);
    edge_mask.reset(block);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // we use this bitmask to mark the other end of to-be-collapse edge during
    // checking for the link condition
    Bitmask v0_mask(cavity.patch_info().num_vertices[0], shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().num_vertices[0], shrd_alloc);

    // precompute EVDiamond
    Query<blockThreads> query(context, cavity.patch_id());
    query.prologue<Op::EVDiamond>(block, shrd_alloc);
    block.sync();


    // 1. mark edge that we want to flip
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);

        const VertexIterator iter =
            query.template get_iterator<VertexIterator>(eh.local_id());

        // only if the edge is not seen before and its not a boundary edge
        if (edge_status(eh) == UNSEEN && iter[1].is_valid() &&
            iter[3].is_valid()) {

            // iter[0] and iter[2] are the edge two vertices
            // iter[1] and iter[3] are the two opposite vertices


            // since we only deal with closed meshes without boundaries
            constexpr int target_valence = 6;


            const int valence_a = v_valence(iter[0]);
            const int valence_b = v_valence(iter[2]);
            const int valence_c = v_valence(iter[1]);
            const int valence_d = v_valence(iter[3]);

            // clang-format off
                const int deviation_pre =
                    (valence_a - target_valence) * (valence_a - target_valence) +
                    (valence_b - target_valence) * (valence_b - target_valence) +
                    (valence_c - target_valence) * (valence_c - target_valence) +
                    (valence_d - target_valence) * (valence_d - target_valence);
            // clang-format on

            // clang-format off
                const int deviation_post =
                    (valence_a - 1 - target_valence)*(valence_a - 1 - target_valence) +
                    (valence_b - 1 - target_valence)*(valence_b - 1 - target_valence) +
                    (valence_c + 1 - target_valence)*(valence_c + 1 - target_valence) +
                    (valence_d + 1 - target_valence)*(valence_d + 1 - target_valence);
            // clang-format on

            if (deviation_pre > deviation_post) {
                edge_mask.set(eh.local_id(), true);
            }
        }
    });
    block.sync();

    // 2. check link condition
    link_condition(
        block, cavity.patch_info(), query, edge_mask, v0_mask, v1_mask, 0, 2);
    block.sync();


    // 3. create cavity for the surviving edges
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (edge_mask(eh.local_id())) {
            cavity.create(eh);
        } else {
            edge_status(eh) = OKAY;
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    if (cavity.prologue(block, shrd_alloc, coords, edge_status)) {

        edge_mask.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            DEdgeHandle new_edge = cavity.add_edge(
                cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));


            if (new_edge.is_valid()) {
                edge_mask.set(new_edge.local_id(), true);
                cavity.add_face(cavity.get_cavity_edge(c, 0),
                                new_edge,
                                cavity.get_cavity_edge(c, 3));


                cavity.add_face(cavity.get_cavity_edge(c, 1),
                                cavity.get_cavity_edge(c, 2),
                                new_edge.get_flip_dedge());
            }
        });
    }
    block.sync();

    cavity.epilogue(block);
    block.sync();

    if (cavity.is_successful()) {
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (edge_mask(eh.local_id())) {
                edge_status(eh) = ADDED;
            }
        });
    }
}

template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    vertex_smoothing(const rxmesh::Context            context,
                     const rxmesh::VertexAttribute<T> coords,
                     rxmesh::VertexAttribute<T>       new_coords)
{
    // VV to compute vertex sum and normal
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto smooth = [&](VertexHandle v_id, VertexIterator& iter) {
        const Vec3<T> v(coords(v_id, 0), coords(v_id, 1), coords(v_id, 2));

        // compute both vertex normal and the new position
        // the new position is the average of the one-ring
        // while we iterate on the one ring to compute this new position, we
        // also compute the vertex normal
        // finally, we project the new position on the tangent plane of the
        // vertex (old position)

        // this is the last vertex in the one-ring (before r_id)
        VertexHandle q_id = iter.back();
        Vec3<T>      q(coords(q_id, 0), coords(q_id, 1), coords(q_id, 2));

        Vec3<T> new_v(0.0, 0.0, 0.0);
        Vec3<T> v_normal(0.0, 0.0, 0.0);

        T w = 0.0;

        for (uint32_t i = 0; i < iter.size(); ++i) {
            // the current one ring vertex
            const VertexHandle r_id = iter[i];

            const Vec3<T> r(coords(r_id, 0), coords(r_id, 1), coords(r_id, 2));

            const Vec3<T> c = glm::cross(q - v, r - v);

            const T area = glm::length(c) / T(2.0);
            w += area;

            const Vec3<T> n = glm::normalize(c) * area;

            v_normal += n;

            new_v += r;

            q_id = r_id;
            q    = r;
        }
        new_v /= T(iter.size());

        v_normal /= w;

        if (glm::length2(v_normal) < 1e-6) {
            new_v = v;
        } else {
            v_normal = glm::normalize(v_normal);

            new_v = new_v + (glm::dot(v_normal, (v - new_v)) * v_normal);
        }

        new_coords(v_id, 0) = new_v[0];
        new_coords(v_id, 1) = new_v[1];
        new_coords(v_id, 2) = new_v[2];
    };

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, smooth, true);
}