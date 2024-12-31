#pragma once

#include <Eigen/Dense>

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"

#include "rxmesh/geometry_util.cuh"

#include "link_condition.cuh"


template <typename T, uint32_t blockThreads>
__global__ static void  //__launch_bounds__(blockThreads)
edge_collapse(rxmesh::Context                   context,
              rxmesh::VertexAttribute<T>        position,
              rxmesh::VertexAttribute<int8_t>   vertex_rank,
              rxmesh::EdgeAttribute<EdgeStatus> edge_status,
              rxmesh::VertexAttribute<int8_t>   is_vertex_bd,
              rxmesh::EdgeAttribute<int8_t>     is_edge_bd,
              const T                           collapser_min_edge_length,
              const T                           max_volume_change,
              const T                           min_triangle_area,
              const T                           min_triangle_angle,
              const T                           max_triangle_angle)
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

    auto new_vertex_position = [&](VertexHandle& v_to_keep,
                                   VertexHandle& v_to_delete) {
        vec3<T> new_p;

        const vec3<T> p_keep(position(v_to_keep, 0),
                             position(v_to_keep, 1),
                             position(v_to_keep, 2));

        const vec3<T> p_delete(position(v_to_delete, 0),
                               position(v_to_delete, 1),
                               position(v_to_delete, 2));


        const int8_t v_keep_rank   = vertex_rank(v_to_keep);
        const int8_t v_delete_rank = vertex_rank(v_to_delete);

        if (v_keep_rank > v_delete_rank) {
            new_p = p_keep;
        } else if (v_delete_rank > v_keep_rank) {
            VertexHandle tmp = v_to_delete;
            v_to_delete      = v_to_keep;
            v_to_keep        = tmp;
            new_p            = p_delete;
        } else {
            // ranks are equal
            new_p = (p_keep + p_delete) * T(0.5);
        }
        return new_p;
    };


    // a bitmask that indicates which edge we want to flip
    // we also use it to mark updated edges (for edge_status)
    Bitmask edge_mask(cavity.patch_info().edges_capacity[0], shrd_alloc);
    edge_mask.reset(block);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // we use this bitmask to mark the other end of to-be-collapse edge during
    // checking for the link condition
    Bitmask v0_mask(cavity.patch_info().num_vertices[0], shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().num_vertices[0], shrd_alloc);

    // Precompute EVDiamond
    Query<blockThreads> query(context, pid);
    query.prologue<Op::EVDiamond>(block, shrd_alloc);
    block.sync();

    // lambda function that check if the edge should be collapsed
    auto should_collapse = [&](const EdgeHandle&     eh,
                               const VertexIterator& iter) {
        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices
        /*
            0a
          /  | \
         c3  |  1d
          \  |  /
            2b
        */
        const VertexHandle ah = iter[0];
        const VertexHandle bh = iter[2];
        const VertexHandle ch = iter[1];
        const VertexHandle dh = iter[3];

        assert(ah.is_valid() && bh.is_valid());

        // don't collapse boundary vertices
        if (ch.is_valid() && dh.is_valid() && is_edge_bd(eh) == 0 &&
            is_vertex_bd(ah) == 0 && is_vertex_bd(bh) == 0) {

            // vertices position
            const vec3<T> va(position(ah, 0), position(ah, 1), position(ah, 2));
            const vec3<T> vb(position(bh, 0), position(bh, 1), position(bh, 2));
            const vec3<T> vc(position(ch, 0), position(ch, 1), position(ch, 2));
            const vec3<T> vd(position(dh, 0), position(dh, 1), position(dh, 2));

            bool should_it = true;

            // prevent collapses on the boundary
            if (is_vertex_bd(ch) || is_vertex_bd(dh)) {
                should_it = false;
            }

            // degenerate cases
            if (ah == bh || ah == ch || ah == dh || bh == ch || bh == dh ||
                ch == dh) {
                should_it = false;
            }

            // don't collapse if the edge is long enough
            if (glm::distance2(va, vb) >= collapser_min_edge_length) {
                should_it = false;
            }


            if (should_it) {
                edge_mask.set(eh.local_id(), true);
            }
        }
    };

    // 1. mark edge that we want to collapse based on the edge length
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);

        if (edge_status(eh) == UNSEEN) {
            const VertexIterator iter =
                query.template get_iterator<VertexIterator>(eh.local_id());

            should_collapse(eh, iter);
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
            edge_status(eh) = SKIP;
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    // create the cavity
    if (cavity.prologue(block,
                        shrd_alloc,
                        position,
                        vertex_rank,
                        edge_status,
                        is_vertex_bd,
                        is_edge_bd)) {

        edge_mask.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);

            VertexHandle v0, v1;
            cavity.get_vertices(src, v0, v1);

            // decide on new vertex position
            vec3<T> new_p = new_vertex_position(v0, v1);

            const vec3<T> p0(position(v0, 0), position(v0, 1), position(v0, 2));
            const vec3<T> p1(position(v1, 0), position(v1, 1), position(v1, 2));

            // check if the new triangles will be bad i.e., will have normal
            // inversion, will have tiny area, will have bad angles
            bool is_bad = false;

            // only if the edge not so tiny
            if (glm::distance2(p0, p1) > std::numeric_limits<T>::epsilon()) {

                for (uint16_t i = 0; i < size; ++i) {

                    const uint16_t j = (i + 1) % size;

                    const VertexHandle vi = cavity.get_cavity_vertex(c, i);
                    const VertexHandle vj = cavity.get_cavity_vertex(c, j);

                    const vec3<T> pi(
                        position(vi, 0), position(vi, 1), position(vi, 2));
                    const vec3<T> pj(
                        position(vj, 0), position(vj, 1), position(vj, 2));

                    // the new triangle will be pi-pj-new_p

                    const vec3<T> n_new = tri_normal(pi, pj, new_p);
                    const vec3<T> n_0   = tri_normal(pi, pj, p0);
                    const vec3<T> n_1   = tri_normal(pi, pj, p1);

                    const T area_new = tri_area(pi, pj, new_p);

                    T min_ang_new, max_ang_new;
                    triangle_min_max_angle(
                        pi, pj, new_p, min_ang_new, max_ang_new);

                    if (area_new < min_triangle_area ||
                        min_ang_new < min_triangle_angle ||
                        max_ang_new > max_triangle_angle ||
                        glm::dot(n_new, n_0) < 1e-5 ||
                        glm::dot(n_new, n_1) < 1e-5) {
                        is_bad = true;
                        break;
                    }
                }
            }

            if (is_bad) {
                // roll back
                cavity.recover(src);
                // mark this edge as SKIP because 1) if all cavities in this
                // patch are successful, then we want to indicate that this
                // edge is okay and should not be attempted again
                // 2) if we have to rollback all changes in this patch, we still
                // don't want to attempt this edge since we know that it creates
                // short edges
                edge_status(src) = SKIP;
            } else {

                const VertexHandle new_v = cavity.add_vertex();

                if (new_v.is_valid()) {

                    position(new_v, 0) = new_p[0];
                    position(new_v, 1) = new_p[1];
                    position(new_v, 2) = new_p[2];

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
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (edge_mask(eh.local_id()) || cavity.is_recovered(eh)) {
                edge_status(eh) = ADDED;
            }
        });
    }
}