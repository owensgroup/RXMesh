#pragma once

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"

enum class EdgeSplitPredicate
{
    Length = 0,  // split large edges
    Angle  = 1,  // split large angles

};


template <typename T, uint32_t blockThreads>
__global__ static void  //__launch_bounds__(blockThreads)
split_edges(rxmesh::Context                   context,
            const rxmesh::VertexAttribute<T>  position,
            rxmesh::EdgeAttribute<EdgeStatus> edge_status,
            rxmesh::VertexAttribute<int8_t>   is_vertex_bd,
            rxmesh::EdgeAttribute<int8_t>     is_edge_bd,
            const T                           splitter_max_edge_length,
            const T                           min_triangle_area,
            const T                           min_triangle_angle,
            const T                           max_triangle_angle,
            const EdgeSplitPredicate          predicate)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, true, false);


    if (cavity.patch_id() == INVALID32) {
        return;
    }

    Bitmask is_updated(cavity.patch_info().edges_capacity[0], shrd_alloc);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    auto should_split = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices
        /*
            0
          / | \
         3  |  1
         \  |  /
            2
        */

        if (edge_status(eh) == UNSEEN) {
            // make sure it is not boundary edge

            if (iter[1].is_valid() && iter[3].is_valid() &&
                is_edge_bd(eh) == 0) {

                assert(iter.size() == 4);
                /*
                    a
                  / | \
                 c  |  d
                 \  |  /
                    b
                */

                const VertexHandle ah = iter[0];
                const VertexHandle bh = iter[2];

                const VertexHandle ch = iter[1];
                const VertexHandle dh = iter[3];

                bool split_it = true;

                // avoid degenerate triangle
                if (ah == bh || bh == ch || ch == ah || ah == dh || bh == dh ||
                    ch == dh) {
                    split_it = false;
                }

                // vertices position
                const vec3<T> va(
                    position(ah, 0), position(ah, 1), position(ah, 2));

                const vec3<T> vb(
                    position(bh, 0), position(bh, 1), position(bh, 2));

                const vec3<T> vc(
                    position(ch, 0), position(ch, 1), position(ch, 2));

                const vec3<T> vd(
                    position(dh, 0), position(dh, 1), position(dh, 2));


                // test the predicate
                if (split_it) {
                    if (predicate == EdgeSplitPredicate::Length) {
                        // is it a long edge
                        if (glm::distance2(va, vb) < splitter_max_edge_length) {
                            split_it = false;
                        }
                    } else if (predicate == EdgeSplitPredicate::Angle) {
                        // is it opposite to large angles
                        if (tri_angle(va, vc, vb) < max_triangle_angle &&
                            tri_angle(va, vd, vb) < max_triangle_angle) {
                            split_it = false;
                        }
                    } else {
                        assert(1 != 1);
                    }
                }

                // splitting degenerate triangles causes problems
                if (split_it) {
                    const T area0 = tri_area(va, vb, vc);
                    const T area1 = tri_area(va, vb, vd);
                    if (area0 < min_triangle_area ||
                        area1 < min_triangle_area) {
                        split_it = false;
                    }
                }

                // Check angles of new triangles
                if (split_it) {
                    // mid point (new) vertex
                    const vec3<T> ve = T(0.5) * (va + vb);

                    // current min and max angles
                    T cur_min1, cur_min2, cur_max1, cur_max2;
                    triangle_min_max_angle(va, vb, vc, cur_min1, cur_max1);
                    triangle_min_max_angle(va, vb, vd, cur_min2, cur_max2);

                    // new triangles angle
                    T min1, min2, min3, min4, max1, max2, max3, max4;

                    triangle_min_max_angle(va, ve, vc, min1, max1);
                    triangle_min_max_angle(vc, ve, vb, min2, max2);
                    triangle_min_max_angle(vd, vb, ve, min3, max3);
                    triangle_min_max_angle(vd, ve, va, min4, max4);

                    if (min1 < min_triangle_angle ||
                        min2 < min_triangle_angle ||
                        min3 < min_triangle_angle ||
                        min4 < min_triangle_angle) {
                        split_it = false;
                    }
                    if (split_it) {
                        // if new angle is greater than the allowed angle, and
                        // doesn't improve the current max angle, prevent the
                        // split

                        if (max1 > max_triangle_angle ||
                            max2 > max_triangle_angle ||
                            max3 > max_triangle_angle ||
                            max4 > max_triangle_angle) {

                            split_it = false;
                        }
                    }
                }


                if (split_it) {
                    cavity.create(eh);
                }
            } else {
                edge_status(eh) = SKIP;
            }
        }
    };

    // 1. mark edge that we want to flip
    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_split);
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    if (cavity.prologue(block,
                        shrd_alloc,
                        position,
                        edge_status,
                        is_vertex_bd,
                        is_edge_bd)) {

        is_updated.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            const VertexHandle v0 = cavity.get_cavity_vertex(c, 0);
            const VertexHandle v1 = cavity.get_cavity_vertex(c, 2);

            const VertexHandle new_v = cavity.add_vertex();

            if (new_v.is_valid()) {

                position(new_v, 0) =
                    T(0.5) * (position(v0, 0) + position(v1, 0));
                position(new_v, 1) =
                    T(0.5) * (position(v0, 1) + position(v1, 1));
                position(new_v, 2) =
                    T(0.5) * (position(v0, 2) + position(v1, 2));

                DEdgeHandle e0 =
                    cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));
                const DEdgeHandle e_init = e0;

                if (e0.is_valid()) {
                    is_updated.set(e0.local_id(), true);

                    for (uint16_t i = 0; i < size; ++i) {
                        const DEdgeHandle e = cavity.get_cavity_edge(c, i);

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
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (is_updated(eh.local_id())) {
                edge_status(eh) = ADDED;
            }
        });
    }
}