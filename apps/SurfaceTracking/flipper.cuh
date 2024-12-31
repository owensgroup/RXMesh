#pragma once

#include <Eigen/Dense>

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"

#include "link_condition.cuh"
#include "rxmesh/geometry_util.cuh"

template <typename T, uint32_t blockThreads>
__global__ static void  // __launch_bounds__(blockThreads)
classify_vertex(const rxmesh::Context                 context,
                const rxmesh::VertexAttribute<T>      position,
                const rxmesh::VertexAttribute<int8_t> is_vertex_bd,
                rxmesh::VertexAttribute<int8_t>       vertex_rank)
{
    // Compute rank of the quadric metric tensor at a vertex Determine the rank
    // of the primary space at the given vertex(see Jiao07)
    // Rank{1, 2, 3} == { smooth, ridge, peak}


    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto ranking = [&](VertexHandle vh, VertexIterator& iter) {
        // since we don't flip or collapse boundary vertices, we could skip
        // their rank computation
        if (is_vertex_bd(vh)) {
            vertex_rank(vh) = 4;
            return;
        }


        const vec3<T> v(position(vh, 0), position(vh, 1), position(vh, 2));

        VertexHandle qh = iter.back();

        vec3<T> q(position(qh, 0), position(qh, 1), position(qh, 2));

        Eigen::Matrix<T, 3, 3> A;
        A << 0, 0, 0, 0, 0, 0, 0, 0, 0;

        // for each triangle q-v-r where v is the vertex we want to compute its
        // displacement
        for (uint32_t i = 0; i < iter.size(); ++i) {

            const VertexHandle rh = iter[i];

            const vec3<T> r(position(rh, 0), position(rh, 1), position(rh, 2));

            // triangle normal
            const vec3<T> c = glm::cross(q - v, r - v);

            assert(glm::length(c) > std::numeric_limits<T>::min());

            const vec3<T> n = glm::normalize(c);

            // triangle area
            const T area = T(0.5) * glm::length(c);

            qh = rh;
            q  = r;

            for (int j = 0; j < 3; ++j) {
                A(0, j) += n[0] * area * n[j];
                A(1, j) += n[1] * area * n[j];
                A(2, j) += n[2] * area * n[j];
            }
        }

        // eigen decomp
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> eigen_solver(A);
        assert(eigen_solver.info() == Eigen::Success);
        Eigen::Matrix<T, 3, 1> eigenvalues = eigen_solver.eigenvalues();

        int8_t rank = 0;
        for (uint32_t i = 0; i < 3; ++i) {
            if (eigenvalues[i] > G_EIGENVALUE_RANK_RATIO * eigenvalues[2]) {
                ++rank;
            }
        }


        vertex_rank(vh) = rank;
    };

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, ranking, true);
}


template <typename T, uint32_t blockThreads>
__global__ static void  //__launch_bounds__(blockThreads)
edge_flip(rxmesh::Context                   context,
          rxmesh::VertexAttribute<T>        position,
          rxmesh::VertexAttribute<int8_t>   vertex_rank,
          rxmesh::EdgeAttribute<EdgeStatus> edge_status,
          rxmesh::VertexAttribute<int8_t>   is_vertex_bd,
          rxmesh::EdgeAttribute<int8_t>     is_edge_bd,
          const T                           edge_flip_min_length_change,
          const T                           max_volume_change,
          const T                           min_triangle_area,
          const T                           min_triangle_angle,
          const T                           max_triangle_angle)
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

    // lambda function that check if the edge should be flipped
    auto should_flip = [&](const EdgeHandle& eh, const VertexIterator& iter) {
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

                const VertexHandle ah = iter[0];
                const VertexHandle bh = iter[2];

                // check if ah or bh is boundary

                if (is_vertex_bd(ah) == 1 || is_vertex_bd(bh) == 1) {
                    edge_status(eh) = SKIP;
                    return;
                }

                const VertexHandle ch = iter[1];
                const VertexHandle dh = iter[3];

                bool flip_it = true;

                // avoid degenerate triangle
                if (ah == bh || bh == ch || ch == ah || ah == dh || bh == dh ||
                    ch == dh) {
                    flip_it = false;
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

                // change in length i.e., delaunay check
                if (flip_it) {
                    T current_length   = glm::distance2(va, vb);
                    T potential_length = glm::distance2(vc, vd);

                    if (potential_length >=
                        current_length - edge_flip_min_length_change) {
                        flip_it = false;
                    }
                }

                // control volume change
                if (flip_it) {
                    T vol = std::fabs(signed_volume(va, vb, vc, vd));

                    if (vol > max_volume_change) {
                        flip_it = false;
                    }
                }


                // if both old triangle normals agree before flipping, make sure
                // they agree after flipping
                if (flip_it) {
                    // old triangles normals
                    const vec3<T> n0 = tri_normal(va, vb, vc);
                    const vec3<T> n1 = tri_normal(va, vd, vb);

                    // new triangles normals
                    const vec3<T> n2 = tri_normal(vc, vd, vb);
                    const vec3<T> n3 = tri_normal(vc, va, vd);

                    if (glm::dot(n0, n1) > T(0)) {
                        if (glm::dot(n2, n3) < T(0)) {
                            flip_it = false;
                        }

                        if (glm::dot(n2, n0) < T(0)) {
                            flip_it = false;
                        }

                        if (glm::dot(n2, n1) < T(0)) {
                            flip_it = false;
                        }

                        if (glm::dot(n3, n0) < T(0)) {
                            flip_it = false;
                        }

                        if (glm::dot(n3, n1) < T(0)) {
                            flip_it = false;
                        }
                    }
                }

                // prevent creating degenerate/tiny triangles
                if (flip_it) {
                    if (tri_area(vc, vb, vd) < min_triangle_area ||
                        tri_area(vc, va, vd) < min_triangle_area) {
                        flip_it = false;
                    }
                }

                // control change in area
                if (flip_it) {
                    T old_area = tri_area(va, vc, vb) + tri_area(va, vd, vb);
                    T new_area = tri_area(vc, vb, vd) + tri_area(vc, va, vd);

                    if (std::fabs(old_area - new_area) > T(0.1) * old_area) {
                        flip_it = false;
                    }
                }

                // Don't flip unless both vertices are on a smooth patch
                if (flip_it) {
                    if (vertex_rank(ah) > 1 || vertex_rank(bh) > 1) {
                        flip_it = false;
                    }
                }

                // don't introduce a large or small angle
                if (flip_it) {
                    T min_angle0, min_angle1, max_angle0, max_angle1;
                    triangle_min_max_angle(vc, vb, vd, min_angle0, max_angle0);
                    triangle_min_max_angle(vc, va, vd, min_angle1, max_angle1);

                    if (min_angle0 < min_triangle_angle ||
                        min_angle1 < min_triangle_angle) {
                        flip_it = false;
                    }

                    if (max_angle0 > max_triangle_angle ||
                        max_angle1 > max_triangle_angle) {
                        flip_it = false;
                    }
                }

                if (flip_it) {
                    edge_mask.set(eh.local_id(), true);
                }
            }
        }
    };


    // 1. mark edge that we want to flip
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);

        const VertexIterator iter =
            query.template get_iterator<VertexIterator>(eh.local_id());

        should_flip(eh, iter);
    });
    block.sync();

    // 2. make sure that the two vertices opposite to a flipped edge are not
    // connected (link condition)
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