#pragma once

#define G_EIGENVALUE_RANK_RATIO 0.03

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;

#include <Eigen/Dense>

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"


template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    null_space_smooth_vertex(const rxmesh::Context            context,
                             const rxmesh::VertexAttribute<T> current_position,
                             rxmesh::VertexAttribute<T>       new_position)
{

    // TODO avoid moving boundary vertices
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto smooth = [&](VertexHandle vh, VertexIterator& iter) {
        const Vec3<T> v(current_position(vh, 0),
                        current_position(vh, 1),
                        current_position(vh, 2));

        VertexHandle qh = iter.back();

        Vec3<T> q(current_position(qh, 0),
                  current_position(qh, 1),
                  current_position(qh, 2));

        Eigen::Matrix<T, 3, 3> A;
        A << 0, 0, 0, 0, 0, 0, 0, 0, 0;


        // for each triangle q-v-r where v is the vertex we want to compute its
        // displacement
        for (uint32_t i = 0; i < iter.size(); ++i) {

            const VertexHandle rh = iter[i];

            const Vec3<T> r(current_position(rh, 0),
                            current_position(rh, 1),
                            current_position(rh, 2));

            // triangle normal
            const Vec3<T> c = glm::cross(q - v, r - v);

            if (glm::length(c) <= std::numeric_limits<T>::min()) {
                // TODO quick workaround boundary vertices. Should be removed
                // when we properly filter out boundary vertices
                qh = rh;
                q  = r;
                continue;
            }


            assert(glm::length(c) >= std::numeric_limits<T>::min());

            const Vec3<T> n = glm::normalize(c);

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

        T eigenvalues[3];
        eigenvalues[0] = eigen_solver.eigenvalues().real()[0];
        eigenvalues[1] = eigen_solver.eigenvalues().real()[1];
        eigenvalues[2] = eigen_solver.eigenvalues().real()[2];

        // compute basis for null space
        Eigen::Matrix<T, 3, 3> tt;
        tt << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        for (uint32_t i = 0; i < 3; ++i) {
            if (eigenvalues[i] < G_EIGENVALUE_RANK_RATIO * eigenvalues[2]) {
                tt(i, 0) = A(0, i);
                tt(i, 1) = A(1, i);
                tt(i, 2) = A(2, i);
            }
        }

        Eigen::Matrix<T, 3, 3> null_space_projection;
        null_space_projection << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        for (uint32_t row = 0; row < 3; ++row) {
            for (uint32_t col = 0; col < 3; ++col) {
                for (size_t i = 0; i < 3; ++i) {
                    null_space_projection(row, col) += tt(i, row) * tt(i, col);
                }
            }
        }

        Eigen::Matrix<T, 3, 1> t;  // displacement
        t << 0, 0, 0;
        T sum_areas = 0;


        // for each triangle p-v-r where v is the vertex we want to compute its
        // displacement
        VertexHandle ph = iter.back();

        Vec3<T> p(current_position(ph, 0),
                  current_position(ph, 1),
                  current_position(ph, 2));

        for (uint32_t i = 0; i < iter.size(); ++i) {

            const VertexHandle rh = iter[i];

            const Vec3<T> r(current_position(rh, 0),
                            current_position(rh, 1),
                            current_position(rh, 2));

            // triangle normal
            const Vec3<T> c = glm::cross(p - v, r - v);

            if (glm::length(c) <= std::numeric_limits<T>::min()) {
                // TODO quick workaround boundary vertices. Should be removed
                // when we properly filter out boundary vertices
                ph = rh;
                p  = r;
                continue;
            }


            assert(glm::length(c) >= std::numeric_limits<T>::min());

            // triangle area
            const T area = T(0.5) * glm::length(c);

            // centriod
            constexpr T third = T(1) / T(3);

            Vec3<T> center = (third * (v + p + r)) - v;

            sum_areas += area;

            t(0) += area * center[0];
            t(1) += area * center[1];
            t(2) += area * center[2];


            ph = rh;
            p  = r;
        }

        t = null_space_projection * t;
        t /= sum_areas;

        new_position(vh, 0) = v[0] + t(0);
        new_position(vh, 1) = v[1] + t(1);
        new_position(vh, 2) = v[2] + t(2);
    };

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, smooth, true);
}


template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    classify_vertex(const rxmesh::Context            context,
                    const rxmesh::VertexAttribute<T> position,
                    rxmesh::VertexAttribute<int8_t>  vertex_rank)
{
    // Compute rank of the quadric metric tensor at a vertex Determine the rank
    // of the primary space at the given vertex(see Jiao07)
    // Rank{1, 2, 3} == { smooth, ridge, peak}

    // TODO avoid moving boundary vertices
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto ranking = [&](VertexHandle vh, VertexIterator& iter) {
        const Vec3<T> v(position(vh, 0), position(vh, 1), position(vh, 2));

        VertexHandle qh = iter.back();

        Vec3<T> q(position(qh, 0), position(qh, 1), position(qh, 2));

        Eigen::Matrix<T, 3, 3> A;
        A << 0, 0, 0, 0, 0, 0, 0, 0, 0;

        bool is_bd_vertex = false;

        // for each triangle q-v-r where v is the vertex we want to compute its
        // displacement
        for (uint32_t i = 0; i < iter.size(); ++i) {

            const VertexHandle rh = iter[i];

            const Vec3<T> r(position(rh, 0), position(rh, 1), position(rh, 2));

            // triangle normal
            const Vec3<T> c = glm::cross(q - v, r - v);

            if (glm::length(c) <= std::numeric_limits<T>::min()) {
                // TODO quick workaround boundary vertices. Should be removed
                // when we properly filter out boundary vertices
                is_bd_vertex = true;
                break;
            }


            assert(glm::length(c) >= std::numeric_limits<T>::min());

            const Vec3<T> n = glm::normalize(c);

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

        if (is_bd_vertex) {
            vertex_rank(vh) = 0;
            return;
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


/**
 * @brief Compute the signed volume of a tetrahedron.
 */
template <typename T>
__inline__ __device__ T signed_volume(const Vec3<T>& x0,
                                      const Vec3<T>& x1,
                                      const Vec3<T>& x2,
                                      const Vec3<T>& x3)
{
    // Equivalent to triple(x1-x0, x2-x0, x3-x0), six times the signed volume of
    // the tetrahedron. But, for robustness, we want the result (up to sign) to
    // be independent of the ordering. And want it as accurate as possible..
    // But all that stuff is hard, so let's just use the common assumption that
    // all coordinates are >0, and do something reasonably accurate in fp.

    // This formula does almost four times too much multiplication, but if the
    // coordinates are non-negative it suffers in a minimal way from
    // cancellation error.
    return (x0[0] * (x1[1] * x3[2] + x3[1] * x2[2] + x2[1] * x1[2]) +
            x1[0] * (x2[1] * x3[2] + x3[1] * x0[2] + x0[1] * x2[2]) +
            x2[0] * (x3[1] * x1[2] + x1[1] * x0[2] + x0[1] * x3[2]) +
            x3[0] * (x1[1] * x2[2] + x2[1] * x0[2] + x0[1] * x1[2]))

           - (x0[0] * (x2[1] * x3[2] + x3[1] * x1[2] + x1[1] * x2[2]) +
              x1[0] * (x3[1] * x2[2] + x2[1] * x0[2] + x0[1] * x3[2]) +
              x2[0] * (x1[1] * x3[2] + x3[1] * x0[2] + x0[1] * x1[2]) +
              x3[0] * (x2[1] * x1[2] + x1[1] * x0[2] + x0[1] * x2[2]));
}

template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    edge_flip(rxmesh::Context                       context,
              const rxmesh::VertexAttribute<T>      position,
              const rxmesh::VertexAttribute<int8_t> vertex_rank,
              rxmesh::EdgeAttribute<EdgeStatus>     edge_status,
              const T                               edge_flip_min_length_change,
              const T                               max_volume_change,
              const T                               min_triangle_area,
              const T                               min_triangle_angle,
              const T                               max_triangle_angle)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, false, false);


    if (cavity.patch_id() == INVALID32) {
        return;
    }

    Bitmask is_updated(cavity.patch_info().edges_capacity[0], shrd_alloc);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // for each edge we want to flip, we add its id in one of its opposite
    // vertices along with the other opposite vertex
    uint16_t* v_info =
        shrd_alloc.alloc<uint16_t>(2 * cavity.patch_info().num_vertices[0]);
    fill_n<blockThreads>(
        v_info, 2 * cavity.patch_info().num_vertices[0], uint16_t(INVALID16));

    // a bitmask that indicates which edge we want to flip
    Bitmask e_flip(cavity.patch_info().num_edges[0], shrd_alloc);
    e_flip.reset(block);

    auto tri_normal =
        [](const Vec3<T>& p0, const Vec3<T>& p1, const Vec3<T>& p2) {
            const Vec3<T> u = p1 - p0;
            const Vec3<T> v = p2 - p0;
            return glm::normalize(glm::cross(u, v));
        };

    auto tri_area =
        [](const Vec3<T>& p0, const Vec3<T>& p1, const Vec3<T>& p2) {
            const Vec3<T> u = p1 - p0;
            const Vec3<T> v = p2 - p0;
            return T(0.5) * glm::length(glm::cross(u, v));
        };

    auto angle = [](const Vec3<T>& l, const Vec3<T>& c, const Vec3<T>& r) {
        glm::vec3 ll = glm::normalize(l - c);
        glm::vec3 rr = glm::normalize(r - c);
        return glm::acos(glm::dot(rr, ll));
    };

    auto triangle_angles = [&](const Vec3<T>& a,
                               const Vec3<T>& b,
                               const Vec3<T>& c,
                               T&             angle_a,
                               T&             angle_b,
                               T&             angle_c) {
        angle_a = angle(b, a, c);
        angle_b = angle(c, b, a);
        angle_c = angle(a, c, b);
    };

    auto triangle_min_max_angle = [&](const Vec3<T>& a,
                                      const Vec3<T>& b,
                                      const Vec3<T>& c,
                                      T&             min_angle,
                                      T&             max_angle) {
        T angle_a, angle_b, angle_c;
        triangle_angles(a, b, c, angle_a, angle_b, angle_c);
        min_angle = std::min(angle_a, angle_b);
        min_angle = std::min(min_angle, angle_c);

        max_angle = std::max(angle_a, angle_b);
        max_angle = std::max(max_angle, angle_c);
    };

    auto should_flip = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices
        //    0
        //  / | \
        // 3  |  1
        // \  |  /
        //    2

        if (edge_status(eh) == UNSEEN) {
            // make sure it is not boundary edge
            if (iter[1].is_valid() && iter[3].is_valid()) {

                assert(iter.size() == 0);

                const VertexHandle ah = iter[0];
                const VertexHandle bh = iter[2];

                // TODO check if ah or bh is boundary

                const VertexHandle ch = iter[1];
                const VertexHandle dh = iter[3];

                bool flip_it = true;

                // avoid degenerate triangle
                if (ah == bh || bh == ch || ch == ah || ah == dh || bh == dh ||
                    ch == dh) {
                    flip_it = false;
                }

                // vertices position
                const Vec3<T> va(
                    position(ah, 0), position(ah, 1), position(ah, 2));

                const Vec3<T> vb(
                    position(bh, 0), position(bh, 1), position(bh, 2));

                const Vec3<T> vc(
                    position(ch, 0), position(ch, 1), position(ch, 2));

                const Vec3<T> vd(
                    position(dh, 0), position(dh, 1), position(dh, 2));

                // change in length i.e., delaunay check
                if (flip_it) {
                    T current_length   = glm::length2(va - vb);
                    T potential_length = glm::length2(vc - vd);

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

                // TODO double check on the normal computation
                assert(false);

                if (flip_it) {
                    // old triangles normals
                    const Vec3<T> n0 = tri_normal(va, vc, vb);
                    const Vec3<T> n1 = tri_normal(va, vb, vd);

                    // new triangles normals
                    const Vec3<T> n2 = tri_normal(vc, vb, vd);
                    const Vec3<T> n3 = tri_normal(vc, vd, va);

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
                    uint16_t v_c(iter.local(1)), v_d(iter.local(3));

                    if (::atomicCAS(v_info + 2 * v_c, INVALID16, v_d) ==
                        INVALID16) {
                        v_info[2 * v_c + 1] = eh.local_id();
                        e_flip.set(eh.local_id(), true);
                    } else {
                        if (::atomicCAS(v_info + 2 * v_d, INVALID16, v_c) ==
                            INVALID16) {
                            v_info[2 * v_d + 1] = eh.local_id();
                            e_flip.set(eh.local_id(), true);
                        }
                    }
                }
            } else {
                edge_status(eh) = OKAY;
            }
        }
    };

    // 1. mark edge that we want to flip
    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_flip);
    block.sync();

    
    // 2. make sure that the two vertices opposite to a flipped edge are not
    // connected
    auto check_edges = [&](const VertexHandle& vh, const VertexIterator& iter) {
        uint16_t opposite_v = v_info[2 * vh.local_id()];
        if (opposite_v != INVALID16) {
            bool is_valid = true;
            for (uint16_t v = 0; v < iter.size(); ++v) {
                if (iter.local(v) == opposite_v) {
                    is_valid = false;
                    break;
                }
            }
            if (!is_valid) {
                e_flip.reset(v_info[2 * vh.local_id() + 1], true);
            }
        }
    };
    query.dispatch<Op::VV>(block, shrd_alloc, check_edges);
    block.sync();


    // 3. create cavity for the surviving edges
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (e_flip(eh.local_id())) {
            cavity.create(eh);
        } else {
            edge_status(eh) = OKAY;
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    if (cavity.prologue(block, shrd_alloc, position, edge_status)) {

        is_updated.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            DEdgeHandle new_edge = cavity.add_edge(
                cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));


            if (new_edge.is_valid()) {
                is_updated.set(new_edge.local_id(), true);
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
            if (is_updated(eh.local_id())) {
                edge_status(eh) = ADDED;
            }
        });
    }
}