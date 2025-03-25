#pragma once

#include <glm/glm.hpp>
#include "rxmesh/cavity_manager2.cuh"
#include "rxmesh/query.cuh"

#include "link_condition.cuh"

template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;

template <typename T>
using Vec4 = glm::vec<4, T, glm::defaultp>;

template <typename T>
using Mat4 = glm::mat<4, 4, T, glm::defaultp>;

template <typename T>
__device__ __inline__ __host__ T
compute_cost(const Mat4<T>& quadric, const T x, const T y, const T z)
{
    // clang-format off
    return     quadric[0][0] * x * x + 
           2 * quadric[0][1] * x * y +
           2 * quadric[0][2] * x * z + 
           2 * quadric[0][3] * x +

               quadric[1][1] * y * y + 
           2 * quadric[1][2] * y * z +
           2 * quadric[1][3] * y +

               quadric[2][2] * z * z + 
           2 * quadric[2][3] * z +

               quadric[3][3];
    // clang-format on
}

template <typename T>
__device__ __inline__ void calc_edge_cost(
    const rxmesh::EdgeHandle&         e,
    const rxmesh::VertexHandle&       v0,
    const rxmesh::VertexHandle&       v1,
    const rxmesh::VertexAttribute<T>& coords,
    const rxmesh::VertexAttribute<T>& vertex_quadrics,
    rxmesh::EdgeAttribute<T>&         edge_cost,
    rxmesh::EdgeAttribute<T>&         edge_col_coord)
{
    Mat4<T> edge_quadric;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            edge_quadric[i][j] =
                vertex_quadrics(v0, i * 4 + j) + vertex_quadrics(v1, i * 4 + j);
        }
    }

    // the edge_quadric but with 4th row is identity
    Mat4<T> edge_quadric_3x3 = edge_quadric;
    edge_quadric_3x3[3][0]   = 0;
    edge_quadric_3x3[3][1]   = 0;
    edge_quadric_3x3[3][2]   = 0;
    edge_quadric_3x3[3][3]   = 1;

    if (std::abs(glm::determinant(edge_quadric_3x3)) >
        std::numeric_limits<T>::epsilon()) {

        Vec4<T> b(0, 0, 0, 1);

        auto invvv = glm::inverse(edge_quadric_3x3);

        Vec4<T> x = glm::transpose(glm::inverse(edge_quadric_3x3)) * b;

        edge_col_coord(e, 0) = x[0];
        edge_col_coord(e, 1) = x[1];
        edge_col_coord(e, 2) = x[2];

        edge_cost(e) =
            std::max(T(0.0), compute_cost(edge_quadric, x[0], x[1], x[2]));
    } else {
        const Vec3<T> p0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
        const Vec3<T> p1(coords(v1, 0), coords(v1, 1), coords(v1, 2));
        const Vec3<T> p2 = T(0.5) * (p0 + p1);

        const T e0 = compute_cost(edge_quadric, p0[0], p0[1], p0[2]);
        const T e1 = compute_cost(edge_quadric, p1[0], p1[1], p1[2]);
        const T e2 = compute_cost(edge_quadric, p2[0], p2[1], p2[2]);

        if (e0 < e1 && e1 < e2) {
            edge_cost(e)         = std::max(T(0.0), e0);
            edge_col_coord(e, 0) = p0[0];
            edge_col_coord(e, 1) = p0[1];
            edge_col_coord(e, 2) = p0[2];

        } else if (e1 < e2) {
            edge_cost(e)         = std::max(T(0.0), e1);
            edge_col_coord(e, 0) = p1[0];
            edge_col_coord(e, 1) = p1[1];
            edge_col_coord(e, 2) = p1[2];

        } else {
            edge_cost(e)         = std::max(T(0.0), e2);
            edge_col_coord(e, 0) = p2[0];
            edge_col_coord(e, 1) = p2[1];
            edge_col_coord(e, 2) = p2[2];
        }
    }
}

template <typename T>
__device__ __inline__ Vec4<T> compute_face_plane(const Vec3<T>& v0,
                                                 const Vec3<T>& v1,
                                                 const Vec3<T>& v2)
{
    Vec3<T> n = glm::cross(v1 - v0, v2 - v0);

    n = glm::normalize(n);

    Vec4<T> ret(n[0], n[1], n[2], -glm::dot(n, v0));

    return ret;
}

template <typename T, uint32_t blockThreads>
__global__ static void compute_edge_cost(
    const rxmesh::Context            context,
    const rxmesh::VertexAttribute<T> coords,
    const rxmesh::VertexAttribute<T> vertex_quadrics,
    rxmesh::EdgeAttribute<T>         edge_cost,
    rxmesh::EdgeAttribute<T>         edge_col_coord,
    rxmesh::Histogram<T>             histo)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    auto cost = [&](const EdgeHandle edge_id, const VertexIterator& ev) {
        const VertexHandle v0 = ev[0];
        const VertexHandle v1 = ev[1];

        calc_edge_cost(edge_id,
                       v0,
                       v1,
                       coords,
                       vertex_quadrics,
                       edge_cost,
                       edge_col_coord);

        atomicMin(histo.min_value(), edge_cost(edge_id));
        atomicMax(histo.max_value(), edge_cost(edge_id));
    };

    Query<blockThreads> query(context);
    query.dispatch<Op::EV>(block, shrd_alloc, cost);
}

template <typename T, uint32_t blockThreads>
__global__ static void compute_vertex_quadric_fv(
    const rxmesh::Context            context,
    const rxmesh::VertexAttribute<T> coords,
    rxmesh::VertexAttribute<T>       vertex_quadrics)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    auto calc_quadrics = [&](const FaceHandle      face_id,
                             const VertexIterator& fv) {
        // TODO avoid boundary faces

        const Vec3<T> v0(coords(fv[0], 0), coords(fv[0], 1), coords(fv[0], 2));
        const Vec3<T> v1(coords(fv[1], 0), coords(fv[1], 1), coords(fv[1], 2));
        const Vec3<T> v2(coords(fv[2], 0), coords(fv[2], 1), coords(fv[2], 2));

        const Vec4<T> face_plane = compute_face_plane(v0, v1, v2);

        const T a = face_plane[0];
        const T b = face_plane[1];
        const T c = face_plane[2];
        const T d = face_plane[3];

        Mat4<T> quadric;

        quadric[0][0] = a * a;
        quadric[0][1] = a * b;
        quadric[0][2] = a * c;
        quadric[0][3] = a * d;
        quadric[1][0] = b * a;
        quadric[1][1] = b * b;
        quadric[1][2] = b * c;
        quadric[1][3] = b * d;
        quadric[2][0] = c * a;
        quadric[2][1] = c * b;
        quadric[2][2] = c * c;
        quadric[2][3] = c * d;
        quadric[3][0] = d * a;
        quadric[3][1] = d * b;
        quadric[3][2] = d * c;
        quadric[3][3] = d * d;

        for (uint16_t v = 0; v < 3; ++v) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ::atomicAdd(&vertex_quadrics(fv[v], i * 4 + j),
                                quadric[i][j]);
                }
            }
        }
    };

    Query<blockThreads> query(context);
    query.dispatch<Op::FV>(block, shrd_alloc, calc_quadrics);
}

template <typename T, uint32_t blockThreads>
__global__ static void simplify(rxmesh::Context            context,
                                const rxmesh::Histogram<T> histo,
                                rxmesh::VertexAttribute<T> coords,
                                rxmesh::VertexAttribute<T> vertex_quadrics,
                                rxmesh::EdgeAttribute<T>   edge_cost,
                                rxmesh::EdgeAttribute<T>   edge_col_coord)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager2<blockThreads, CavityOp::EV> cavity(
        block, context, shrd_alloc, true);

    const uint32_t pid = cavity.patch_id();

    if (pid == INVALID32) {
        return;
    }

    uint16_t num_edges = cavity.patch_info().num_edges[0];

    uint8_t* edge_predicate = shrd_alloc.alloc<uint8_t>(num_edges);

    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        edge_predicate[e] = 0;
    }
    block.sync();


    auto collapse = [&](const VertexHandle& vh, const EdgeIterator& iter) {
        // TODO handle boundary edges

        T   min_cost      = std::numeric_limits<T>::max();
        int min_cost_edge = -1;
        for (int e = 0; e < iter.size(); ++e) {
            if (iter[e].patch_id() != pid) {
                continue;
            }
            T cost = edge_cost(iter[e]);
            if (cost < min_cost) {
                cost          = min_cost;
                min_cost_edge = iter[e].local_id();
            }
        }

        if (min_cost_edge != -1) {
            atomicAdd(edge_predicate + min_cost_edge, 1);
        }
    };

    Query<blockThreads> query(context, pid);
    query.dispatch<Op::VE>(block, shrd_alloc, collapse);
    block.sync();

    for_each_edge(cavity.patch_info(), [&](const EdgeHandle eh) {
        // if two vertices says that this edge is the edge with min cost
        // connected to them, then this edge is eligible for collapse
        if (edge_predicate[eh.local_id()] == 2) {
            cavity.create(eh);
        }
    });
    block.sync();


    // create the cavity
    if (cavity.prologue(block,
                        shrd_alloc,
                        coords,
                        vertex_quadrics,
                        edge_cost,
                        edge_col_coord)) {

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);

            VertexHandle v0, v1;

            cavity.get_vertices(src, v0, v1);

            const VertexHandle new_v = cavity.add_vertex();

            if (new_v.is_valid()) {

                coords(new_v, 0) = edge_col_coord(src, 0);
                coords(new_v, 1) = edge_col_coord(src, 1);
                coords(new_v, 2) = edge_col_coord(src, 2);

                for (int i = 0; i < 16; ++i) {
                    vertex_quadrics(new_v, i) =
                        vertex_quadrics(v0, i) + vertex_quadrics(v1, i);
                }

                DEdgeHandle e0 =
                    cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));

                if (e0.is_valid()) {
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

                        calc_edge_cost(e1.get_edge_handle(),
                                       v_end,
                                       new_v,
                                       coords,
                                       vertex_quadrics,
                                       edge_cost,
                                       edge_col_coord);

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
}


template <typename T, uint32_t blockThreads>
__global__ static void simplify_ev(rxmesh::Context            context,
                                   rxmesh::VertexAttribute<T> coords,
                                   const rxmesh::Histogram<T> histo,
                                   const int                  reduce_threshold,
                                   rxmesh::VertexAttribute<T> vertex_quadrics,
                                   rxmesh::EdgeAttribute<T>   edge_cost,
                                   rxmesh::EdgeAttribute<T>   edge_col_coord)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager2<blockThreads, CavityOp::EV> cavity(
        block, context, shrd_alloc, true);

    const uint32_t pid = cavity.patch_id();

    if (pid == INVALID32) {
        return;
    }

    // we first use this mask to set the edge we want to collapse (and then
    // filter them). Then after cavity.prologue, we reuse this bitmask to mark
    // the newly added edges
    Bitmask edge_mask(cavity.patch_info().edges_capacity, shrd_alloc);
    edge_mask.reset(block);

    // we use this bitmask to mark the other end of to-be-collapse edge during
    // checking for the link condition
    Bitmask v0_mask(cavity.patch_info().num_vertices[0], shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().num_vertices[0], shrd_alloc);


    // Precompute EV
    Query<blockThreads> ev_query(context, pid);
    ev_query.prologue<Op::EV>(block, shrd_alloc);


    // 1) mark edge we want to collapse
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);
        T cost = edge_cost(eh);
        if (histo.below_threshold(cost, reduce_threshold)) {
            edge_mask.set(eh.local_id(), true);
        }
    });
    block.sync();


    // 2) check edge link condition.
    link_condition(
        block, cavity.patch_info(), ev_query, edge_mask, v0_mask, v1_mask);


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
    if (cavity.prologue(block,
                        shrd_alloc,
                        coords,
                        vertex_quadrics,
                        edge_cost,
                        edge_col_coord)) {

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

                // coords(new_v, 0) = edge_col_coord(src, 0);
                // coords(new_v, 1) = edge_col_coord(src, 1);
                // coords(new_v, 2) = edge_col_coord(src, 2);

                coords(new_v, 0) = (coords(v0, 0) + coords(v1, 0)) * T(0.5);
                coords(new_v, 1) = (coords(v0, 1) + coords(v1, 1)) * T(0.5);
                coords(new_v, 2) = (coords(v0, 2) + coords(v1, 2)) * T(0.5);

                for (int i = 0; i < 16; ++i) {
                    vertex_quadrics(new_v, i) =
                        vertex_quadrics(v0, i) + vertex_quadrics(v1, i);
                }

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

                        calc_edge_cost(e1.get_edge_handle(),
                                       v_end,
                                       new_v,
                                       coords,
                                       vertex_quadrics,
                                       edge_cost,
                                       edge_col_coord);

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
}


template <typename T, uint32_t blockThreads>
__global__ static void edge_cost_histogram(
    const rxmesh::Context          context,
    const rxmesh::EdgeAttribute<T> edge_cost,
    const int                      num_bins,
    int*                           d_bins)
{
}