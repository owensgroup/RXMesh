#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"

#include <glm/glm.hpp>

#include "rxmesh/kernels/debug.cuh"

template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;

template <typename T, uint32_t blockThreads>
__global__ static void compute_average_edge_length(
    const rxmesh::Context            context,
    const rxmesh::VertexAttribute<T> coords,
    T*                               average_edge_length)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    auto sum_edge_len = [&](const EdgeHandle      edge_id,
                            const VertexIterator& ev) {
        const Vec3<T> v0(coords(ev[0], 0), coords(ev[0], 1), coords(ev[0], 2));
        const Vec3<T> v1(coords(ev[1], 0), coords(ev[1], 1), coords(ev[1], 2));

        T edge_len = glm::distance(v0, v1);

        ::atomicAdd(average_edge_length, edge_len);
    };

    Query<blockThreads> query(context);
    query.dispatch<Op::EV>(block, shrd_alloc, sum_edge_len);
}


template <typename T, uint32_t blockThreads>
__global__ static void edge_split(rxmesh::Context                  context,
                                  const rxmesh::VertexAttribute<T> coords,
                                  rxmesh::EdgeAttribute<int8_t>    updated,
                                  const T high_edge_len_sq)
{
    // EV for calc edge len

    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, true);


    if (cavity.patch_id() == INVALID32) {
        return;
    }

    //{
    //    if (threadIdx.x == 0) {
    //        printf("\n S patch= %u, before num_edges= %u, is_dirty= %d",
    //               cavity.patch_id(),
    //               context.m_patches_info[cavity.patch_id()].num_edges[0],
    //               int(context.m_patches_info[cavity.patch_id()].is_dirty()));
    //    }
    //    block.sync();
    //}

    Bitmask is_updated(cavity.patch_info().edges_capacity[0], shrd_alloc);

    auto should_split = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        if (updated(eh) == 0) {
            const Vec3<T> v0(
                coords(iter[0], 0), coords(iter[0], 1), coords(iter[0], 2));
            const Vec3<T> v1(
                coords(iter[1], 0), coords(iter[1], 1), coords(iter[1], 2));

            const T edge_len = glm::distance2(v0, v1);

            if (edge_len > high_edge_len_sq) {
                cavity.create(eh);
            }
        }
    };

    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EV>(block, shrd_alloc, should_split);
    block.sync();

    if (cavity.prologue(block, shrd_alloc, coords, updated)) {

        is_updated.reset(block);

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
                    for (uint16_t i = 0; i < size; ++i) {
                        const DEdgeHandle e = cavity.get_cavity_edge(c, i);

                        is_updated.set(e.local_id(), true);

                        const DEdgeHandle e1 =
                            (i == size - 1) ?
                                e_init.get_flip_dedge() :
                                cavity.add_edge(
                                    cavity.get_cavity_vertex(c, i + 1), new_v);
                        if (!e1.is_valid()) {
                            break;
                        }

                        is_updated.set(e1.local_id(), true);

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

    if (cavity.is_successful()) {
        detail::for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (is_updated(eh.local_id())) {
                updated(eh) = 1;
            }
        });
    }
    //{
    //    if (threadIdx.x == 0) {
    //        printf(
    //            "\n S patch= %u, after num_edges= %u, should_slice = %d, "
    //            "m_write_to_gmem= %d, is_dirty= %d",
    //            cavity.patch_id(),
    //            context.m_patches_info[cavity.patch_id()].num_edges[0],
    //            int(context.m_patches_info[cavity.patch_id()].should_slice),
    //            int(cavity.m_write_to_gmem),
    //            int(context.m_patches_info[cavity.patch_id()].is_dirty()));
    //    }
    //}
}

template <typename T, uint32_t blockThreads>
__global__ static void edge_collapse(rxmesh::Context                  context,
                                     const rxmesh::VertexAttribute<T> coords,
                                     rxmesh::EdgeAttribute<int8_t>    updated,
                                     const T low_edge_len_sq,
                                     const T high_edge_len_sq)
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

    //{
    //    if (threadIdx.x == 0) {
    //        printf("\n C patch= %u, before num_edges= %u, is_dirty= %d",
    //               cavity.patch_id(),
    //               context.m_patches_info[cavity.patch_id()].num_edges[0],
    //               int(context.m_patches_info[cavity.patch_id()].is_dirty()));
    //    }
    //    block.sync();
    //}

    Bitmask is_updated(cavity.patch_info().edges_capacity[0], shrd_alloc);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    Bitmask is_tri_valent(cavity.patch_info().num_vertices[0], shrd_alloc);
    is_tri_valent.reset(block);

    uint8_t* edge_valence =
        shrd_alloc.alloc<uint8_t>(cavity.patch_info().num_edges[0]);

    fill_n<blockThreads>(
        edge_valence, cavity.patch_info().num_edges[0], uint8_t(0));
    block.sync();

    for (uint16_t f = threadIdx.x; f < cavity.patch_info().num_faces[0];
         f += blockThreads) {
        if (!cavity.patch_info().is_deleted(LocalFaceT(f))) {
            uint16_t e0 = cavity.patch_info().fe[3 * f + 0].id >> 1;
            uint16_t e1 = cavity.patch_info().fe[3 * f + 1].id >> 1;
            uint16_t e2 = cavity.patch_info().fe[3 * f + 2].id >> 1;

            atomicAdd(edge_valence + e0, 1);
            atomicAdd(edge_valence + e1, 1);
            atomicAdd(edge_valence + e2, 1);
        }
    }
    block.sync();


    Query<blockThreads> ev_query(context, pid);
    ev_query.compute_vertex_valence(block, shrd_alloc);

    ev_query.prologue<Op::EV>(block, shrd_alloc, false, false);

    // mark a vertex if it is connected to a tri-valent vertex by an edge
    auto mark_vertices = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        if (eh.patch_id() == pid) {
            const uint16_t v0(iter.local(0)), v1(iter.local(1));
            if (ev_query.vertex_valence(v0) == 3) {
                is_tri_valent.set(v1, true);
            }
            if (ev_query.vertex_valence(v1) == 3) {
                is_tri_valent.set(v0, true);
            }
        }
    };
    ev_query.run_compute(block, mark_vertices);
    block.sync();

    auto should_collapse = [&](const EdgeHandle&     eh,
                               const VertexIterator& iter) {
        // only when both the two end of the edge are not tri-valent, we may do
        // the collapse. Otherwise, we will run into inconsistent topology
        // problem i.e., fold over
        if (eh.patch_id() == pid && updated(eh) == 0 &&
            !(is_tri_valent(iter.local(0)) && is_tri_valent(iter.local(1)))) {
            const VertexHandle v0(iter[0]), v1(iter[1]);

            const Vec3<T> p0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
            const Vec3<T> p1(coords(v1, 0), coords(v1, 1), coords(v1, 2));
            const T       edge_len_sq = glm::distance2(p0, p1);
            if (edge_len_sq < low_edge_len_sq &&
                edge_valence[eh.local_id()] < 3) {
                cavity.create(eh);
            }
        }
    };
    ev_query.run_compute(block, should_collapse);
    block.sync();

    ev_query.epilogue(block, shrd_alloc);
    block.sync();


    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    // create the cavity
    if (cavity.prologue(block, shrd_alloc, coords, updated)) {

        is_updated.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);

            VertexHandle v0, v1;

            cavity.get_vertices(src, v0, v1);

            const Vec3<T> p0(coords(v0, 0), coords(v0, 1), coords(v0, 2));


            // check if we will create a long edge
            bool long_edge = false;

            for (uint16_t i = 0; i < size; ++i) {
                const VertexHandle v1 = cavity.get_cavity_vertex(c, i);
                const Vec3<T> p1(coords(v1, 0), coords(v1, 1), coords(v1, 2));
                const T       edge_len_sq = glm::distance2(p0, p1);
                if (edge_len_sq > high_edge_len_sq) {
                    long_edge = true;
                    break;
                }
            }

            if (long_edge) {
                // roll back
                cavity.recover(src);
            } else {

                const VertexHandle new_v = cavity.add_vertex();

                if (new_v.is_valid()) {

                    coords(new_v, 0) = p0[0];
                    coords(new_v, 1) = p0[1];
                    coords(new_v, 2) = p0[2];

                    DEdgeHandle e0 =
                        cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));

                    if (e0.is_valid()) {
                        is_updated.set(e0.local_id(), true);

                        const DEdgeHandle e_init = e0;

                        for (uint16_t i = 0; i < size; ++i) {
                            const DEdgeHandle e = cavity.get_cavity_edge(c, i);

                            is_updated.set(e.local_id(), true);

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

                            is_updated.set(e1.local_id(), true);

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

    cavity.epilogue(block);
    block.sync();

    if (cavity.is_successful()) {
        detail::for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (is_updated(eh.local_id())) {
                updated(eh) = 1;
            }
        });
    }

    //{
    //    if (threadIdx.x == 0) {
    //        printf(
    //            "\n C patch= %u, after num_edges= %u, should_slice = %d, "
    //            "m_write_to_gmem= %d, is_dirty= %d",
    //            cavity.patch_id(),
    //            context.m_patches_info[cavity.patch_id()].num_edges[0],
    //            int(context.m_patches_info[cavity.patch_id()].should_slice),
    //            int(cavity.m_write_to_gmem),
    //            int(context.m_patches_info[cavity.patch_id()].is_dirty()));
    //    }
    //    block.sync();
    //}
}

template <typename T, uint32_t blockThreads>
__global__ static void edge_flip(rxmesh::Context                  context,
                                 const rxmesh::VertexAttribute<T> coords,
                                 rxmesh::EdgeAttribute<int8_t>    updated,
                                 rxmesh::VertexAttribute<int>     v_valence)
{
    // EVDiamond and valence
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, false);


    if (cavity.patch_id() == INVALID32) {
        return;
    }

    Bitmask is_updated(cavity.patch_info().edges_capacity[0], shrd_alloc);

    Query<blockThreads> query(context, cavity.patch_id());
    query.compute_vertex_valence(block, shrd_alloc);
    block.sync();

    detail::for_each_vertex(cavity.patch_info(), [&](VertexHandle vh) {
        v_valence(vh) = query.vertex_valence(vh.local_id());
    });

    block.sync();


    auto should_flip = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices


        // we use the local index since we are only interested in the
        // valence which computed on the local index space
        if (updated(eh) == 0 && iter[1].is_valid() && iter[3].is_valid()) {
            const uint16_t va = iter.local(0);
            const uint16_t vb = iter.local(2);

            const uint16_t vc = iter.local(1);
            const uint16_t vd = iter.local(3);

            // since we only deal with closed meshes without boundaries
            constexpr int target_valence = 6;

            // we don't deal with boundary edges
            assert(vc != INVALID16 && vd != INVALID16);

            const int valence_a = query.vertex_valence(va);
            const int valence_b = query.vertex_valence(vb);
            const int valence_c = query.vertex_valence(vc);
            const int valence_d = query.vertex_valence(vd);


            const int deviation_pre = std::abs(valence_a - target_valence) +
                                      std::abs(valence_b - target_valence) +
                                      std::abs(valence_c - target_valence) +
                                      std::abs(valence_d - target_valence);


            const int deviation_post =
                std::abs(valence_a - 1 - target_valence) +
                std::abs(valence_b - 1 - target_valence) +
                std::abs(valence_c + 1 - target_valence) +
                std::abs(valence_d + 1 - target_valence);
            if (deviation_pre > deviation_post) {
                cavity.create(eh);
            }
        }
    };


    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_flip);
    block.sync();


    if (cavity.prologue(block, shrd_alloc, coords, updated, v_valence)) {

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
        detail::for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (is_updated(eh.local_id())) {
                updated(eh) = 1;
            }
        });
    }
}

template <typename T, uint32_t blockThreads>
__global__ static void vertex_smoothing(const rxmesh::Context context,
                                        const rxmesh::VertexAttribute<T> coords,
                                        rxmesh::VertexAttribute<T> new_coords)
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

        T vq = glm::distance(v, q);

        Vec3<T> new_v(0.0, 0.0, 0.0);
        Vec3<T> v_normal(0.0, 0.0, 0.0);

        for (uint32_t i = 0; i < iter.size(); ++i) {
            // the current one ring vertex
            const VertexHandle r_id = iter[i];
            const Vec3<T> r(coords(r_id, 0), coords(r_id, 1), coords(r_id, 2));
            const T       vr = glm::distance(v, r);

            const Vec3<T> n = glm::cross(q - v, r - v) / (vr + vq);

            v_normal += n;

            new_v += r;

            q_id = r_id;
            q    = r;
            vq   = vr;
        }
        new_v /= T(iter.size());

        v_normal = glm::normalize(v_normal);

        new_v = new_v + glm::dot(v_normal, (v - new_v)) * v_normal;

        new_coords(v_id, 0) = new_v[0];
        new_coords(v_id, 1) = new_v[1];
        new_coords(v_id, 2) = new_v[2];
    };

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, smooth, true);
}