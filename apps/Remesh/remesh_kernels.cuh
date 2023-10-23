#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"

#include <glm/glm.hpp>

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
                                  const T high_edge_len_sq)
{
    // EV for calc edge len
}

template <typename T, uint32_t blockThreads>
__global__ static void edge_collapse(rxmesh::Context                  context,
                                     const rxmesh::VertexAttribute<T> coords,
                                     const T low_edge_len_sq,
                                     const T high_edge_len_sq)
{
    // edge len EV
    // is valid collapse VE
}

template <typename T, uint32_t blockThreads>
__global__ static void edge_flip(rxmesh::Context                  context,
                                 const rxmesh::VertexAttribute<T> coords)
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

    Query<blockThreads> query(context, cavity.patch_id());
    query.compute_vertex_valence(block, shrd_alloc);

    auto should_flip = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices


        // we use the local index since we are only interested in the valence
        // which computed on the local index space
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


        const int deviation_post = std::abs(valence_a - 1 - target_valence) +
                                   std::abs(valence_b - 1 - target_valence) +
                                   std::abs(valence_c + 1 - target_valence) +
                                   std::abs(valence_d + 1 - target_valence);
        if (deviation_pre > deviation_post) {
            cavity.create(eh);
        }
    };


    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_flip);
    block.sync();

    if (cavity.prologue(block, shrd_alloc, coords)) {

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            DEdgeHandle new_edge = cavity.add_edge(
                cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));

            if (new_edge.is_valid()) {
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