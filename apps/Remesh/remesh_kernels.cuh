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
__global__ static void edge_split(const rxmesh::Context            context,
                                  const rxmesh::VertexAttribute<T> coords,
                                  const T high_edge_len_sq)
{
    // EV for calc edge len
}

template <typename T, uint32_t blockThreads>
__global__ static void edge_collapse(const rxmesh::Context            context,
                                     const rxmesh::VertexAttribute<T> coords,
                                     const T low_edge_len_sq,
                                     const T high_edge_len_sq)
{
    // edge len EV
    // is valid collapse VE
}

template <typename T, uint32_t blockThreads>
__global__ static void edge_flip(const rxmesh::Context            context,
                                 const rxmesh::VertexAttribute<T> coords)
{
    // EVDiamond and valence 
}