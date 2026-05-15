#include "rxmesh/query.h"

#include "rxmesh/iterator.cuh"
#include "rxmesh/kernels/util.cuh"

namespace rxmesh {

template <uint32_t blockThreads>
__device__ void Query<blockThreads>::compute_vertex_valence(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc)
{
    if (get_patch_id() == INVALID32) {
        return;
    }

    const uint16_t num_vertices =
        m_context.m_patches_info[m_pid].num_vertices[0];
    const uint16_t num_edges = m_context.m_patches_info[m_pid].num_edges[0];

    m_s_valence = shrd_alloc.alloc<uint8_t>(num_vertices);

    fill_n<blockThreads>(m_s_valence, num_vertices, uint8_t(0));
    block.sync();

    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        if (!m_context.m_patches_info[m_pid].is_deleted(LocalEdgeT(e))) {
            auto [v0, v1] =
                m_context.m_patches_info[m_pid].get_edge_vertices(e);
            atomicAdd(m_s_valence + v0, uint8_t(1));
            atomicAdd(m_s_valence + v1, uint8_t(1));
            assert(m_s_valence[v0] < 255);
            assert(m_s_valence[v1] < 255);
        }
    }
    block.sync();
}

// ---- Explicit instantiations
template struct Query<128>;
template struct Query<256>;
template struct Query<320>;
template struct Query<384>;
template struct Query<512>;
template struct Query<768>;
template struct Query<1024>;


#define RXMESH_QUERY_INSTANTIATE_PROLOGUE(BLOCK_THREADS)         \
    template void Query<BLOCK_THREADS>::prologue<Op::V>(         \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::E>(         \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::F>(         \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::VV>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::VE>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::VF>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::FV>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::FE>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::FF>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::EV>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::EE>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::EF>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template void Query<BLOCK_THREADS>::prologue<Op::EVDiamond>( \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);

RXMESH_QUERY_INSTANTIATE_PROLOGUE(128)
RXMESH_QUERY_INSTANTIATE_PROLOGUE(256)
RXMESH_QUERY_INSTANTIATE_PROLOGUE(384)
RXMESH_QUERY_INSTANTIATE_PROLOGUE(512)
RXMESH_QUERY_INSTANTIATE_PROLOGUE(768)
RXMESH_QUERY_INSTANTIATE_PROLOGUE(1024)

#undef RXMESH_QUERY_INSTANTIATE_PROLOGUE

#define RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(BLOCK_THREADS)                  \
    template VertexIterator                                                   \
    Query<BLOCK_THREADS>::get_iterator<VertexIterator>(uint16_t) const;       \
    template EdgeIterator Query<BLOCK_THREADS>::get_iterator<EdgeIterator>(   \
        uint16_t) const;                                                      \
    template DEdgeIterator Query<BLOCK_THREADS>::get_iterator<DEdgeIterator>( \
        uint16_t) const;                                                      \
    template FaceIterator Query<BLOCK_THREADS>::get_iterator<FaceIterator>(   \
        uint16_t) const;

RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(128)
RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(256)
RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(384)
RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(512)
RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(768)
RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(1024)

#undef RXMESH_QUERY_INSTANTIATE_GET_ITERATOR

}  // namespace rxmesh
