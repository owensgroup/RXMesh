#include "rxmesh/query.h"

#include "rxmesh/iterator.cuh"
#include "rxmesh/kernels/util.cuh"

namespace rxmesh {

// ---- Explicit instantiations
template struct Query<128>;
template struct Query<256>;
template struct Query<320>;
template struct Query<384>;
template struct Query<512>;
template struct Query<768>;
template struct Query<1024>;


#define RXMESH_QUERY_INSTANTIATE_PROLOGUE(BLOCK_THREADS)         \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::V>(         \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::E>(         \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::F>(         \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::VV>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::VE>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::VF>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::FV>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::FE>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::FF>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::EV>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::EE>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::EF>(        \
        cooperative_groups::thread_block&,                       \
        ShmemAllocator&,                                         \
        const bool,                                              \
        const bool);                                             \
    template __device__ void Query<BLOCK_THREADS>::prologue<Op::EVDiamond>( \
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
    template __device__ VertexIterator                                        \
    Query<BLOCK_THREADS>::get_iterator<VertexIterator>(uint16_t) const;       \
    template __device__ EdgeIterator                                          \
    Query<BLOCK_THREADS>::get_iterator<EdgeIterator>(uint16_t) const;         \
    template __device__ DEdgeIterator                                         \
    Query<BLOCK_THREADS>::get_iterator<DEdgeIterator>(uint16_t) const;        \
    template __device__ FaceIterator                                          \
    Query<BLOCK_THREADS>::get_iterator<FaceIterator>(uint16_t) const;

RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(128)
RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(256)
RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(384)
RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(512)
RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(768)
RXMESH_QUERY_INSTANTIATE_GET_ITERATOR(1024)

#undef RXMESH_QUERY_INSTANTIATE_GET_ITERATOR

}  // namespace rxmesh
