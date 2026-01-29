#pragma once

#include <assert.h>
#include <cooperative_groups.h>

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/query_dispatcher.cuh"
#include "rxmesh/kernels/shmem_allocator.cuh"
#include "rxmesh/types.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

template <uint32_t blockThreads>
struct Query
{
    Query(const Query&)            = delete;
    Query& operator=(const Query&) = delete;

    __device__ __inline__ Query(const Context& context,
                                const uint32_t pid = blockIdx.x);

    /**
     * @brief return the patch id associated with this instance
     */
    __device__ __inline__ int get_patch_id() const;

    /**
     * @brief return the patch info associated with this instance
     */
    __device__ __inline__ const PatchInfo& get_patch_info() const;

    /**
     * @brief compute the vertex valence
     */
    __device__ __inline__ void compute_vertex_valence(
        cooperative_groups::thread_block& block,
        ShmemAllocator&                   shrd_alloc);

    /**
     * @brief return the vertex valence. compute_vertex_valence has to be called
     * first
     * @param v vertex for which valence will be returned
     */
    __device__ __inline__ uint16_t vertex_valence(uint16_t v) const;

    /**
     * @brief return the vertex valence. compute_vertex_valence has to be called
     * first
     * @param vh vertex for which valence will be returned
     */
    __device__ __inline__ uint16_t vertex_valence(VertexHandle vh) const;

    /**
     * @brief The query dispatch function to be called by the whole block.
     */
    template <Op op, typename computeT>
    __device__ __inline__ void dispatch(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc,
                                        computeT        compute_op,
                                        const bool      oriented = false);

    /**
     * @brief The query dispatch function with active set predicate.
     */
    template <Op op, typename computeT, typename activeSetT>
    __device__ __inline__ void dispatch(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc,
                                        computeT        compute_op,
                                        activeSetT      compute_active_set,
                                        const bool      oriented        = false,
                                        const bool      allow_not_owned = false);

    /**
     * @brief run the query and prepare internal data structure to run the
     * computation on top of the queries
     */
    template <Op op>
    __device__ __inline__ void prologue(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc,
                                        const bool      oriented        = false,
                                        const bool      allow_not_owned = true);

    /**
     * @brief run the query and prepare internal data structure to run the
     * computation on top of the queries
     */
    template <Op op, typename activeSetT>
    __device__ __inline__ void prologue(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc,
                                        activeSetT      compute_active_set,
                                        const bool      oriented        = false,
                                        const bool      allow_not_owned = true);

    /**
     * @brief run the computation on the query operations.
     */
    template <typename computeT>
    __device__ __inline__ void run_compute(
        cooperative_groups::thread_block& block,
        computeT                          compute_op);

    /**
     * @brief return an iterator over the queries elements give a local index of
     * a source element
     */
    template <typename IteratorT>
    __device__ __inline__ IteratorT get_iterator(uint16_t local_id) const;

    /**
     * @brief free up shared memory allocated to store the query operations.
     */
    __device__ __inline__ void epilogue(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc);

   private:
    const Context&    m_context;
    const PatchInfo&  m_patch_info;
    uint32_t          m_shmem_before;
    uint32_t          m_num_src_in_patch;
    uint32_t*         m_s_participant_bitmask;
    uint32_t*         m_s_output_owned_bitmask;
    uint16_t*         m_s_output_offset;
    uint16_t*         m_s_output_value;
    uint8_t*          m_s_valence;
    LPHashTable       m_output_lp_hashtable;
    LPPair*           m_s_table;
    Op                m_op;
};

}  // namespace rxmesh

#include "rxmesh/query.inl"
