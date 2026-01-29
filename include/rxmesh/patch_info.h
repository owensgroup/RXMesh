#pragma once

#include <stdint.h>

#include "rxmesh/handle.h"
#include "rxmesh/local.h"
#include "rxmesh/patch_lock.h"
#include "rxmesh/patch_stash.h"
#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/macros.h"

#include "rxmesh/lp_hashtable.h"

namespace rxmesh {

/**
 * @brief PatchInfo stores the information needed for query operations in a
 * patch
 */
struct ALIGN(16) PatchInfo
{
    __device__ __host__ PatchInfo()
        : ev(nullptr),
          fe(nullptr),
          active_mask_v(nullptr),
          active_mask_e(nullptr),
          active_mask_f(nullptr),
          owned_mask_v(nullptr),
          owned_mask_e(nullptr),
          owned_mask_f(nullptr),
          num_vertices(nullptr),
          num_edges(nullptr),
          num_faces(nullptr),
          vertices_capacity(0),
          edges_capacity(0),
          faces_capacity(0),
          patch_id(INVALID32) {};

    __device__ __host__            PatchInfo(const PatchInfo& other) = default;
    __device__ __host__            PatchInfo(PatchInfo&&)            = default;
    __device__ __host__ PatchInfo& operator=(const PatchInfo&)       = default;
    __device__ __host__ PatchInfo& operator=(PatchInfo&&)            = default;
    __device__                     __host__ ~PatchInfo()             = default;

    // The topology information: edge incident vertices and face incident edges
    LocalVertexT* ev;
    LocalEdgeT*   fe;


    // Active bitmask where 1 indicates active/existing mesh element and 0
    // if the mesh element is deleted
    uint32_t *active_mask_v, *active_mask_e, *active_mask_f;

    // Owned bitmask where 1 indicates that the mesh element is owned by this
    // patch
    uint32_t *owned_mask_v, *owned_mask_e, *owned_mask_f;

    // Number of mesh elements in the patch
    uint16_t *num_vertices, *num_edges, *num_faces;

    // Capacity of v/e/f. This controls the allocations of ev, fe,
    // active_mask_v/e/f, owned_mask_v/e/f
    uint16_t vertices_capacity, edges_capacity, faces_capacity;

    // The index of this patch
    uint32_t patch_id;

    uint32_t color;

    // neighbor patches stash
    PatchStash patch_stash;

    // Hash table storing the mapping from local indices of ribbon (not-owned)
    // mesh elements to their owner patch and their local indices in their owner
    // patch
    LPHashTable lp_v, lp_e, lp_f;

    // a lock for the patch that should be acquired before modifying the patch
    // specially if more than one thread is updating the patch
    PatchLock lock;

    int* dirty;


    // for a sliced patch, the patch we sliced from
    uint32_t child_id;

    bool should_slice;

    /**
     * @brief update the dirty flag associated with this patch. The calling
     * thread should have locked the patch before updating
     * @return
     */
    __device__ __inline__ void set_dirty()
    {
#ifdef __CUDA_ARCH__
        assert(lock.is_locked());
        ::atomicAdd(dirty, 1u);
        __threadfence();
#endif
    }

    /**
     * @brief clear up the dirty flag
     */
    __device__ __inline__ void clear_dirty()
    {
        dirty[0] = 0;
    }

    /**
     * @brief check if the patch is dirty (ew!)
     */
    __device__ bool is_dirty() const;

    template <typename HandleT>
    __device__ __host__ HandleT find(const LPPair::KeyT key,
                                     const LPPair*      table,
                                     const LPPair*      stash,
                                     const PatchStash&  pstash) const;

    template <typename HandleT>
    __device__ __host__ HandleT find(const LPPair::KeyT key,
                                     const LPPair*      table = nullptr,
                                     const LPPair*      stash = nullptr) const;

    /**
     * @brief return the edge two vertices by reading them from memory as a
     * single 32-bit
     */
    __device__ std::pair<uint16_t, uint16_t> get_edge_vertices(
        const uint16_t e_id) const;

    /**
     * @brief convert an LPPair to a handle. The LPPair should be generated but
     * one of the LPHashTable stored here
     */
    template <typename HandleT>
    __device__ __host__ HandleT get_handle(const LPPair lp) const;

    /**
     * @brief return pointer to the number of elements corresponding  to the
     * handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ const uint16_t* get_num_elements() const;

    /**
     * @brief return the capacity corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ uint16_t get_capacity() const;

    /**
     * @brief return the active mask corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ const uint32_t* get_active_mask() const;

    /**
     * @brief return the active mask corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ uint32_t* get_active_mask();

    /**
     * @brief return the owned mask corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ const uint32_t* get_owned_mask() const;

    /**
     * @brief return the owned mask corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ uint32_t* get_owned_mask();

    /**
     * @brief return LP hashtable corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ const LPHashTable& get_lp() const;

    /**
     * @brief return LP hashtable corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ LPHashTable& get_lp();

    /**
     * @brief check if a vertex within this patch is owned by it
     */
    __device__ __host__ __inline__ bool is_owned(LocalVertexT vh) const
    {
        assert(vh.id != INVALID16);
        return detail::is_owned(vh.id, get_owned_mask<VertexHandle>());
    }

    /**
     * @brief check if an edge within this patch is owned by it
     */
    __device__ __host__ __inline__ bool is_owned(LocalEdgeT eh) const
    {
        assert(eh.id != INVALID16);
        return detail::is_owned(eh.id, get_owned_mask<EdgeHandle>());
    }

    /**
     * @brief check if a face within this patch is owned by it
     */
    __device__ __host__ __inline__ bool is_owned(LocalFaceT fh) const
    {
        assert(fh.id != INVALID16);
        return detail::is_owned(fh.id, get_owned_mask<FaceHandle>());
    }

    /**
     * @brief check if a vertex within this patch is deleted
     */
    __device__ __host__ __inline__ bool is_deleted(LocalVertexT vh) const
    {
        assert(vh.id != INVALID16);
        return detail::is_deleted(vh.id, get_active_mask<VertexHandle>());
    }

    /**
     * @brief check if an edge within this patch is deleted
     */
    __device__ __host__ __inline__ bool is_deleted(LocalEdgeT eh) const
    {
        assert(eh.id != INVALID16);
        return detail::is_deleted(eh.id, get_active_mask<EdgeHandle>());
    }

    /**
     * @brief check if a face within this patch is deleted
     */
    __device__ __host__ __inline__ bool is_deleted(LocalFaceT fh) const
    {
        assert(fh.id != INVALID16);
        return detail::is_deleted(fh.id, get_active_mask<FaceHandle>());
    }

    /**
     * @brief count number of owned active elements with type HandleT
     */
    template <typename HandleT>
    __host__ __device__ uint16_t get_num_owned() const;

    __host__ __device__ uint16_t count_num_owned(const uint32_t* owned_bitmask,
                                                 const uint32_t* active_bitmask,
                                                 const uint16_t  size) const;
};
}  // namespace rxmesh
