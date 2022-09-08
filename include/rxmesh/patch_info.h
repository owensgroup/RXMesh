#pragma once
#include <stdint.h>

#include "rxmesh/local.h"
#include "rxmesh/lp_hashtable.cuh"
#include "rxmesh/patch_stash.cuh"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/bitmask_util.h"

namespace rxmesh {

/**
 * @brief PatchInfo stores the information needed for query operations in a
 * patch
 */
struct ALIGN(16) PatchInfo
{
    PatchInfo()
        : ev(nullptr),
          fe(nullptr),
          active_mask_v(nullptr),
          active_mask_e(nullptr),
          active_mask_f(nullptr),
          owned_mask_v(nullptr),
          owned_mask_e(nullptr),
          owned_mask_f(nullptr),
          num_vertices(0),
          num_edges(0),
          num_faces(0),
          vertices_capacity(0),
          edges_capacity(0),
          faces_capacity(0),
          patch_id(INVALID32){};

    PatchInfo(const PatchInfo& other) = default;
    PatchInfo(PatchInfo&&)            = default;
    PatchInfo& operator=(const PatchInfo&) = default;
    PatchInfo& operator=(PatchInfo&&) = default;
    ~PatchInfo()                      = default;

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
    uint16_t *vertices_capacity, *edges_capacity, *faces_capacity;

    // The index of this patch (relative to all other patches)
    uint32_t patch_id;

    // neighbor patches stash
    PatchStash patch_stash;

    // Hash table storing the mapping from local indices of ribbon (not-owned)
    // mesh elements to their owner patch and their local indices in their owner
    // patch
    LPHashTable lp_v, lp_e, lp_f;


    /**
     * @brief count number of owned active vertices from the bitmasks
     */
    __host__ __inline__ uint16_t get_num_owned_vertices() const
    {
        return count_num_owned(owned_mask_v, active_mask_v, num_vertices[0]);
    }

    /**
     * @brief count the number of owned active edges from the bitmasks
     * @return
     */
    __host__ __inline__ uint16_t get_num_owned_edges() const
    {
        return count_num_owned(owned_mask_e, active_mask_e, num_edges[0]);
    }

    /**
     * @brief count the number of owned active faces from the bitmaks
     */
    __host__ __inline__ uint16_t get_num_owned_faces() const
    {
        return count_num_owned(owned_mask_f, active_mask_f, num_faces[0]);
    }

   private:
    __host__ __inline__ uint16_t count_num_owned(const uint32_t* owned_bitmask,
                                                 const uint32_t* active_bitmask,
                                                 const uint16_t  size) const
    {
        uint16_t ret = 0;
        for (uint16_t i = 0; i < size; ++i) {
            if (detail::is_owned(i, owned_bitmask) &&
                !detail::is_deleted(i, active_bitmask)) {
                ret++;
            }
        }
        return ret;
    }
};
}  // namespace rxmesh