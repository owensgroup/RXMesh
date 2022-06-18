#pragma once
#include <stdint.h>

#include "rxmesh/local.h"
#include "rxmesh/lp_hashtable.cuh"
#include "rxmesh/patch_stash.cuh"
#include "rxmesh/util/macros.h"

namespace rxmesh {

/**
 * @brief PatchInfo stores the information needed for query operations in a
 * patch
 */
struct ALIGN(16) PatchInfo
{
    // The topology information: edge incident vertices and face incident edges
    LocalVertexT* ev;
    LocalEdgeT*   fe;


    // Non-owned mesh elements patch ID
    uint32_t* not_owned_patch_v;
    uint32_t* not_owned_patch_e;
    uint32_t* not_owned_patch_f;


    // Non-owned mesh elements local ID
    LocalVertexT* not_owned_id_v;
    LocalEdgeT*   not_owned_id_e;
    LocalFaceT*   not_owned_id_f;


    // bitmask for existing "owned" mesh elements
    uint32_t *active_mask_v, *active_mask_e, *active_mask_f;

    // Number of mesh elements in the patch
    uint16_t num_vertices, num_edges, num_faces;

    // Capacity of v/e/f. This controls the allocations of ev, fe,
    // active_mask_v/e/f
    uint16_t vertices_capacity, edges_capacity, faces_capacity;

    // Number of mesh elements owned by this patch
    uint16_t num_owned_vertices, num_owned_edges, num_owned_faces;

    // The index of this patch (relative to all other patches)
    uint32_t patch_id;
};


/**
 * @brief PatchInfo stores the information needed for query operations in a
 * patch
 */
struct ALIGN(16) PatchInfoV2
{
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
    uint16_t num_vertices, num_edges, num_faces;

    // Capacity of v/e/f. This controls the allocations of ev, fe,
    // active_mask_v/e/f, owned_mask_v/e/f
    uint16_t vertices_capacity, edges_capacity, faces_capacity;

    // The index of this patch (relative to all other patches)
    uint32_t patch_id;

    // neighbor patches stash
    PatchStash patch_stash;

    // Hash table storing the mapping from local indices of ribbon (not-owned)
    // mesh elements to their owner patch and their local indices in their owner
    // patch
    LPHashTable lp_v, lp_e, lp_f;
};
}  // namespace rxmesh