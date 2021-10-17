#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string>
#include "rxmesh/rxmesh_types.h"
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

    // Number of mesh elements in the patch
    uint16_t num_vertices, num_edges, num_faces;
    uint16_t num_owned_vertices, num_owned_edges, num_owned_faces;

    // The index of this patch (relative to all other patches)
    uint32_t patch_id;

    /**
     * @brief Return a unique handle to a local vertex
     * @param local_id the local within-patch vertex id
     * @return
     */
    uint64_t __device__ __host__ __inline__ unique_id(LocalVertexT local_id)
    {
        return unique_id(local_id.id, patch_id);
    }


    /**
     * @brief Return a unique handle to a local edge
     * @param local_id the local within-patch edge id
     * @return
     */
    uint64_t __device__ __host__ __inline__ unique_id(LocalEdgeT local_id)
    {
        return unique_id(local_id.id, patch_id);
    }


    /**
     * @brief Return a unique handle to a local face
     * @param local_id the local within-patch face id
     * @return
     */
    uint64_t __device__ __host__ __inline__ unique_id(LocalFaceT local_id)
    {
        return unique_id(local_id.id, patch_id);
    }


    /**
     * @brief Return unique index of the local mesh element composed by the
     * patch id and the local index
     *
     * @param local_id the local within-patch mesh element id
     * @param patch_id the patch owning the mesh element
     * @return
     */
    static uint64_t __device__ __host__ __inline__ unique_id(uint16_t local_id,
                                                             uint32_t patch_id)
    {
        uint64_t ret = local_id;
        ret |= ret << 32;
        ret |= patch_id;
        return ret;
    }
};

}  // namespace rxmesh