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
    LocalVertexT* m_ev;
    LocalEdgeT*   m_fe;


    // Non-owned mesh elements patch ID
    uint32_t* m_not_owned_patch_v;
    uint32_t* m_not_owned_patch_e;
    uint32_t* m_not_owned_patch_f;


    // Non-owned mesh elements local ID
    LocalVertexT* m_not_owned_id_v;
    LocalEdgeT*   m_not_owned_id_e;
    LocalFaceT*   m_not_owned_id_f;

    // Number of mesh elements in the patch
    uint16_t m_num_vertices, m_num_edges, m_num_faces;
    uint16_t m_num_owned_vertices, m_num_owned_edges, m_num_owned_faces;

    // The index of this patch (relative to all other patches)
    uint32_t m_patch_id;

    /**
     * @brief Return a unique handle to a local vertex
     * @param local_id the local within-patch vertex id
     * @return
     */
    uint64_t __device__ __host__ __inline__ unique_id(LocalVertexT local_id)
    {
        return unique_id(local_id.id, m_patch_id);
    }


    /**
     * @brief Return a unique handle to a local edge
     * @param local_id the local within-patch edge id
     * @return
     */
    uint64_t __device__ __host__ __inline__ unique_id(LocalEdgeT local_id)
    {
        return unique_id(local_id.id, m_patch_id);
    }


    /**
     * @brief Return a unique handle to a local face
     * @param local_id the local within-patch face id
     * @return
     */
    uint64_t __device__ __host__ __inline__ unique_id(LocalFaceT local_id)
    {
        return unique_id(local_id.id, m_patch_id);
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