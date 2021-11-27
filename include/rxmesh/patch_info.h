#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string>
#include <utility>
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


    __host__ __device__ std::pair<uint32_t, uint16_t>
    get_patch_and_local_id(const VertexHandle vh) const
    {
        assert(vh.is_valid());
        assert(patch_id == vh.m_patch_id);

        uint32_t p = patch_id;
        uint16_t l = vh.m_v.id;

        if (l >= num_owned_vertices) {
            p = not_owned_patch_v[l - num_owned_vertices];
            l = not_owned_id_v[l - num_owned_vertices].id;            
        }

        return std::make_pair(p, l);
    }
};

}  // namespace rxmesh