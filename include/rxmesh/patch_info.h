#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string>
#include <utility>
#include "rxmesh/handle.h"
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


    __host__ __device__ __forceinline__ std::pair<uint32_t, uint16_t>
    get_patch_and_local_id(const VertexHandle& vh) const
    {
        auto ret = vh.unpack();

        if (!vh.is_valid()) {
            return ret;
        }
        assert(patch_id == ret.first);

        if (ret.second >= num_owned_vertices) {
            ret.first  = not_owned_patch_v[ret.second - num_owned_vertices];
            ret.second = not_owned_id_v[ret.second - num_owned_vertices].id;
        }

        return ret;
    }


    __host__ __device__ __forceinline__ std::pair<uint32_t, uint16_t>
    get_patch_and_local_id(const EdgeHandle& eh) const
    {

        auto ret = eh.unpack();

        if (!eh.is_valid()) {
            return ret;
        }

        assert(patch_id == ret.first);

        if (ret.second >= num_owned_edges) {
            ret.first  = not_owned_patch_e[ret.second - num_owned_edges];
            ret.second = not_owned_id_e[ret.second - num_owned_edges].id;
        }

        return ret;
    }

    __host__ __device__ __forceinline__ std::pair<uint32_t, uint16_t>
    get_patch_and_local_id(const FaceHandle& fh) const
    {
        auto ret = fh.unpack();

        if (!fh.is_valid()) {
            return ret;
        }

        assert(patch_id == ret.first);

        if (ret.second >= num_owned_faces) {
            ret.first  = not_owned_patch_f[ret.second - num_owned_faces];
            ret.second = not_owned_id_f[ret.second - num_owned_faces].id;
        }

        return ret;
    }
};

}  // namespace rxmesh