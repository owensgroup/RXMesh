#pragma once
#include <stdint.h>

#include "rxmesh/handle.h"
#include "rxmesh/local.h"
#include "rxmesh/lp_hashtable.cuh"
#include "rxmesh/patch_stash.cuh"
#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/macros.h"

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
          num_vertices(0),
          num_edges(0),
          num_faces(0),
          vertices_capacity(0),
          edges_capacity(0),
          faces_capacity(0),
          patch_id(INVALID32){};

    __device__ __host__ PatchInfo(const PatchInfo& other) = default;
    __device__ __host__ PatchInfo(PatchInfo&&)            = default;
    __device__ __host__ PatchInfo& operator=(const PatchInfo&) = default;
    __device__ __host__ PatchInfo& operator=(PatchInfo&&) = default;
    __device__                     __host__ ~PatchInfo()  = default;

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

    // The index of this patch
    uint32_t patch_id;

    // neighbor patches stash
    PatchStash patch_stash;

    // Hash table storing the mapping from local indices of ribbon (not-owned)
    // mesh elements to their owner patch and their local indices in their owner
    // patch
    LPHashTable lp_v, lp_e, lp_f;

    uint32_t* mutex;

    __device__ bool try_lock(uint32_t p)
    {
#ifdef __CUDA_ARCH__
        uint32_t old = ::atomicCAS(mutex, INVALID32, p);
        if (old == INVALID32) {
            return true;
        } else {
            return false;
        }
#endif
    }


    __device__ void unlock()
    {
#ifdef __CUDA_ARCH__
        atomicExch(mutex, INVALID32);
#endif
    }


    /**
     * @brief return pointer to the number of elements corresponding  to the
     * handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ __inline__ const uint16_t* get_num_elements() const
    {
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            return num_vertices;
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            return num_edges;
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            return num_faces;
        }
    }


    /**
     * @brief return the capacity corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ __inline__ const uint16_t* get_capacity() const
    {
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            return vertices_capacity;
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            return edges_capacity;
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            return faces_capacity;
        }
    }


    /**
     * @brief return the active mask corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ __inline__ const uint32_t* get_active_mask() const
    {
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            return active_mask_v;
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            return active_mask_e;
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            return active_mask_f;
        }
    }

    /**
     * @brief return the active mask corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ __inline__ uint32_t* get_active_mask()
    {
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            return active_mask_v;
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            return active_mask_e;
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            return active_mask_f;
        }
    }


    /**
     * @brief return the owned mask corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ __inline__ const uint32_t* get_owned_mask() const
    {
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            return owned_mask_v;
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            return owned_mask_e;
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            return owned_mask_f;
        }
    }


    /**
     * @brief return LP hashtable corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ __inline__ const LPHashTable& get_lp() const
    {
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            return lp_v;
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            return lp_e;
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            return lp_f;
        }
    }


    /**
     * @brief return LP hashtable corresponding to the handle type
     * @tparam HandleT
     */
    template <typename HandleT>
    __device__ __host__ __inline__ LPHashTable& get_lp()
    {
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            return lp_v;
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            return lp_e;
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            return lp_f;
        }
    }


    /**
     * @brief check if a vertex within this patch is owned by it
     */
    __device__ __host__ __inline__ const bool is_owned(LocalVertexT vh) const
    {
        return detail::is_owned(vh.id, get_owned_mask<VertexHandle>());
    }

    /**
     * @brief check if an edge within this patch is owned by it
     */
    __device__ __host__ __inline__ const bool is_owned(LocalEdgeT eh) const
    {
        return detail::is_owned(eh.id, get_owned_mask<EdgeHandle>());
    }

    /**
     * @brief check if a face within this patch is owned by it
     */
    __device__ __host__ __inline__ const bool is_owned(LocalFaceT fh) const
    {
        return detail::is_owned(fh.id, get_owned_mask<FaceHandle>());
    }

    /**
     * @brief check if a vertex within this patch is deleted
     */
    __device__ __host__ __inline__ const bool is_deleted(LocalVertexT vh) const
    {
        return detail::is_deleted(vh.id, get_active_mask<VertexHandle>());
    }

    /**
     * @brief check if an edge within this patch is deleted
     */
    __device__ __host__ __inline__ const bool is_deleted(LocalEdgeT eh) const
    {
        return detail::is_deleted(eh.id, get_active_mask<EdgeHandle>());
    }

    /**
     * @brief check if a face within this patch is deleted
     */
    __device__ __host__ __inline__ const bool is_deleted(LocalFaceT fh) const
    {
        return detail::is_deleted(fh.id, get_active_mask<FaceHandle>());
    }


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