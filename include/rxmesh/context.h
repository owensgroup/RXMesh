#pragma once

#include <stdint.h>
#include "rxmesh/patch_info.h"
#include "rxmesh/patch_scheduler.cuh"
#include "rxmesh/util/macros.h"

namespace rxmesh {

/**
 * @brief context for the mesh parameters and pointers. Everything is allocated
 * on and managed by RXMesh. This class is meant to be a vehicle to copy various
 * parameters to the device kernels.
 */
class Context
{
   public:
    friend class RXMesh;
    friend class RXMeshDynamic;

    /**
     * @brief Default constructor
     */
    __host__ __device__ Context()
        : m_num_edges(nullptr),
          m_num_faces(nullptr),
          m_num_vertices(nullptr),
          m_num_patches(nullptr),
          m_max_num_vertices(nullptr), 
          m_max_num_edges(nullptr), 
          m_max_num_faces(nullptr),          
          m_d_vertex_prefix(nullptr),
          m_d_edge_prefix(nullptr),
          m_d_face_prefix(nullptr),
          m_h_vertex_prefix(nullptr),
          m_h_edge_prefix(nullptr),
          m_h_face_prefix(nullptr),                    
          m_max_lp_capacity_v(0),
          m_max_lp_capacity_e(0),
          m_max_lp_capacity_f(0),
          m_patches_info(nullptr),
          m_capacity_factor(0.0f),
          m_max_num_patches(0)
    {
    }

    __device__          Context(const Context&)   = default;
    __device__          Context(Context&&)        = default;
    __device__ Context& operator=(const Context&) = default;
    __device__ Context& operator=(Context&&)      = default;

    /**
     * @brief Total number of edges in mesh
     */
    __device__ __forceinline__ uint32_t* get_num_edges()
    {
        return m_num_edges;
    }

    /**
     * @brief Total number of faces in mesh
     */
    __device__ __forceinline__ uint32_t* get_num_faces()
    {
        return m_num_faces;
    }

    /**
     * @brief Total number of vertices in mesh
     */
    __device__ __forceinline__ uint32_t* get_num_vertices()
    {
        return m_num_vertices;
    }

    /**
     * @brief Total number of patches in mesh
     */
    __device__ __forceinline__ uint32_t get_num_patches() const
    {
        return m_num_patches[0];
    }

    /**
     * @brief Unpack an edge to its edge ID and direction
     * @param edge_dir The input packed edge as stored in PatchInfo and
     * internally in RXMesh
     * @param edge The unpacked edge ID
     * @param dir The unpacked edge direction
     */
    static __device__ __host__ __forceinline__ void
    unpack_edge_dir(const uint16_t edge_dir, uint16_t& edge, flag_t& dir)
    {
        dir  = (edge_dir & 1) != 0;
        edge = edge_dir >> 1;
    }

    __device__ __host__ __inline__ const uint32_t* vertex_prefix() const
    {
#ifdef __CUDA_ARCH__
        return m_d_vertex_prefix;
#else
        return m_h_vertex_prefix;
#endif
    }

    __device__ __host__ __inline__ const uint32_t* edge_prefix() const
    {
#ifdef __CUDA_ARCH__
        return m_d_edge_prefix;
#else
        return m_h_edge_prefix;
#endif
    }

    __device__ __host__ __inline__ const uint32_t* face_prefix() const
    {
#ifdef __CUDA_ARCH__
        return m_d_face_prefix;
#else
        return m_h_face_prefix;
#endif
    }

    /**
     * @brief get the owner handle of a given mesh element handle
     * @param handle the mesh element handle
     * @param table pointer to LPPair hashtable in case it's stored in shared
     * memory
     */
    template <typename HandleT>
    __device__ __host__ __inline__ HandleT get_owner_handle(
        const HandleT    handle,
        const PatchInfo* patches_info = nullptr,
        const LPPair*    table        = nullptr,
        const LPPair*    stash        = nullptr,
        const bool       check0       = true,
        const bool       check1       = true) const
    {
        using LocalT   = typename HandleT::LocalT;
        uint32_t owner = handle.patch_id();
        uint16_t lid   = handle.local_id();

        const PatchInfo* pi =
            (patches_info == nullptr) ? m_patches_info : patches_info;

        if (check0) {
            assert(!pi[owner].is_deleted(LocalT(lid)));
        }


        if (pi[owner].is_owned(LocalT(lid))) {
            return handle;
        } else {

            LPPair lp = pi[owner].get_lp<HandleT>().find(lid, table, stash);

            assert(!lp.is_sentinel());
            owner = pi[owner].patch_stash.get_patch(lp);

            if (check1) {
                assert(!pi[owner].is_deleted(
                    LocalT(lp.local_id_in_owner_patch())));
            }

            while (!pi[owner].is_owned(LocalT(lp.local_id_in_owner_patch()))) {

                lp = pi[owner].get_lp<HandleT>().find(
                    lp.local_id_in_owner_patch(), nullptr, nullptr);

                assert(!lp.is_sentinel());
                owner = pi[owner].patch_stash.get_patch(lp);

                if (check1) {
                    assert(!pi[owner].is_deleted(
                        LocalT(lp.local_id_in_owner_patch())));
                }
            }

            return HandleT(owner, lp.local_id_in_owner_patch());
        }
    }

    /**
     * @brief compute a linear compact index for a give vertex/edge/face handle.
     * This is only valid for static mesh processing i.e., RXMeshStatic.
     * @tparam HandleT the type of the input handle
     * @param input handle
     */
    template <typename HandleT>
    __device__ __host__ __inline__ uint32_t linear_id(HandleT input) const
    {        
        assert(input.is_valid());

        assert(input.patch_id() < m_num_patches[0]);


        const HandleT owner_handle = get_owner_handle(input);

        uint32_t p_id = owner_handle.patch_id();
        uint16_t ret  = owner_handle.local_id();

        assert(m_patches_info[p_id].is_owned(HandleT::LocalT(ret)));

        // TODO we don't need to do count the number of owned elements if the
        // mesh is static, i.e., we have not modified the topology of the mesh
        // yet since we number the owned elements first and there is no deleted
        // elements yet
        ret = this->m_patches_info[p_id].count_num_owned(
            m_patches_info[p_id].get_owned_mask<HandleT>(),
            m_patches_info[p_id].get_active_mask<HandleT>(),
            ret);

        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            return ret + m_d_vertex_prefix[p_id];
        }
        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            return ret + m_d_edge_prefix[p_id];
        }
        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            return ret + m_d_face_prefix[p_id];
        }
    }


    /**
     * @brief initialize various members
     * @param num_vertices total number of vertices in the mesh
     * @param num_edges total number of edges in the mesh
     * @param num_faces total number of faces in the mesh
     * @param max_num_vertices max number of vertices in a patch
     * @param max_num_edges max number of edges in a patch
     * @param max_num_faces max number of faces in a patch
     * @param num_patches number of patches
     * @param patches pointer to PatchInfo that contains different info about
     * the patches
     */
    void init(const uint32_t num_vertices,
              const uint32_t num_edges,
              const uint32_t num_faces,
              const uint32_t max_num_vertices,
              const uint32_t max_num_edges,
              const uint32_t max_num_faces,
              const uint32_t num_patches,
              const uint32_t max_num_patches,
              const float    capacity_factor,
              uint32_t*      d_vertex_prefix,
              uint32_t*      d_edge_prefix,
              uint32_t*      d_face_prefix,
              uint32_t*      h_vertex_prefix,
              uint32_t*      h_edge_prefix,
              uint32_t*      h_face_prefix,
              uint16_t       max_lp_capacity_v,
              uint16_t       max_lp_capacity_e,
              uint16_t       max_lp_capacity_f,
              PatchInfo*     d_patches,
              PatchScheduler scheduler)
    {
        uint32_t* buffer = nullptr;
        CUDA_ERROR(cudaMalloc((void**)&buffer, 7 * sizeof(uint32_t)));
        m_num_vertices     = buffer + 0;
        m_num_edges        = buffer + 1;
        m_num_faces        = buffer + 2;
        m_num_patches      = buffer + 3;
        m_max_num_vertices = buffer + 4;
        m_max_num_edges    = buffer + 5;
        m_max_num_faces    = buffer + 6;
        m_capacity_factor  = capacity_factor;
        m_max_num_patches  = max_num_patches;

        CUDA_ERROR(cudaMemcpy(m_num_vertices,
                              &num_vertices,
                              sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(
            m_num_edges, &num_edges, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(
            m_num_faces, &num_faces, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(m_num_patches,
                              &num_patches,
                              sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        CUDA_ERROR(cudaMemcpy(m_max_num_vertices,
                              &max_num_vertices,
                              sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(m_max_num_edges,
                              &max_num_edges,
                              sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(m_max_num_faces,
                              &max_num_faces,
                              sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        m_h_vertex_prefix = h_vertex_prefix;
        m_h_edge_prefix   = h_edge_prefix;
        m_h_face_prefix   = h_face_prefix;

        m_d_vertex_prefix = d_vertex_prefix;
        m_d_edge_prefix   = d_edge_prefix;
        m_d_face_prefix   = d_face_prefix;

        m_max_lp_capacity_v = max_lp_capacity_v;
        m_max_lp_capacity_e = max_lp_capacity_e;
        m_max_lp_capacity_f = max_lp_capacity_f;

        m_patches_info = d_patches;

        m_patch_scheduler = scheduler;
    }

    void release()
    {
        CUDA_ERROR(cudaFree(m_num_vertices));
    }


    uint32_t *m_num_edges, *m_num_faces, *m_num_vertices, *m_num_patches;
    // per-patch max v/e/f
    uint32_t *m_max_num_vertices, *m_max_num_edges, *m_max_num_faces;
    uint32_t *m_d_vertex_prefix, *m_d_edge_prefix, *m_d_face_prefix,
        *m_h_vertex_prefix, *m_h_edge_prefix, *m_h_face_prefix;
    uint16_t   m_max_lp_capacity_v, m_max_lp_capacity_e, m_max_lp_capacity_f;
    PatchInfo* m_patches_info;
    float      m_capacity_factor;
    uint32_t   m_max_num_patches;
    PatchScheduler m_patch_scheduler;
};
}  // namespace rxmesh