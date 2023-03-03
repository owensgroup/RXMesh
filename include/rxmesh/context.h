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
          m_vertex_prefix(nullptr),
          m_edge_prefix(nullptr),
          m_face_prefix(nullptr),
          m_capacity_factor(0.0f),
          m_patches_info(nullptr)
    {
    }

    Context(const Context&) = default;

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

    /**
     * @brief get the owner handle of a given vertex handle
     * @param vh the vertex handle
     * @param table pointer to LPPair hashtable in case it's stored in shared
     * memory
     */
    __device__ __inline__ VertexHandle get_owner_vertex_handle(
        const VertexHandle vh,
        const LPPair*      table = nullptr,
        const bool         check = true) const
    {
        return get_owner_handle(vh, m_patches_info, table, check);
    }

    /**
     * @brief get the owner handle of a given edge handle
     * @param eh the edge handle
     * @param table pointer to LPPair hashtable in case it's stored in shared
     * memory
     */
    __device__ __inline__ EdgeHandle get_owner_edge_handle(
        const EdgeHandle eh,
        const LPPair*    table = nullptr,
        const bool       check = true) const
    {
        return get_owner_handle(eh, m_patches_info, table, check);
    }

    /**
     * @brief get the owner handle of a given face handle
     * @param fh the face handle
     * @param table pointer to LPPair hashtable in case it's stored in shared
     * memory
     */
    __device__ __inline__ FaceHandle get_owner_face_handle(
        const FaceHandle fh,
        const LPPair*    table = nullptr,
        const bool       check = true) const
    {
        return get_owner_handle(fh, m_patches_info, table, check);
    }

    /**
     * @brief get the owner handle of a given mesh element handle
     * @param handle the mesh element handle
     * @param table pointer to LPPair hashtable in case it's stored in shared
     * memory
     */
    template <typename HandleT>
    static __device__ __host__ __inline__ HandleT get_owner_handle(
        const HandleT    handle,
        const PatchInfo* patches_info,
        const LPPair*    table = nullptr,
        const bool       check = true)
    {
        using LocalT   = typename HandleT::LocalT;
        uint32_t owner = handle.patch_id();
        uint16_t lid   = handle.local_id();


        if (check) {
            assert(!patches_info[owner].is_deleted(LocalT(lid)));
        }


        if (patches_info[owner].is_owned(LocalT(lid))) {
            return handle;
        } else {

            LPPair lp = patches_info[owner].get_lp<HandleT>().find(lid, table);

            assert(!lp.is_sentinel());
            owner = patches_info[owner].patch_stash.get_patch(lp);

            assert(!patches_info[owner].is_deleted(
                LocalT(lp.local_id_in_owner_patch())));

            while (!patches_info[owner].is_owned(
                LocalT(lp.local_id_in_owner_patch()))) {

                lp = patches_info[owner].get_lp<HandleT>().find(
                    lp.local_id_in_owner_patch());

                assert(!lp.is_sentinel());
                owner = patches_info[owner].patch_stash.get_patch(lp);

                assert(!patches_info[owner].is_deleted(
                    LocalT(lp.local_id_in_owner_patch())));
            }

            return HandleT(owner, lp.local_id_in_owner_patch());
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
              const float    capacity_factor,
              uint32_t*      vertex_prefix,
              uint32_t*      edge_prefix,
              uint32_t*      face_prefix,
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

        m_vertex_prefix = vertex_prefix;
        m_edge_prefix   = edge_prefix;
        m_face_prefix   = face_prefix;

        m_patches_info = d_patches;

        m_patch_scheduler = scheduler;
    }

    void release()
    {
        CUDA_ERROR(cudaFree(m_num_vertices));
    }


    uint32_t *     m_num_edges, *m_num_faces, *m_num_vertices, *m_num_patches;
    uint32_t *     m_max_num_vertices, *m_max_num_edges, *m_max_num_faces;
    uint32_t *     m_vertex_prefix, *m_edge_prefix, *m_face_prefix;
    PatchInfo*     m_patches_info;
    float          m_capacity_factor;
    PatchScheduler m_patch_scheduler;
};
}  // namespace rxmesh