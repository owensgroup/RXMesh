#pragma once

#include <stdint.h>
#include "rxmesh/patch_info.h"
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
    Context()
        : m_num_edges(nullptr),
          m_num_faces(nullptr),
          m_num_vertices(nullptr),
          m_num_patches(nullptr),
          m_patches_info(nullptr)
    {
    }

    Context(const Context&) = default;

    /**
     * @brief Total number of edges in mesh
     */
    __device__ __forceinline__ uint32_t get_num_edges() const
    {
        return *m_num_edges;
    }

    /**
     * @brief Total number of faces in mesh
     */
    __device__ __forceinline__ uint32_t get_num_faces() const
    {
        return *m_num_faces;
    }

    /**
     * @brief Total number of vertices in mesh
     */
    __device__ __forceinline__ uint32_t get_num_vertices() const
    {
        return *m_num_vertices;
    }

    /**
     * @brief Total number of patches in mesh
     */
    __device__ __forceinline__ uint32_t get_num_patches() const
    {
        return *m_num_patches;
    }

    /**
     * @brief A pointer to device PatchInfo used to store various information
     * about the patches
     */
    __device__ __forceinline__ PatchInfo* get_patches_info() const
    {
        return m_patches_info;
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

   private:
    /**
     * @brief initialize various members
     * @param num_edges total number of edges in the mesh
     * @param num_faces total number of faces in the mesh
     * @param num_vertices total number of vertices in the mesh
     * @param num_patches number of patches
     * @param patches pointer to PatchInfo that contains different info about
     * the patches
     */
    void init(const uint32_t num_edges,
              const uint32_t num_faces,
              const uint32_t num_vertices,
              const uint32_t num_patches,
              PatchInfo*     patches)
    {
        CUDA_ERROR(cudaMalloc((void**)&m_num_vertices, sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc((void**)&m_num_edges, sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc((void**)&m_num_faces, sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc((void**)&m_num_patches, sizeof(uint32_t)));

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
        m_patches_info = patches;
    }

    void release()
    {
        CUDA_ERROR(cudaFree(m_num_edges));
        CUDA_ERROR(cudaFree(m_num_faces));
        CUDA_ERROR(cudaFree(m_num_vertices));
        CUDA_ERROR(cudaFree(m_num_patches));
    }


    uint32_t * m_num_edges, *m_num_faces, *m_num_vertices, *m_num_patches;
    PatchInfo* m_patches_info;
};
}  // namespace rxmesh