#pragma once

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/macros.h"

#include <cub/device/device_scan.cuh>
namespace rxmesh {

namespace detail {
void init(RXMeshStatic& rx,
          uint32_t*&    d_vertex,
          uint32_t*&    d_edge,
          uint32_t*&    d_face)
{

    uint32_t num_patches = rx.get_num_patches();

    CUDA_ERROR(
        cudaMalloc((void**)&d_vertex, (num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&d_edge, (num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&d_face, (num_patches + 1) * sizeof(uint32_t)));

    CUDA_ERROR(cudaMemset(d_vertex, 0, (num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_edge, 0, (num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_face, 0, (num_patches + 1) * sizeof(uint32_t)));

    Context context = rx.get_context();

    // We kind "hack" for_each_vertex to store the owned vertex/edge/face
    // count in d_vertex/edge/face. Since in for_each_vertex we lunch
    // one block per patch, then blockIdx.x correspond to the patch id. We
    // then use only one thread from the block to write the owned
    // vertex/edge/face count
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle vh) {
        if (threadIdx.x == 0) {
            uint32_t patch_id = blockIdx.x;
            d_vertex[patch_id] =
                context.get_patches_info()[patch_id].num_owned_vertices;

            d_edge[patch_id] =
                context.get_patches_info()[patch_id].num_owned_edges;

            d_face[patch_id] =
                context.get_patches_info()[patch_id].num_owned_faces;
        }
    });

    CUDA_ERROR(cudaDeviceSynchronize());

    // Exclusive perfix sum computation. Increase the size by 1 so that we dont
    // need to stick in the total number of owned vertices/edges/faces at the
    // end of manually
    void*  d_cub_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                  temp_storage_bytes,
                                  d_vertex,
                                  d_vertex,
                                  num_patches + 1);
    CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage, temp_storage_bytes));

    cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                  temp_storage_bytes,
                                  d_vertex,
                                  d_vertex,
                                  num_patches + 1);
    CUDA_ERROR(cudaMemset(d_cub_temp_storage, 0, temp_storage_bytes));

    cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                  temp_storage_bytes,
                                  d_edge,
                                  d_edge,
                                  num_patches + 1);
    CUDA_ERROR(cudaMemset(d_cub_temp_storage, 0, temp_storage_bytes));


    cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                  temp_storage_bytes,
                                  d_face,
                                  d_face,
                                  num_patches + 1);

    CUDA_ERROR(cudaFree(d_cub_temp_storage));
}
}  // namespace detail

/**
 * @brief Calculate and store the prefix sum of (the owned) vertex/edge/face
 * count per patch. All calculations and storage is done on the GPU
 */
struct PatchPtr
{
    PatchPtr(RXMeshStatic& rx)
        : m_d_vertex(nullptr), m_d_edge(nullptr), m_d_face(nullptr)
    {
        detail::init(rx, m_d_vertex, m_d_edge, m_d_face);
    }


    const uint32_t* get_pointer(ELEMENT ele) const
    {
        switch (ele) {
            case ELEMENT::VERTEX:
                return m_d_vertex;
            case ELEMENT::EDGE:
                return m_d_edge;
            case ELEMENT::FACE:
                return m_d_face;
        }
    }


    void free()
    {
        CUDA_ERROR(cudaFree(m_d_vertex));
        CUDA_ERROR(cudaFree(m_d_edge));
        CUDA_ERROR(cudaFree(m_d_face));
    }

   private:
    uint32_t *m_d_vertex, *m_d_edge, *m_d_face;
};
}  // namespace rxmesh
