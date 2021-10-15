#pragma once

#include <stdint.h>
#include "rxmesh/rxmesh_types.h"
#include "rxmesh/util/macros.h"

namespace rxmesh {

// context for the mesh parameters and pointers. everything is allocated
// on rxmesh. this class is meant to be a vehicle to copy various parameters
// to the device kernels.
// TODO make sure that __align__(16) is the right one
class __align__(16) RXMeshContext
{

   public:
    RXMeshContext()
        : m_num_edges(0),
          m_num_faces(0),
          m_num_vertices(0),
          m_max_valence(0),
          m_max_edge_incident_faces(0),
          m_max_face_adjacent_faces(0),
          m_num_patches(0),
          m_d_face_patch(nullptr),
          m_d_edge_patch(nullptr),
          m_d_vertex_patch(nullptr),
          m_d_patches_ltog_v(nullptr),
          m_d_patches_ltog_e(nullptr),
          m_d_patches_ltog_f(nullptr),
          m_d_ad_size_ltog_v(nullptr),
          m_d_ad_size_ltog_e(nullptr),
          m_d_ad_size_ltog_f(nullptr),
          m_d_patches_edges(nullptr),
          m_d_patches_faces(nullptr),
          m_d_ad_size(nullptr),
          m_d_num_owned_f(nullptr),
          m_d_num_owned_e(nullptr),
          m_d_num_owned_v(nullptr),
          m_patches_info(nullptr)
    {
    }

    void init(const uint32_t num_edges,
              const uint32_t num_faces,
              const uint32_t num_vertices,
              const uint32_t max_valence,
              const uint32_t max_edge_incident_faces,
              const uint32_t max_face_adjacent_faces,
              const uint32_t num_patches,
              uint32_t*      d_face_patch,
              uint32_t*      d_edge_patch,
              uint32_t*      d_vertex_patch,
              uint32_t*      d_patches_ltog_v,
              uint32_t*      d_patches_ltog_e,
              uint32_t*      d_patches_ltog_f,
              uint2*         d_ad_size_ltog_v,
              uint2*         d_ad_size_ltog_e,
              uint2*         d_ad_size_ltog_f,
              uint16_t*      d_patches_edges,
              uint16_t*      d_patches_faces,
              uint4*         d_ad_size,
              uint16_t*      d_num_owned_f,
              uint16_t*      d_num_owned_e,
              uint16_t*      d_num_owned_v,
              PatchInfo*     patches)
    {

        m_num_edges               = num_edges;
        m_num_faces               = num_faces;
        m_num_vertices            = num_vertices;
        m_max_valence             = max_valence;
        m_max_edge_incident_faces = max_edge_incident_faces;
        m_max_face_adjacent_faces = max_face_adjacent_faces;
        m_num_patches             = num_patches;
        m_d_face_patch            = d_face_patch;
        m_d_edge_patch            = d_edge_patch;
        m_d_vertex_patch          = d_vertex_patch;
        m_d_patches_ltog_v        = d_patches_ltog_v;
        m_d_patches_ltog_e        = d_patches_ltog_e;
        m_d_patches_ltog_f        = d_patches_ltog_f;
        m_d_ad_size_ltog_v        = d_ad_size_ltog_v;
        m_d_ad_size_ltog_e        = d_ad_size_ltog_e;
        m_d_ad_size_ltog_f        = d_ad_size_ltog_f;
        m_d_patches_edges         = d_patches_edges;
        m_d_patches_faces         = d_patches_faces;
        m_d_ad_size               = d_ad_size;
        m_d_num_owned_f           = d_num_owned_f;
        m_d_num_owned_e           = d_num_owned_e;
        m_d_num_owned_v           = d_num_owned_v;
        m_patches_info                 = patches;
    }


    template <typename dataT>
    __device__ void print_data(const dataT*   arr,
                               const uint32_t start_id,
                               const uint32_t len,
                               int            shift = 0) const
    {
        printf(" start_id = %u, len = %u\n", start_id, len);

        uint32_t end = len + start_id;
        for (uint32_t i = start_id; i < end; ++i) {
            printf(" [%u] ", arr[i] >> shift);
            if (i % 20 == 0 && i != start_id) {
                printf("\n");
            }
        }
        printf("\n\n");
    }

    __device__ void print_patch(uint32_t p_id) const
    {
        // print all relevant data of a single patch

        // if (threadIdx.x == 0){
        printf("\n ********* p_id = %u *********\n", p_id);
        printf(" global_num_vertices=%u \n", m_num_vertices);
        printf(" global_num_edges=%u \n", m_num_edges);
        printf(" global_num_faces=%u \n", m_num_faces);
        printf(" global_num_patches=%u \n", m_num_patches);

        printf(" patch #vertices = %u, start_id= %u \n",
               m_d_ad_size_ltog_v[p_id].y,
               m_d_ad_size_ltog_v[p_id].x);
        printf(" patch #edges = %u, start_id= %u\n",
               m_d_ad_size_ltog_e[p_id].y,
               m_d_ad_size_ltog_e[p_id].x);
        printf(" patch #faces = %u, start_id= %u\n",
               m_d_ad_size_ltog_f[p_id].y,
               m_d_ad_size_ltog_f[p_id].x);

        printf("\n ** d_ltog_v **\n");
        print_data(m_d_patches_ltog_v,
                   uint32_t(m_d_ad_size_ltog_v[p_id].x),
                   uint32_t(m_d_ad_size_ltog_v[p_id].y),
                   1);

        printf("\n ** d_ltog_e **\n");
        print_data(m_d_patches_ltog_e,
                   uint32_t(m_d_ad_size_ltog_e[p_id].x),
                   uint32_t(m_d_ad_size_ltog_e[p_id].y),
                   1);

        printf("\n ** d_ltog_f **\n");
        print_data(m_d_patches_ltog_f,
                   uint32_t(m_d_ad_size_ltog_f[p_id].x),
                   uint32_t(m_d_ad_size_ltog_f[p_id].y),
                   1);


        printf("\n ** d_edges **\n");
        print_data(m_d_patches_edges,
                   uint32_t(m_d_ad_size[p_id].x),
                   uint32_t(m_d_ad_size[p_id].y));

        printf("\n ** d_faces **\n");
        print_data(m_d_patches_faces,
                   uint32_t(m_d_ad_size[p_id].z),
                   uint32_t(m_d_ad_size[p_id].w),
                   1);
        //}
    }

    __device__ __forceinline__ uint32_t get_num_edges() const
    {
        return m_num_edges;
    }
    __device__ __forceinline__ uint32_t get_num_faces() const
    {
        return m_num_faces;
    }
    __device__ __forceinline__ uint32_t get_num_vertices() const
    {
        return m_num_vertices;
    }
    __device__ __forceinline__ uint32_t get_max_valence() const
    {
        return m_max_valence;
    }
    __device__ __forceinline__ uint32_t get_max_edge_incident_faces() const
    {
        return m_max_edge_incident_faces;
    }

    __device__ __forceinline__ uint32_t get_max_edge_adjacent_faces() const
    {
        return m_max_face_adjacent_faces;
    }
    __device__ __forceinline__ uint32_t get_num_patches() const
    {
        return m_num_patches;
    }
    __device__ __forceinline__ uint32_t* get_face_patch() const
    {
        return m_d_face_patch;
    }
    __device__ __forceinline__ uint32_t* get_edge_patch() const
    {
        return m_d_edge_patch;
    }
    __device__ __forceinline__ uint32_t* get_vertex_patch() const
    {
        return m_d_vertex_patch;
    }
    __device__ __forceinline__ uint32_t* get_patches_ltog_v() const
    {
        return m_d_patches_ltog_v;
    }
    __device__ __forceinline__ uint32_t* get_patches_ltog_e() const
    {
        return m_d_patches_ltog_e;
    }
    __device__ __forceinline__ uint32_t* get_patches_ltog_f() const
    {
        return m_d_patches_ltog_f;
    }
    __device__ __forceinline__ uint2* get_ad_size_ltog_v() const
    {
        return m_d_ad_size_ltog_v;
    }
    __device__ __forceinline__ uint2* get_ad_size_ltog_e() const
    {
        return m_d_ad_size_ltog_e;
    }
    __device__ __forceinline__ uint2* get_ad_size_ltog_f() const
    {
        return m_d_ad_size_ltog_f;
    }
    __device__ __forceinline__ uint16_t* get_patches_edges() const
    {
        return m_d_patches_edges;
    }
    __device__ __forceinline__ uint16_t* get_patches_faces() const
    {
        return m_d_patches_faces;
    }
    __device__ __forceinline__ uint4* get_ad_size() const
    {
        return m_d_ad_size;
    }
    __device__ __forceinline__ uint16_t* get_num_owned_faces() const
    {
        return m_d_num_owned_f;
    }

    __device__ __forceinline__ uint16_t* get_num_owned_edges() const
    {
        return m_d_num_owned_e;
    }

    __device__ __forceinline__ uint16_t* get_num_owned_vertices() const
    {
        return m_d_num_owned_v;
    }

    __device__ __forceinline__ PatchInfo* get_patches_info() const
    {
        return m_patches_info;
    }
    static __device__ __host__ __forceinline__ void unpack_edge_dir(
        const uint16_t edge_dir, uint16_t& edge, flag_t& dir)
    {
        dir  = (edge_dir & 1) != 0;
        edge = edge_dir >> 1;
    }

   private:
    // mesh elements count
    uint32_t m_num_edges, m_num_faces, m_num_vertices, m_max_valence,
        m_max_edge_incident_faces, m_max_face_adjacent_faces, m_num_patches;

    // face/vertex/edge patch (indexed by in global space)
    uint32_t *m_d_face_patch, *m_d_edge_patch, *m_d_vertex_patch;

    // mapping
    uint32_t *m_d_patches_ltog_v, *m_d_patches_ltog_e, *m_d_patches_ltog_f;
    uint2 *   m_d_ad_size_ltog_v, *m_d_ad_size_ltog_e, *m_d_ad_size_ltog_f;

    // incidence
    uint16_t *m_d_patches_edges, *m_d_patches_faces;

    //.x edge address .y edge size  .z face address .w face size
    uint4* m_d_ad_size;

    // number of owned faces, edges, and vertices per patch
    uint16_t *m_d_num_owned_f, *m_d_num_owned_e, *m_d_num_owned_v;

    PatchInfo* m_patches_info;
};
}  // namespace rxmesh