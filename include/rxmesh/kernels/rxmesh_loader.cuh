#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/rxmesh.h"
#include "rxmesh/rxmesh_context.h"

namespace RXMESH {

/**
 * load_patch_ad_size()
 */
__device__ __forceinline__ void load_patch_ad_size(const RXMeshContext& context,
                                                   const uint32_t       p_id,
                                                   uint4&               ad_size,
                                                   uint2& ad_size_ltog_v,
                                                   uint2& ad_size_ltog_e,
                                                   uint2& ad_size_ltog_f)
{

    ad_size.x = context.get_ad_size()[p_id].x;
    ad_size.y = context.get_ad_size()[p_id].y;
    ad_size.z = context.get_ad_size()[p_id].z;
    ad_size.w = context.get_ad_size()[p_id].w;

    ad_size_ltog_v = context.get_ad_size_ltog_v()[p_id];
    ad_size_ltog_e = context.get_ad_size_ltog_e()[p_id];
    ad_size_ltog_f = context.get_ad_size_ltog_f()[p_id];
    assert(ad_size.y % 2 == 0);
    assert(ad_size.w % context.get_face_degree() == 0);

    /*if (threadIdx.x == 0) {
        printf("\n   blockIdx.x= %u, p_id = %u \n"
            "   edges_add= %u, edges_size= %u \n"
            "   faces_add= %u, faces_size= %u \n"
            "   s_ad_size_ltog_v.x= %u, s_ad_size_ltog_v.y= %u \n"
            "   s_ad_size_ltog_e.x= %u, s_ad_size_ltog_e.y= %u \n"
            "   s_ad_size_ltog_f.x= %u, s_ad_size_ltog_f.y= %u \n",
            blockIdx.x, p_id,
            s_ad_size.x, s_ad_size.y, s_ad_size.z, s_ad_size.w,
            s_ad_size_ltog_v.x, s_ad_size_ltog_v.y,
            s_ad_size_ltog_e.x, s_ad_size_ltog_e.y,
            s_ad_size_ltog_f.x, s_ad_size_ltog_f.y);
    }*/
}

/**
 * load_patch_edges()
 */
__device__ __forceinline__ void load_patch_edges(const RXMeshContext& context,
                                                 uint16_t*    patch_edges,
                                                 const uint4& ad_sz)
{

    // whole block should be calling this

    // load edges
    assert(ad_sz.y % 2 == 0);
    uint32_t        size32 = ad_sz.y / 2;
    const uint32_t* edges_ptr32 =
        (const uint32_t*)(context.get_patches_edges() + ad_sz.x);
    uint32_t* patch_edges32 = (uint32_t*)(patch_edges);
#pragma unroll 2
    for (uint32_t i = threadIdx.x; i < size32; i += blockDim.x) {
        uint32_t a       = edges_ptr32[i];
        patch_edges32[i] = a;
    }
}

/**
 * load_patch_faces()
 */
__device__ __forceinline__ void load_patch_faces(const RXMeshContext& context,
                                                 uint16_t*    patch_faces,
                                                 const uint4& ad_sz)
{

    // whole block should be calling this

    // load faces
    assert(ad_sz.w % 3 == 0);

    uint32_t        size32   = ad_sz.w / 2;
    uint32_t        reminder = ad_sz.w % 2;
    const uint32_t* faces_ptr32 =
        (const uint32_t*)(context.get_patches_faces() + ad_sz.z);
    uint32_t* patch_faces32 = (uint32_t*)(patch_faces);
    //#pragma unroll 3
    for (uint32_t i = threadIdx.x; i < size32; i += blockDim.x) {
        uint32_t a       = faces_ptr32[i];
        patch_faces32[i] = a;
    }

    if (reminder != 0) {
        if (threadIdx.x == 0) {
            patch_faces[ad_sz.w - 1] =
                context.get_patches_faces()[ad_sz.z + ad_sz.w - 1];
        }
    }
}

/**
 * load_mapping()
 */
__device__ __forceinline__ void load_mapping(const RXMeshContext& context,
                                             const ELEMENT        ele,
                                             const uint2& s_ad_size_ltog,
                                             uint32_t*    mapping,
                                             const bool   keep_patch_bit)
{
    // whole block should be calling this
    for (uint32_t i = threadIdx.x, start = s_ad_size_ltog.x;
         i < s_ad_size_ltog.y;
         i += blockDim.x) {

        switch (ele) {
            case ELEMENT::VERTEX:
                if (keep_patch_bit) {
                    mapping[i] = context.get_patches_ltog_v()[i + start];
                } else {
                    mapping[i] = (context.get_patches_ltog_v()[i + start] >> 1);
                }

                break;
            case ELEMENT::EDGE:
                if (keep_patch_bit) {
                    mapping[i] = context.get_patches_ltog_e()[i + start];
                } else {
                    mapping[i] = (context.get_patches_ltog_e()[i + start] >> 1);
                }
                break;
            case ELEMENT::FACE:
                if (keep_patch_bit) {
                    mapping[i] = context.get_patches_ltog_f()[i + start];
                } else {
                    mapping[i] = (context.get_patches_ltog_f()[i + start] >> 1);
                }
                break;
            default:
                assert(1 != 1);
                break;
        }
    }
}

/**
 * load_mesh()
 */
__device__ __forceinline__ void load_mesh(const RXMeshContext& context,
                                          const bool           load_edges,
                                          const bool           load_faces,
                                          uint16_t*&           s_patch_edges,
                                          uint16_t*&           s_patch_faces,
                                          const uint4&         ad_size)
{

    if (load_edges) {
        load_patch_edges(context, s_patch_edges, ad_size);
    }
    // load patch faces
    if (load_faces) {
        if (load_edges) {
            // if we loaded the edges, then we need to move where
            // s_patch_faces is pointing at to avoid overwrite
            s_patch_faces = &s_patch_edges[ad_size.y];
        }
        load_patch_faces(context, s_patch_faces, ad_size);
    }
}

}  // namespace RXMESH
