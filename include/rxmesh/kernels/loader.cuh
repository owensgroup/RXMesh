#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/context.h"
#include "rxmesh/types.h"

namespace rxmesh {
/**
 * @brief
 * @param context
 * @param p_id
 * @param ad_size
 * @param ad_size_ltog_v
 * @param ad_size_ltog_e
 * @param ad_size_ltog_f
 * @return
 * TODO remove
 */
__device__ __forceinline__ void load_patch_ad_size(const Context& context,
                                                   const uint32_t p_id,
                                                   uint4&         ad_size,
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
    assert(ad_size.w % 3 == 0);
}

/**
 * @brief
 * @param context
 * @param patch_edges
 * @param ad_sz
 * @return
 * TODO remove
 */
__device__ __forceinline__ void load_patch_edges(const Context& context,
                                                 uint16_t*      patch_edges,
                                                 const uint4&   ad_sz)
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
 * @brief
 * @param context
 * @param patch_faces
 * @param ad_sz
 * @return
 */
__device__ __forceinline__ void load_patch_faces(const Context& context,
                                                 uint16_t*      patch_faces,
                                                 const uint4&   ad_sz)
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
 * @brief
 * @param context
 * @param ele
 * @param s_ad_size_ltog
 * @param mapping
 * @return
 * TODO remove
 */
__device__ __forceinline__ void load_mapping(const Context& context,
                                             const ELEMENT  ele,
                                             const uint2&   s_ad_size_ltog,
                                             uint32_t*      mapping)
{
    // whole block should be calling this
    for (uint32_t i = threadIdx.x, start = s_ad_size_ltog.x;
         i < s_ad_size_ltog.y;
         i += blockDim.x) {

        switch (ele) {
            case ELEMENT::VERTEX:
                mapping[i] = context.get_patches_ltog_v()[i + start];
                break;
            case ELEMENT::EDGE:
                mapping[i] = context.get_patches_ltog_e()[i + start];
                break;
            case ELEMENT::FACE:
                mapping[i] = context.get_patches_ltog_f()[i + start];
                break;
            default:
                assert(1 != 1);
                break;
        }
    }
}


/**
 * @brief
 * @param context
 * @param load_edges
 * @param load_faces
 * @param s_patch_edges
 * @param s_patch_faces
 * @param ad_size
 * @return
 * TODO remove
 */
__device__ __forceinline__ void load_mesh(const Context& context,
                                          const bool     load_edges,
                                          const bool     load_faces,
                                          uint16_t*&     s_patch_edges,
                                          uint16_t*&     s_patch_faces,
                                          const uint4&   ad_size)
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

/**
 * @brief load the patch FE
 * @param patch_info input patch info
 * @param patch_faces output FE
 * @return
 */
template <uint32_t blockThreads>
__device__ __forceinline__ void load_patch_FE(const PatchInfo& patch_info,
                                              LocalEdgeT*      fe)
{
    const uint32_t  size     = patch_info.num_faces * 3;
    const uint32_t  size32   = size / 2;
    const uint32_t  reminder = size % 2;
    const uint32_t* input_fe32 =
        reinterpret_cast<const uint32_t*>(patch_info.fe);
    uint32_t* output_fe32 = reinterpret_cast<uint32_t*>(fe);
    //#pragma unroll 3
    for (uint32_t i = threadIdx.x; i < size32; i += blockThreads) {
        uint32_t a     = input_fe32[i];
        output_fe32[i] = a;
    }

    if (reminder != 0) {
        if (threadIdx.x == 0) {
            fe[size - 1] = patch_info.fe[size - 1];
        }
    }
}

/**
 * @brief load the patch EV
 * @param patch_info input patch info
 * @param ev output EV
 * @return
 */
template <uint32_t blockThreads>
__device__ __forceinline__ void load_patch_EV(const PatchInfo& patch_info,
                                              LocalVertexT*    ev)
{
    const uint32_t  num_edges = patch_info.num_edges;
    const uint32_t* input_ev32 =
        reinterpret_cast<const uint32_t*>(patch_info.ev);
    uint32_t* output_ev32 = reinterpret_cast<uint32_t*>(ev);
#pragma unroll 2
    for (uint32_t i = threadIdx.x; i < num_edges; i += blockThreads) {
        uint32_t a     = input_ev32[i];
        output_ev32[i] = a;
    }
}

/**
 * @brief load the patch topology i.e., EV and FE
 * @param patch_info input patch info
 * @param load_ev input indicates if we should load EV
 * @param load_fe input indicates if we should load FE
 * @param s_ev where EV will be loaded
 * @param s_fe where FE will be loaded
 * @return
 */
template <uint32_t blockThreads>
__device__ __forceinline__ void load_mesh(const PatchInfo& patch_info,
                                          const bool       load_ev,
                                          const bool       load_fe,
                                          LocalVertexT*&   s_ev,
                                          LocalEdgeT*&     s_fe)
{

    if (load_ev) {
        load_patch_EV<blockThreads>(patch_info, s_ev);
    }
    // load patch faces
    if (load_fe) {
        if (load_ev) {
            // if we loaded the edges, then we need to move where
            // s_fe is pointing at to avoid overwrite
            s_fe =
                reinterpret_cast<LocalEdgeT*>(&s_ev[patch_info.num_edges * 2]);
        }
        load_patch_FE<blockThreads>(patch_info, s_fe);
    }
}
}  // namespace rxmesh
