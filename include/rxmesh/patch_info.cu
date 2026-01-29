#include "rxmesh/patch_info.h"

#include <cassert>
#include <type_traits>
#include <utility>

#ifdef __CUDA_ARCH__
#include "rxmesh/kernels/util.cuh"
#endif


namespace rxmesh {

__device__ std::pair<uint16_t, uint16_t> PatchInfo::get_edge_vertices(
    const uint16_t e_id) const
{
    const uint32_t uin32_val = reinterpret_cast<const uint32_t*>(ev)[e_id];

    uint16_t v0 = detail::extract_low_bits<16>(uin32_val);
    uint16_t v1 = detail::extract_high_bits<16>(uin32_val);

    assert(v0 == ev[2 * e_id + 0].id);
    assert(v1 == ev[2 * e_id + 1].id);

    return std::make_pair(v0, v1);
}

__host__ __device__ uint16_t
PatchInfo::count_num_owned(const uint32_t* owned_bitmask,
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

__device__ bool PatchInfo::is_dirty() const
{
#ifdef __CUDA_ARCH__
    return atomic_read(dirty) != 0;
#else
    return dirty[0] != 0;
#endif
}

template <typename HandleT>
__device__ __host__ HandleT PatchInfo::find(const LPPair::KeyT key,
                                            const LPPair*      table,
                                            const LPPair*      stash,
                                            const PatchStash&  pstash) const
{
    LPPair lp = get_lp<HandleT>().find(key, table, stash);
    if (lp.is_sentinel()) {
        return HandleT();
    } else {
        return HandleT(pstash.get_patch(lp), {lp.local_id_in_owner_patch()});
    }
}

template <typename HandleT>
__device__ __host__ HandleT PatchInfo::find(const LPPair::KeyT key,
                                            const LPPair*      table,
                                            const LPPair*      stash) const
{
    if (is_owned(typename HandleT::LocalT(key))) {
        return HandleT(patch_id, key);
    }
    LPPair lp = get_lp<HandleT>().find(key, table, stash);

    if (lp.is_sentinel()) {
        return HandleT();
    } else {
        return get_handle<HandleT>(lp);
    }
}

template <typename HandleT>
__device__ __host__ HandleT PatchInfo::get_handle(const LPPair lp) const
{
    return HandleT(patch_stash.get_patch(lp), {lp.local_id_in_owner_patch()});
}

template <typename HandleT>
__device__ __host__ const uint16_t* PatchInfo::get_num_elements() const
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
    return nullptr;
}

template <typename HandleT>
__device__ __host__ uint16_t PatchInfo::get_capacity() const
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
    return 0;
}

template <typename HandleT>
__device__ __host__ const uint32_t* PatchInfo::get_active_mask() const
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
    return nullptr;
}

template <typename HandleT>
__device__ __host__ uint32_t* PatchInfo::get_active_mask()
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
    return nullptr;
}

template <typename HandleT>
__device__ __host__ const uint32_t* PatchInfo::get_owned_mask() const
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
    return nullptr;
}

template <typename HandleT>
__device__ __host__ uint32_t* PatchInfo::get_owned_mask()
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
    return nullptr;
}

template <typename HandleT>
__device__ __host__ const LPHashTable& PatchInfo::get_lp() const
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
    return lp_v;
}

template <typename HandleT>
__device__ __host__ LPHashTable& PatchInfo::get_lp()
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
    return lp_v;
}

template <typename HandleT>
__host__ __device__ uint16_t PatchInfo::get_num_owned() const
{
    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        return count_num_owned(owned_mask_v, active_mask_v, num_vertices[0]);
    }
    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        return count_num_owned(owned_mask_e, active_mask_e, num_edges[0]);
    }
    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        return count_num_owned(owned_mask_f, active_mask_f, num_faces[0]);
    }
    return 0;
}

// Explicit instantiations
#define PATCH_INFO_INSTANTIATE(HandleT)                                          \
    template HandleT PatchInfo::find<HandleT>(                                   \
        const LPPair::KeyT, const LPPair*, const LPPair*, const PatchStash&)     \
        const;                                                                   \
    template HandleT PatchInfo::find<HandleT>(                                   \
        const LPPair::KeyT, const LPPair*, const LPPair*) const;                 \
    template HandleT         PatchInfo::get_handle<HandleT>(const LPPair) const; \
    template const uint16_t* PatchInfo::get_num_elements<HandleT>() const;       \
    template uint16_t        PatchInfo::get_capacity<HandleT>() const;           \
    template const uint32_t* PatchInfo::get_active_mask<HandleT>() const;        \
    template uint32_t*       PatchInfo::get_active_mask<HandleT>();              \
    template const uint32_t* PatchInfo::get_owned_mask<HandleT>() const;         \
    template uint32_t*       PatchInfo::get_owned_mask<HandleT>();               \
    template const LPHashTable& PatchInfo::get_lp<HandleT>() const;              \
    template LPHashTable&       PatchInfo::get_lp<HandleT>();                    \
    template uint16_t           PatchInfo::get_num_owned<HandleT>() const;

PATCH_INFO_INSTANTIATE(VertexHandle)
PATCH_INFO_INSTANTIATE(EdgeHandle)
PATCH_INFO_INSTANTIATE(FaceHandle)

#undef PATCH_INFO_INSTANTIATE

}  // namespace rxmesh
