#include "rxmesh/patch_info.h"

#include <cassert>
#include <type_traits>
#include <utility>



namespace rxmesh {

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
