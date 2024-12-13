#pragma once
#include "rxmesh/rxmesh_static.h"

#include <cooperative_groups.h>

#include "rxmesh/bitmask.cuh"
#include "rxmesh/kernels/rxmesh_queries.cuh"

#define SLICE_GGP

namespace rxmesh {
namespace detail {

template <uint32_t blockThreads>
__device__ void bi_assignment(cooperative_groups::thread_block& block,
                              const uint16_t                    num_vertices,
                              const uint16_t                    num_edges,
                              const uint16_t                    num_faces,
                              const Bitmask&                    s_owned_v,
                              const Bitmask&                    s_owned_e,
                              const Bitmask&                    s_owned_f,
                              const Bitmask&                    s_active_v,
                              const Bitmask&                    s_active_e,
                              const Bitmask&                    s_active_f,
                              const uint16_t*                   s_ev,
                              const uint16_t*                   s_fv,
                              Bitmask&                          s_new_p_owned_v,
                              Bitmask&                          s_new_p_owned_e,
                              Bitmask& s_new_p_owned_f);


template <uint32_t blockThreads>
__device__ void bi_assignment_ggp(cooperative_groups::thread_block& block,
                                  const uint16_t  num_vertices,
                                  const Bitmask&  s_owned_v,
                                  const bool      ignore_owned_v,
                                  const Bitmask&  s_active_v,
                                  const uint16_t* m_s_vv_offset,
                                  const uint16_t* m_s_vv,
                                  Bitmask&        s_assigned_v,
                                  Bitmask&        s_current_frontier_v,
                                  Bitmask&        s_next_frontier_v,
                                  Bitmask&        s_partition_a_v,
                                  Bitmask&        s_partition_b_v,
                                  int             num_iter);

template <uint32_t blockThreads>
__device__ void slice(Context&                          context,
                      cooperative_groups::thread_block& block,
                      PatchInfo&                        pi,
                      const uint32_t                    new_patch_id,
                      const uint16_t                    num_vertices,
                      const uint16_t                    num_edges,
                      const uint16_t                    num_faces,
                      PatchStash&                       s_new_patch_stash,
                      // PatchStash&     s_original_patch_stash,
                      Bitmask&        s_owned_v,
                      Bitmask&        s_owned_e,
                      Bitmask&        s_owned_f,
                      const Bitmask&  s_active_v,
                      const Bitmask&  s_active_e,
                      const Bitmask&  s_active_f,
                      const uint16_t* s_ev,
                      const uint16_t* s_fe,
                      Bitmask&        s_new_p_active_v,
                      Bitmask&        s_new_p_active_e,
                      Bitmask&        s_new_p_active_f,
                      Bitmask&        s_new_p_owned_v,
                      Bitmask&        s_new_p_owned_e,
                      Bitmask&        s_new_p_owned_f);

template <uint32_t blockThreads, typename AttributeT>
__inline__ __device__ void post_slicing_update_attributes(
    const PatchInfo& pi,
    const uint32_t   new_patch_id,
    const Bitmask&   ownership_change_v,
    const Bitmask&   ownership_change_e,
    const Bitmask&   ownership_change_f,
    AttributeT&      attribute)
{
    using HandleT = typename AttributeT::HandleType;

    const uint32_t num_attr = attribute.get_num_attributes();

    const uint16_t num_elements = pi.get_num_elements<HandleT>()[0];

    const uint32_t patch_id = pi.patch_id;


    for (uint16_t vp = threadIdx.x; vp < num_elements; vp += blockThreads) {
        bool change = false;
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            change = ownership_change_v(vp);
        }
        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            change = ownership_change_e(vp);
        }
        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            change = ownership_change_f(vp);
        }
        if (change) {
            for (uint32_t attr = 0; attr < num_attr; ++attr) {
                attribute(new_patch_id, vp, attr) =
                    attribute(patch_id, vp, attr);
            }
        }
    }
}


template <uint32_t blockThreads,
          uint32_t itemPerThread,
          typename... AttributesT>
__global__ static void slice_patches(Context        context,
                                     const uint32_t current_num_patches,
                                     AttributesT... attributes)
{
    // ev, fe, active_v/e/f, owned_v/e/f, patch_v/e/f
    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    const uint32_t pid = blockIdx.x;
    if (pid >= current_num_patches) {
        return;
    }

    auto alloc_masks = [&](uint16_t        num_elements,
                           Bitmask&        owned,
                           Bitmask&        active,
                           Bitmask&        new_active,
                           Bitmask&        patch,
                           const uint32_t* g_owned,
                           const uint32_t* g_active) {
        owned      = Bitmask(num_elements, shrd_alloc);
        active     = Bitmask(num_elements, shrd_alloc);
        new_active = Bitmask(num_elements, shrd_alloc);
        patch      = Bitmask(num_elements, shrd_alloc);

        owned.reset(block);
        active.reset(block);
        new_active.reset(block);
        patch.reset(block);

        // to remove the racecheck hazard report due to WAW on owned and active
        block.sync();

        detail::load_async(block,
                           reinterpret_cast<const char*>(g_owned),
                           owned.num_bytes(),
                           reinterpret_cast<char*>(owned.m_bitmask),
                           false);
        detail::load_async(block,
                           reinterpret_cast<const char*>(g_active),
                           active.num_bytes(),
                           reinterpret_cast<char*>(active.m_bitmask),
                           false);
    };

    PatchInfo pi = context.m_patches_info[pid];
    if (pi.should_slice) {
        const uint16_t num_vertices = pi.num_vertices[0];
        const uint16_t num_edges    = pi.num_edges[0];
        const uint16_t num_faces    = pi.num_faces[0];

        __shared__ uint32_t s_new_patch_id;
        if (threadIdx.x == 0) {
            // check if any of the neighbor patches are dirty. If so, we don't
            // slice because it messes the connectivity of other neighbor
            // patches
            bool ok = true;
            for (uint32_t i = 0; i < PatchStash::stash_size; ++i) {
                uint32_t q = pi.patch_stash.get_patch(i);
                if (q != INVALID32) {
                    if (context.m_patches_info[q].is_dirty()) {
                        // printf("\n slicing: patch %u finds %u dirty",
                        //        pi.patch_id,
                        //        q);
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) {
                s_new_patch_id =
                    ::atomicAdd(context.m_num_patches, uint32_t(1));
                assert(s_new_patch_id < context.m_max_num_patches);
                context.m_patch_scheduler.push(s_new_patch_id);
                // printf("\n slicing %u into %u", pi.patch_id, s_new_patch_id);
            } else {
                s_new_patch_id                           = INVALID32;
                context.m_patches_info[pid].should_slice = false;
            }
        }
        Bitmask s_owned_v, s_owned_e, s_owned_f;
        Bitmask s_active_v, s_active_e, s_active_f;
        Bitmask s_new_p_owned_v, s_new_p_owned_e, s_new_p_owned_f;
        Bitmask s_new_p_active_v, s_new_p_active_e, s_new_p_active_f;

        Bitmask s_assigned_v         = Bitmask(num_vertices, shrd_alloc);
        Bitmask s_current_frontier_v = Bitmask(num_vertices, shrd_alloc);
        Bitmask s_next_frontier_v    = Bitmask(num_vertices, shrd_alloc);

        block.sync();
        if (s_new_patch_id == INVALID32) {
            return;
        }


        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(2 * 2 * num_edges);
        uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(3 * num_faces);
        uint16_t* s_fv = s_fe;

        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(pi.ev),
                           2 * num_edges,
                           s_ev,
                           false);


        PatchStash s_new_patch_stash;
        s_new_patch_stash.m_stash =
            shrd_alloc.alloc<uint32_t>(PatchStash::stash_size);

        // PatchStash s_original_patch_stash;
        // s_original_patch_stash.m_stash =
        //     shrd_alloc.alloc<uint32_t>(PatchStash::stash_size);

        alloc_masks(num_vertices,
                    s_owned_v,
                    s_active_v,
                    s_new_p_active_v,
                    s_new_p_owned_v,
                    pi.owned_mask_v,
                    pi.active_mask_v);
        alloc_masks(num_edges,
                    s_owned_e,
                    s_active_e,
                    s_new_p_active_e,
                    s_new_p_owned_e,
                    pi.owned_mask_e,
                    pi.active_mask_e);
        alloc_masks(num_faces,
                    s_owned_f,
                    s_active_f,
                    s_new_p_active_f,
                    s_new_p_owned_f,
                    pi.owned_mask_f,
                    pi.active_mask_f);

        cooperative_groups::wait(block);
        block.sync();

#ifdef SLICE_GGP
        // compute VV
        uint16_t* s_vv        = &s_ev[num_vertices + 1];
        uint16_t* s_vv_offset = s_ev;
        v_e<blockThreads, itemPerThread>(
            num_vertices, num_edges, s_ev, s_vv, s_active_e.m_bitmask);
        block.sync();
        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
            uint16_t start = s_vv_offset[v];
            uint16_t end   = s_vv_offset[v + 1];

            for (uint16_t e = start; e < end; ++e) {
                uint16_t edge = s_vv[e];
                auto [v0, v1] = pi.get_edge_vertices(edge);
                assert(v0 != INVALID16 && v1 != INVALID16);
                assert(v0 == v || v1 == v);
                s_vv[e] = (v0 == v) * v1 + (v1 == v) * v0;
            }
        }
        block.sync();

        // slice
        bi_assignment_ggp<blockThreads>(block,
                                        num_vertices,
                                        s_owned_v,
                                        false,
                                        s_active_v,
                                        s_vv_offset,
                                        s_vv,
                                        s_assigned_v,
                                        s_current_frontier_v,
                                        s_next_frontier_v,
                                        s_new_p_active_v,  // reuse
                                        s_new_p_owned_v,
                                        10);
        block.sync();

        s_new_p_active_v.reset(block);

        // load FE and EV(again)
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(pi.ev),
                           2 * num_edges,
                           s_ev,
                           false);
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(pi.fe),
                           3 * num_faces,
                           s_fe,
                           true);
        block.sync();

        // compute FV
        f_v<blockThreads>(
            num_edges, s_ev, num_faces, s_fv, s_active_f.m_bitmask);
        block.sync();

        // Assign faces through vertices
        for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
            if (s_active_f(f)) {
                const uint16_t v0(s_fv[3 * f + 0]), v1(s_fv[3 * f + 1]),
                    v2(s_fv[3 * f + 2]);
                assert(v0 < num_vertices);
                assert(v1 < num_vertices);
                assert(v2 < num_vertices);
                if (s_new_p_owned_v(v0) || s_new_p_owned_v(v1) ||
                    s_new_p_owned_v(v2)) {
                    s_new_p_owned_f.set(f, true);
                }
            }
        }
        block.sync();


#else
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(pi.fe),
                           3 * num_faces,
                           s_fe,
                           true);
        block.sync();
        f_v<blockThreads>(
            num_edges, s_ev, num_faces, s_fv, s_active_f.m_bitmask);
        block.sync();

        bi_assignment<blockThreads>(block,
                                    num_vertices,
                                    num_edges,
                                    num_faces,
                                    s_owned_v,
                                    s_owned_e,
                                    s_owned_f,
                                    s_active_v,
                                    s_active_e,
                                    s_active_f,
                                    s_ev,
                                    s_fv,
                                    s_new_p_owned_v,
                                    s_new_p_owned_e,
                                    s_new_p_owned_f);
        block.sync();
#endif

        /// Load FE
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(pi.fe),
                           3 * num_faces,
                           s_fe,
                           true);
        block.sync();

        // assign edges through faces
        for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
            if (s_active_f(f) && s_new_p_owned_f(f) && s_owned_f(f)) {
                const uint16_t e0(s_fe[3 * f + 0] >> 1),
                    e1(s_fe[3 * f + 1] >> 1), e2(s_fe[3 * f + 2] >> 1);
                assert(e0 < num_edges);
                assert(e1 < num_edges);
                assert(e2 < num_edges);
                s_new_p_owned_e.set(e0, true);
                s_new_p_owned_e.set(e1, true);
                s_new_p_owned_e.set(e2, true);
            }
        }
        block.sync();

#ifndef NDEBUG
        // record the number of vertices/edges/faces that are active and owned
        // then we make sure that the number of vertices/edges/faces that are
        // active and owned by the two patches (old and new sliced one) are
        // equal the old number of vertices/edges/faces
        __shared__ uint32_t s_total_num_vertices, s_total_num_edges,
            s_total_num_faces;
        if (threadIdx.x == 0) {
            s_total_num_vertices = 0;
            s_total_num_edges    = 0;
            s_total_num_faces    = 0;
        }
        block.sync();
        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
            if (s_active_v(v) && s_owned_v(v)) {
                ::atomicAdd(&s_total_num_vertices, uint32_t(1));
            }
        }
        for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
            if (s_active_e(e) && s_owned_e(e)) {
                ::atomicAdd(&s_total_num_edges, uint32_t(1));
            }
        }
        for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
            if (s_active_f(f) && s_owned_f(f)) {
                ::atomicAdd(&s_total_num_faces, uint32_t(1));
            }
        }
        block.sync();
#endif

        slice<blockThreads>(context,
                            block,
                            pi,
                            s_new_patch_id,
                            num_vertices,
                            num_edges,
                            num_faces,
                            s_new_patch_stash,
                            // ls_original_patch_stash,
                            s_owned_v,
                            s_owned_e,
                            s_owned_f,
                            s_active_v,
                            s_active_e,
                            s_active_f,
                            s_ev,
                            s_fe,
                            s_new_p_active_v,
                            s_new_p_active_e,
                            s_new_p_active_f,
                            s_new_p_owned_v,
                            s_new_p_owned_e,
                            s_new_p_owned_f);

        (
            [&] {
                post_slicing_update_attributes<blockThreads>(pi,
                                                             s_new_patch_id,
                                                             s_new_p_owned_v,
                                                             s_new_p_owned_e,
                                                             s_new_p_owned_f,
                                                             attributes);
            }(),
            ...);

        context.m_patches_info[pid].should_slice = false;

#ifndef NDEBUG
        block.sync();
        __shared__ uint32_t s_new_num_vertices, s_new_num_edges,
            s_new_num_faces, s_old_num_vertices, s_old_num_edges,
            s_old_num_faces;
        if (threadIdx.x == 0) {
            s_new_num_vertices = 0;
            s_new_num_edges    = 0;
            s_new_num_faces    = 0;
            s_old_num_vertices = 0;
            s_old_num_edges    = 0;
            s_old_num_faces    = 0;
        }
        block.sync();
        PatchInfo old_pi = context.m_patches_info[pid];
        PatchInfo new_pi = context.m_patches_info[s_new_patch_id];
        // vertices
        for (uint16_t v = threadIdx.x; v < old_pi.num_vertices[0];
             v += blockThreads) {
            if (!old_pi.is_deleted(LocalVertexT(v)) &&
                old_pi.is_owned(LocalVertexT(v))) {
                ::atomicAdd(&s_old_num_vertices, uint32_t(1));
            }
        }
        for (uint16_t v = threadIdx.x; v < new_pi.num_vertices[0];
             v += blockThreads) {
            if (!new_pi.is_deleted(LocalVertexT(v)) &&
                new_pi.is_owned(LocalVertexT(v))) {
                ::atomicAdd(&s_new_num_vertices, uint32_t(1));
            }
        }

        // edges
        for (uint16_t e = threadIdx.x; e < old_pi.num_edges[0];
             e += blockThreads) {
            if (!old_pi.is_deleted(LocalEdgeT(e)) &&
                old_pi.is_owned(LocalEdgeT(e))) {
                ::atomicAdd(&s_old_num_edges, uint32_t(1));
            }
        }
        for (uint16_t e = threadIdx.x; e < new_pi.num_edges[0];
             e += blockThreads) {
            if (!new_pi.is_deleted(LocalEdgeT(e)) &&
                new_pi.is_owned(LocalEdgeT(e))) {
                ::atomicAdd(&s_new_num_edges, uint32_t(1));
            }
        }

        // faces
        for (uint16_t f = threadIdx.x; f < old_pi.num_faces[0];
             f += blockThreads) {
            if (!old_pi.is_deleted(LocalFaceT(f)) &&
                old_pi.is_owned(LocalFaceT(f))) {
                ::atomicAdd(&s_old_num_faces, uint32_t(1));
            }
        }
        for (uint16_t f = threadIdx.x; f < new_pi.num_faces[0];
             f += blockThreads) {
            if (!new_pi.is_deleted(LocalFaceT(f)) &&
                new_pi.is_owned(LocalFaceT(f))) {
                ::atomicAdd(&s_new_num_faces, uint32_t(1));
            }
        }
        block.sync();

        if (threadIdx.x == 0) {
            assert(s_total_num_vertices ==
                   s_new_num_vertices + s_old_num_vertices);
            assert(s_total_num_edges == s_new_num_edges + s_old_num_edges);
            assert(s_total_num_faces == s_new_num_faces + s_old_num_faces);
            // printf(
            //     "\n slicing %u into %u, #Vt= %u, #Vo= %u, #Vn= %u, #Et= %u, "
            //     "#Eo= %u, #En= %u, #Ft= %u, #Fo= %u, #Fn= %u",
            //     pi.patch_id,
            //     s_new_patch_id,
            //     s_total_num_vertices,
            //     s_old_num_vertices,
            //     s_new_num_vertices,
            //     s_total_num_edges,
            //     s_old_num_edges,
            //     s_new_num_edges,
            //     s_total_num_faces,
            //     s_old_num_faces,
            //     s_new_num_faces);
        }

        // check ribbons for new and old patch

        // auto check_ribbon =
        //     [](PatchInfo& info, char* name, PatchInfo& other_info) {
        //         // vertices
        //         for (uint16_t v = threadIdx.x; v < info.num_vertices[0];
        //              v += blockThreads) {
        //             if (!info.is_deleted(LocalVertexT(v)) &&
        //                 !info.is_owned(LocalVertexT(v))) {
        //
        //                 LPPair lp = info.get_lp<VertexHandle>().find(
        //                     v, nullptr, nullptr);
        //
        //                 if (lp.is_sentinel()) {
        //                     printf(
        //                         "\n @@ %s - vertex: B=%u, T= %u, patch_id "
        //                         "= %u, v= %u, other_info.is_deleted= %d, "
        //                         "other_info.is_owned= %d",
        //                         name,
        //                         blockIdx.x,
        //                         threadIdx.x,
        //                         info.patch_id,
        //                         v,
        //                         other_info.is_deleted(LocalVertexT(v)),
        //                         other_info.is_owned(LocalVertexT(v)));
        //                 }
        //                 myAssert(!lp.is_sentinel());
        //             }
        //         }
        //
        //
        //         // edges
        //         for (uint16_t e = threadIdx.x; e < info.num_edges[0];
        //              e += blockThreads) {
        //             if (!info.is_deleted(LocalEdgeT(e)) &&
        //                 !info.is_owned(LocalEdgeT(e))) {
        //
        //                 LPPair lp =
        //                     info.get_lp<EdgeHandle>().find(e, nullptr,
        //                     nullptr);
        //
        //                 if (lp.is_sentinel()) {
        //                     printf(
        //                         "\n @@ %s - edge: B=%u, T= %u, patch_id = "
        //                         "%u, e= %u, other_info.is_deleted= %d, "
        //                         "other_info.is_owned= %d",
        //                         name,
        //                         blockIdx.x,
        //                         threadIdx.x,
        //                         info.patch_id,
        //                         e,
        //                         other_info.is_deleted(LocalEdgeT(e)),
        //                         other_info.is_owned(LocalEdgeT(e)));
        //                 }
        //                 myAssert(!lp.is_sentinel());
        //             }
        //         }
        //
        //
        //         // faces
        //         for (uint16_t f = threadIdx.x; f < info.num_faces[0];
        //              f += blockThreads) {
        //             if (!info.is_deleted(LocalFaceT(f)) &&
        //                 !info.is_owned(LocalFaceT(f))) {
        //
        //                 LPPair lp =
        //                     info.get_lp<FaceHandle>().find(f, nullptr,
        //                     nullptr);
        //
        //                 if (lp.is_sentinel()) {
        //                     printf(
        //                         "\n @@ %s - face: B=%u, T= %u, patch_id = "
        //                         "%u, f= %u, other_info.is_deleted= %d, "
        //                         "other_info.is_owned= %d",
        //                         name,
        //                         blockIdx.x,
        //                         threadIdx.x,
        //                         info.patch_id,
        //                         f,
        //                         other_info.is_deleted(LocalFaceT(f)),
        //                         other_info.is_owned(LocalFaceT(f)));
        //                 }
        //                 myAssert(!lp.is_sentinel());
        //             }
        //         }
        //     };
        //
        // check_ribbon(old_pi, "old_pi", new_pi);
        // check_ribbon(new_pi, "new_pi", old_pi);

#endif
    }
}
}  // namespace detail


class RXMeshDynamic : public RXMeshStatic
{
   public:
    RXMeshDynamic(const RXMeshDynamic&) = delete;

    /**
     * @brief Constructor using path to obj file
     * @param file_path path to an obj file
     */
    explicit RXMeshDynamic(const std::string file_path,
                           const std::string patcher_file             = "",
                           const uint32_t    patch_size               = 256,
                           const float       capacity_factor          = 1.8,
                           const float       patch_alloc_factor       = 5.0,
                           const float       lp_hashtable_load_factor = 0.5)
        : RXMeshStatic(file_path,
                       patcher_file,
                       patch_size,
                       capacity_factor,
                       patch_alloc_factor,
                       lp_hashtable_load_factor)
    {
    }

    /**
     * @brief Constructor using triangles and vertices
     * @param fv Face incident vertices as read from an obj file
     */
    explicit RXMeshDynamic(std::vector<std::vector<uint32_t>>& fv,
                           const std::string patcher_file             = "",
                           const uint32_t    patch_size               = 256,
                           const float       capacity_factor          = 1.8,
                           const float       patch_alloc_factor       = 5.0,
                           const float       lp_hashtable_load_factor = 0.5)
        : RXMeshStatic(fv,
                       patcher_file,
                       patch_size,
                       capacity_factor,
                       patch_alloc_factor,
                       lp_hashtable_load_factor)
    {
    }

    /**
     * @brief save/seralize the patcher info to a file
     * @param filename
     */
    virtual void save(std::string filename) override;

    /**
     * @brief populate the launch_box with grid size and dynamic shared memory
     * needed for a kernel that may use dynamic and query operations
     * @param op List of query operations done inside the kernel
     * @param launch_box input launch box to be populated
     * @param kernel The kernel to be launched
     * @param is_dyn if there will be dynamic updates
     * @param oriented if the query is oriented. Valid only for Op::VV queries
     * @param with_vertex_valence if vertex valence is requested to be
     * pre-computed and stored in shared memory
     * @param is_concurrent: in case of multiple queries (i.e., op.size() > 1),
     * this parameter indicates if queries needs to be access at the same time
     * @param user_shmem a (lambda) function that takes the number of vertices,
     * edges, and faces and returns additional user-desired shared memory in
     * bytes
     */
    template <uint32_t blockThreads>
    void prepare_launch_box(
        const std::vector<Op>    op,
        LaunchBox<blockThreads>& launch_box,
        const void*              kernel,
        const bool               is_dyn              = true,
        const bool               oriented            = false,
        const bool               with_vertex_valence = false,
        const bool               is_concurrent       = false,
        std::function<size_t(uint32_t, uint32_t, uint32_t)> user_shmem =
            [](uint32_t v, uint32_t e, uint32_t f) { return 0; }) const
    {
        update_launch_box(op,
                          launch_box,
                          kernel,
                          is_dyn,
                          oriented,
                          with_vertex_valence,
                          is_concurrent,
                          user_shmem);

        RXMESH_TRACE(
            "RXMeshDynamic::calc_shared_memory() launching {} blocks with "
            "{} threads on the device",
            launch_box.blocks,
            blockThreads);


        check_shared_memory(launch_box.smem_bytes_dyn,
                            launch_box.smem_bytes_static,
                            launch_box.num_registers_per_thread,
                            launch_box.local_mem_per_thread,
                            blockThreads,
                            kernel);
    }


    /**
     * @brief populate the launch_box with grid size and dynamic shared memory
     * needed for a kernel that may use dynamic and query operations. Similar
     * to prepare_launch_box but here we don't do any checks to verify that
     * the amount of shared memory is okay and we don't print any info. This
     * function can be used to update the launch box in a loop where printing
     * out info could impact the timing
     * @param op List of query operations done inside the kernel
     * @param launch_box input launch box to be populated
     * @param kernel The kernel to be launched
     * @param is_dyn if there will be dynamic updates
     * @param oriented if the query is oriented. Valid only for Op::VV queries
     * @param with_vertex_valence if vertex valence is requested to be
     * pre-computed and stored in shared memory
     * @param is_concurrent: in case of multiple queries (i.e., op.size() > 1),
     * this parameter indicates if queries needs to be access at the same time
     * @param user_shmem a (lambda) function that takes the number of vertices,
     * edges, and faces and returns additional user-desired shared memory in
     * bytes
     */
    template <uint32_t blockThreads>
    void update_launch_box(
        const std::vector<Op>    op,
        LaunchBox<blockThreads>& launch_box,
        const void*              kernel,
        const bool               is_dyn              = true,
        const bool               oriented            = false,
        const bool               with_vertex_valence = false,
        const bool               is_concurrent       = false,
        std::function<size_t(uint32_t, uint32_t, uint32_t)> user_shmem =
            [](uint32_t v, uint32_t e, uint32_t f) { return 0; }) const
    {

        launch_box.blocks = this->m_num_patches;

        // static query shared memory
        size_t static_shmem = 0;
        for (auto o : op) {
            size_t sh = this->calc_shared_memory<blockThreads>(o, oriented);
            if (is_concurrent) {
                static_shmem += sh;
            } else {
                static_shmem = std::max(static_shmem, sh);
            }
        }

        const uint16_t vertex_cap = get_per_patch_max_vertex_capacity();
        const uint16_t edge_cap   = get_per_patch_max_edge_capacity();
        const uint16_t face_cap   = get_per_patch_max_face_capacity();

        if (is_dyn) {

            // connecivity (FE and EV) shared memory
            size_t connectivity_shmem = 0;
            connectivity_shmem += 3 * face_cap * sizeof(uint16_t) +
                                  2 * edge_cap * sizeof(uint16_t) +
                                  2 * ShmemAllocator::default_alignment;

            // cavity ID (which overlapped with hashtable shared memory)
            size_t cavity_id_shmem = 0;
            cavity_id_shmem += std::max(
                vertex_cap * sizeof(uint16_t),
                max_lp_hashtable_capacity<LocalVertexT>() * sizeof(LPPair));
            cavity_id_shmem += std::max(
                edge_cap * sizeof(uint16_t),
                max_lp_hashtable_capacity<LocalEdgeT>() * sizeof(LPPair));
            cavity_id_shmem += std::max(
                face_cap * sizeof(uint16_t),
                max_lp_hashtable_capacity<LocalFaceT>() * sizeof(LPPair));
            cavity_id_shmem += 3 * ShmemAllocator::default_alignment;

            // cavity boundary edges
            size_t cavity_bdr_shmem = 0;
            cavity_bdr_shmem +=
                edge_cap * sizeof(uint16_t) + ShmemAllocator::default_alignment;

            // store cavity size (assume number of cavities is half the patch
            // size)
            const uint16_t half_face_cap = DIVIDE_UP(face_cap, 2);

            size_t cavity_size_shmem = 0;
            cavity_size_shmem +=
                half_face_cap * sizeof(int) + ShmemAllocator::default_alignment;

            // cavity src element
            size_t cavity_creator_shmem = half_face_cap * sizeof(uint16_t) +
                                          ShmemAllocator::default_alignment;

            // size_t q_lp_shmem =
            //     std::max(max_lp_hashtable_capacity<LocalVertexT>(),
            //              max_lp_hashtable_capacity<LocalEdgeT>());
            //
            // q_lp_shmem =
            //     std::max(q_lp_shmem,
            //              size_t(max_lp_hashtable_capacity<LocalFaceT>())) *
            //     sizeof(LPPair);

            // active, owned, migrate(for vertices only), src bitmask (for
            // vertices and edges only), src connect (for vertices and edges
            // only), ownership owned_cavity_bdry (for vertices only), ribbonize
            // (for vertices only) added_to_lp, in_cavity, recover
            size_t bitmasks_shmem = 0;
            bitmasks_shmem += 10 * detail::mask_num_bytes(vertex_cap) +
                              10 * ShmemAllocator::default_alignment;
            bitmasks_shmem += 7 * detail::mask_num_bytes(edge_cap) +
                              7 * ShmemAllocator::default_alignment;
            bitmasks_shmem += 5 * detail::mask_num_bytes(face_cap) +
                              5 * ShmemAllocator::default_alignment;

            // active cavity bitmask
            bitmasks_shmem += detail::mask_num_bytes(face_cap);


            // correspondence buffer
            static_assert(LPPair::PatchStashNumBits <= 8);

            const size_t cv = (sizeof(uint16_t) + sizeof(uint8_t)) * vertex_cap;
            const size_t ce = (sizeof(uint16_t) + sizeof(uint8_t)) * edge_cap;
            const size_t cf = (sizeof(uint16_t) + sizeof(uint8_t)) * face_cap;
            const size_t correspond_shmem =
                ce + std::max(cv, cf) + 4 * ShmemAllocator::default_alignment;

            // shared memory is the max of 1. static query shared memory + the
            // cavity ID shared memory (since we need to mark seed elements) 2.
            // dynamic rxmesh shared memory which includes cavity ID shared
            // memory and other things
            launch_box.smem_bytes_dyn = std::max(
                connectivity_shmem + cavity_id_shmem + cavity_bdr_shmem +
                    cavity_size_shmem + bitmasks_shmem + correspond_shmem +
                    cavity_creator_shmem,
                static_shmem + cavity_id_shmem + cavity_creator_shmem);
        } else {
            launch_box.smem_bytes_dyn = static_shmem;
        }

        launch_box.smem_bytes_dyn += user_shmem(vertex_cap, edge_cap, face_cap);

        if (with_vertex_valence) {
            if (get_input_max_valence() > 256) {
                RXMESH_ERROR(
                    "RXMeshDynamic::prepare_launch_box() input max valence if "
                    "greater than 256 and thus using uint8_t to store the "
                    "vertex valence will lead to overflow");
            }
            launch_box.smem_bytes_dyn +=
                this->m_max_vertices_per_patch * sizeof(uint8_t) +
                ShmemAllocator::default_alignment;
        }

        check_shared_memory(launch_box.smem_bytes_dyn,
                            launch_box.smem_bytes_static,
                            launch_box.num_registers_per_thread,
                            launch_box.local_mem_per_thread,
                            blockThreads,
                            kernel,
                            false);
    }

    virtual ~RXMeshDynamic() = default;

    /**
     * @brief check if there is remaining patches not processed yet
     */
    bool is_queue_empty(cudaStream_t stream = NULL)
    {
        return this->m_rxmesh_context.m_patch_scheduler.is_empty(stream);
    }


    /**
     * @brief reset the patches for a another kernel. This needs only to be
     * called where more than one kernel is called. For a single kernel, the
     * queue is initialized during the construction so the user does not to call
     * this
     */
    void reset_scheduler()
    {
        this->m_rxmesh_context.m_patch_scheduler.refill(get_num_patches());
    }

    /**
     * @brief Validate the topology information stored in RXMesh. All checks are
     * done on the information stored on the GPU memory and thus all checks are
     * done on the GPU
     * @return true in case all information stored are valid
     */
    bool validate();

    /**
     * @brief cleanup after topology changes by removing surplus elements
     * and make sure that hashtable store owner patches. Also, reset the number
     * of vertices/edges/faces
     */
    void cleanup();

    /**
     * @brief slice a patch if the number of faces in the patch is greater
     * than a threshold
     */
    template <typename... AttributesT>
    void slice_patches(AttributesT... attributes)
    {

        const uint32_t grid_size = get_num_patches();

        CUDA_ERROR(cudaMemcpy(&this->m_max_vertices_per_patch,
                              this->m_rxmesh_context.m_max_num_vertices,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(&this->m_max_edges_per_patch,
                              this->m_rxmesh_context.m_max_num_edges,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(&this->m_max_faces_per_patch,
                              this->m_rxmesh_context.m_max_num_faces,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        // ev, fe
        uint32_t dyn_shmem =
            2 * ShmemAllocator::default_alignment +
            (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
            (2 * 2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

        // active_v/e/f, owned_v/e/f, patch_v/e/f
        dyn_shmem +=
            7 * detail::mask_num_bytes(this->m_max_vertices_per_patch) +
            7 * ShmemAllocator::default_alignment;

        dyn_shmem += 4 * detail::mask_num_bytes(this->m_max_edges_per_patch) +
                     4 * ShmemAllocator::default_alignment;

        dyn_shmem += 4 * detail::mask_num_bytes(this->m_max_faces_per_patch) +
                     4 * ShmemAllocator::default_alignment;

        dyn_shmem += PatchStash::stash_size * sizeof(uint32_t) +
                     ShmemAllocator::default_alignment;

        constexpr uint32_t block_size = 256;

        auto launch = [&](int add_item) {
            if (add_item == 0) {
                detail::slice_patches<block_size, TRANSPOSE_ITEM_PER_THREAD>
                    <<<grid_size, block_size, dyn_shmem>>>(
                        this->m_rxmesh_context,
                        get_num_patches(),
                        attributes...);
            } else if (add_item == 1) {
                detail::slice_patches<block_size, TRANSPOSE_ITEM_PER_THREAD + 1>
                    <<<grid_size, block_size, dyn_shmem>>>(
                        this->m_rxmesh_context,
                        get_num_patches(),
                        attributes...);
            } else if (add_item == 2) {
                detail::slice_patches<block_size, TRANSPOSE_ITEM_PER_THREAD + 2>
                    <<<grid_size, block_size, dyn_shmem>>>(
                        this->m_rxmesh_context,
                        get_num_patches(),
                        attributes...);
            } else if (add_item == 3) {
                detail::slice_patches<block_size, TRANSPOSE_ITEM_PER_THREAD + 3>
                    <<<grid_size, block_size, dyn_shmem>>>(
                        this->m_rxmesh_context,
                        get_num_patches(),
                        attributes...);
            } else if (add_item == 4) {
                detail::slice_patches<block_size, TRANSPOSE_ITEM_PER_THREAD + 4>
                    <<<grid_size, block_size, dyn_shmem>>>(
                        this->m_rxmesh_context,
                        get_num_patches(),
                        attributes...);
            } else if (add_item == 5) {
                detail::slice_patches<block_size, TRANSPOSE_ITEM_PER_THREAD + 5>
                    <<<grid_size, block_size, dyn_shmem>>>(
                        this->m_rxmesh_context,
                        get_num_patches(),
                        attributes...);
            } else {
                RXMESH_ERROR(
                    "RXMeshDynamic::slice_patches() can not find good "
                    "configuration to  run slice_patches kernel");
            }
        };


        auto check = [&](int add_item) {
            size_t   smem_bytes_static;
            uint32_t num_reg_per_thread;
            size_t   local_mem_per_thread;

            if (add_item == 0) {
                check_shared_memory(
                    dyn_shmem,
                    smem_bytes_static,
                    num_reg_per_thread,
                    local_mem_per_thread,
                    block_size,
                    (void*)detail::slice_patches<block_size,
                                                 TRANSPOSE_ITEM_PER_THREAD + 0>,
                    false);
            } else if (add_item == 1) {
                check_shared_memory(
                    dyn_shmem,
                    smem_bytes_static,
                    num_reg_per_thread,
                    local_mem_per_thread,
                    block_size,
                    (void*)detail::slice_patches<block_size,
                                                 TRANSPOSE_ITEM_PER_THREAD + 1>,
                    false);
            } else if (add_item == 2) {
                check_shared_memory(
                    dyn_shmem,
                    smem_bytes_static,
                    num_reg_per_thread,
                    local_mem_per_thread,
                    block_size,
                    (void*)detail::slice_patches<block_size,
                                                 TRANSPOSE_ITEM_PER_THREAD + 2>,
                    false);
            } else if (add_item == 3) {
                check_shared_memory(
                    dyn_shmem,
                    smem_bytes_static,
                    num_reg_per_thread,
                    local_mem_per_thread,
                    block_size,
                    (void*)detail::slice_patches<block_size,
                                                 TRANSPOSE_ITEM_PER_THREAD + 3>,
                    false);
            } else if (add_item == 4) {
                check_shared_memory(
                    dyn_shmem,
                    smem_bytes_static,
                    num_reg_per_thread,
                    local_mem_per_thread,
                    block_size,
                    (void*)detail::slice_patches<block_size,
                                                 TRANSPOSE_ITEM_PER_THREAD + 4>,
                    false);
            } else if (add_item == 5) {
                check_shared_memory(
                    dyn_shmem,
                    smem_bytes_static,
                    num_reg_per_thread,
                    local_mem_per_thread,
                    block_size,
                    (void*)detail::slice_patches<block_size,
                                                 TRANSPOSE_ITEM_PER_THREAD + 5>,
                    false);
            } else {
                RXMESH_ERROR(
                    "RXMeshDynamic::slice_patches() can not find good "
                    "configuration to run slice_patches kernel");
            }
        };

        for (uint32_t it = 0; it < 6; ++it) {
            if (2 * this->m_max_edges_per_patch <=
                block_size * (TRANSPOSE_ITEM_PER_THREAD + it)) {
                check(it);
                launch(it);
                break;
            }
        }
        CUDA_ERROR(cudaGetLastError());
    }


    /**
     * @brief update the host side. Use this function to update the host side
     * after performing (dynamic) updates on the GPU. This function may
     * re-allocates the host side memory buffers in case it is not enough (e.g.,
     * after performing mesh refinement on the GPU)
     */
    void update_host();

    /**
     * @brief update polyscope after performing dynamic changes. This function
     * is supposed to be called after a call to update_host since polyscope
     * reads information from the host side of RXMesh which include the topology
     * (stored in RXMesh/RXMeshStatic/RXMeshDynamic) and the input vertex
     * coordinates as well. Thus, a call to `move(DEVICE, HOST)` should be done
     * to RXMesh-stored vertex coordinates before calling this function.
     */
    void update_polyscope(std::string new_name = "");
};
}  // namespace rxmesh