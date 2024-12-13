#include <numeric>

#include <cooperative_groups.h>

#include "rxmesh/bitmask.cuh"
#include "rxmesh/kernels/dynamic_util.cuh"
#include "rxmesh/kernels/for_each.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/shmem_allocator.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace rxmesh {

namespace detail {
template <uint32_t blockThreads, typename HandleT>
__device__ __inline__ void hashtable_calibration(const Context context,
                                                 PatchInfo&    pi,
                                                 ShmemMutex& patch_stash_mutex)
{
    // TODO load the hashtable in shared memory
    using LocalT = typename HandleT::LocalT;

    const uint16_t num_elements = *(pi.get_num_elements<HandleT>());

    const uint16_t num_elements_up =
        ROUND_UP_TO_NEXT_MULTIPLE(num_elements, blockThreads);

    auto check_child = [&context](uint32_t&     owner,
                                  const LPPair& lp,
                                  bool&         is_del,
                                  bool&         replace) {
        if (is_del) {
            // maybe this owner was sliced and so let's check its
            // child
            uint32_t owner_child_id = context.m_patches_info[owner].child_id;
            if (owner_child_id != INVALID32) {
                is_del = context.m_patches_info[owner_child_id].is_deleted(
                    LocalT(lp.local_id_in_owner_patch()));
                if (!is_del) {
                    // if it is not deleted, then we have found the
                    // right entry
                    replace = true;
                    owner   = owner_child_id;
                }
            }
        }
    };

    for (uint16_t i = threadIdx.x; i < num_elements_up; i += blockThreads) {
        HandleT handle;
        bool    replace = false;

        if (i < num_elements) {

            if (!pi.is_owned(LocalT(i)) && !pi.is_deleted(LocalT(i))) {

                // This is the same implementation in
                // Context::get_owner_handle()

                uint32_t owner = pi.patch_id;
                uint16_t lid   = i;

                LPPair lp = pi.get_lp<HandleT>().find(lid, nullptr, nullptr);

                // if (lp.is_sentinel()) {
                //     printf("\n ## B=%u, T= %u, patch_id = %u, i= %u",
                //            blockIdx.x,
                //            threadIdx.x,
                //            pi.patch_id,
                //            i);
                // }
                assert(!lp.is_sentinel());

                owner = pi.patch_stash.get_patch(lp);

                assert(owner != INVALID32);

                // This only happen when the element i resides in the cavity of
                // the owner where it will be cleaned up later in
                // remove_surplus_elements
                bool is_del = context.m_patches_info[owner].is_deleted(
                    LocalT(lp.local_id_in_owner_patch()));

                check_child(owner, lp, is_del, replace);

                if (!is_del) {
                    while (!context.m_patches_info[owner].is_owned(
                        LocalT(lp.local_id_in_owner_patch()))) {

                        replace = true;

                        lp = context.m_patches_info[owner]
                                 .get_lp<HandleT>()
                                 .find(lp.local_id_in_owner_patch(),
                                       nullptr,
                                       nullptr);

                        assert(!lp.is_sentinel());

                        owner =
                            context.m_patches_info[owner].patch_stash.get_patch(
                                lp);

                        is_del = context.m_patches_info[owner].is_deleted(
                            LocalT(lp.local_id_in_owner_patch()));

                        check_child(owner, lp, is_del, replace);

                        if (is_del) {
                            replace = false;
                            break;
                        }
                    }
                    handle = HandleT(owner, lp.local_id_in_owner_patch());
                }
            }
        }

        __syncthreads();

        if (replace) {

            uint8_t o = pi.patch_stash.insert_patch(handle.patch_id(),
                                                    patch_stash_mutex);
            assert(o != INVALID8);
            LPPair lp(i, handle.local_id(), o);

            pi.get_lp<HandleT>().replace(lp);
        }
    }
}

template <uint32_t blockThreads>
__global__ static void hashtable_calibration(const Context context)
{
    const uint32_t pid = blockIdx.x;
    if (pid >= context.m_num_patches[0]) {
        return;
    }

    ShmemMutex patch_stash_mutex;
    patch_stash_mutex.alloc();
    PatchInfo pi = context.m_patches_info[pid];

    hashtable_calibration<blockThreads, VertexHandle>(
        context, pi, patch_stash_mutex);
    hashtable_calibration<blockThreads, EdgeHandle>(
        context, pi, patch_stash_mutex);
    hashtable_calibration<blockThreads, FaceHandle>(
        context, pi, patch_stash_mutex);
}

template <uint32_t blockThreads>
__device__ __inline__ void tag_edges_and_vertices_through_face(
    const uint16_t  num_vertices,
    const uint16_t  num_edges,
    const uint16_t  num_faces,
    const uint16_t* s_ev,
    const uint16_t* s_fe,
    const Bitmask&  s_active_f,
    const Bitmask&  s_active_e,
    const Bitmask&  s_active_v,
    const Bitmask&  s_owned_f,
    Bitmask&        s_vert_tag,
    Bitmask&        s_edge_tag,
    const Bitmask&  s_face_tag)
{
    // tag edges and vertices that are incident to owned/tagged faces
    for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        if (s_active_f(f) && (s_owned_f(f) || s_face_tag(f))) {
            for (int i = 0; i < 3; ++i) {
                const uint16_t e = s_fe[3 * f + i] >> 1;
                assert(e < num_edges);
                assert(s_active_e(e));

                const uint16_t v0 = s_ev[2 * e + 0];
                assert(v0 < num_vertices);
                assert(s_active_v(v0));

                const uint16_t v1 = s_ev[2 * e + 1];
                assert(v1 < num_vertices);
                assert(s_active_v(v1));

                s_edge_tag.set(e, true);
                s_vert_tag.set(v0, true);
                s_vert_tag.set(v1, true);
            }
        }
    }
};


template <uint32_t blockThreads>
__device__ __inline__ void tag_faces_through_edges_and_vertices(
    const uint16_t  num_vertices,
    const uint16_t  num_edges,
    const uint16_t  num_faces,
    const uint16_t* s_ev,
    const uint16_t* s_fe,
    const Bitmask&  s_active_v,
    const Bitmask&  s_active_e,
    const Bitmask&  s_active_f,
    const Bitmask&  s_owned_v,
    const Bitmask&  s_owned_e,
    const Bitmask&  s_vert_tag,
    const Bitmask&  s_edge_tag,
    Bitmask&        s_face_tag)
{
    // tag a face if one of its edges or vertices are either tagged or owned
    for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        if (s_active_f(f)) {

            for (int i = 0; i < 3; ++i) {
                const uint16_t e = s_fe[3 * f + i] >> 1;
                assert(e < num_edges);
                assert(s_active_e(e));

                const uint16_t v0 = s_ev[2 * e + 0];
                assert(v0 < num_vertices);
                assert(s_active_v(v0));

                const uint16_t v1 = s_ev[2 * e + 1];
                assert(v1 < num_vertices);
                assert(s_active_v(v1));

                if (s_edge_tag(e) || s_vert_tag(v0) || s_vert_tag(v1) ||
                    s_owned_e(e) || s_owned_v(v0) || s_owned_v(v1)) {
                    s_face_tag.set(f, true);
                    break;
                }
            }
        }
    }
}

template <uint32_t blockThreads>
__device__ __inline__ void tag_vertices_through_edges(
    const uint16_t  num_vertices,
    const uint16_t  num_edges,
    const uint16_t* s_ev,
    const Bitmask&  s_active_v,
    const Bitmask&  s_active_e,
    const Bitmask&  s_owned_e,
    Bitmask&        s_vert_tag,
    Bitmask&        s_edge_tag)
{
    // tag a vertex if it incident to an edge that is tagged or owned
    // if the edge is owned and it tagged a vertex, we also tag the edge as well
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        if (s_active_e(e) && (s_owned_e(e) || s_edge_tag(e))) {

            const uint16_t v0 = s_ev[2 * e + 0];
            assert(v0 < num_vertices);
            assert(s_active_v(v0));

            const uint16_t v1 = s_ev[2 * e + 1];
            assert(v1 < num_vertices);
            assert(s_active_v(v1));

            s_vert_tag.set(v0, true);
            s_vert_tag.set(v1, true);
            s_edge_tag.set(e, true);
        }
    }
}

template <uint32_t blockThreads>
__device__ __inline__ void reevaluate_active_elements(
    cooperative_groups::thread_block& block,
    const uint16_t                    num_vertices,
    const uint16_t                    num_edges,
    const uint16_t                    num_faces,
    const uint16_t*                   s_ev,
    const uint16_t*                   s_fe,
    const Bitmask&                    s_active_f,
    const Bitmask&                    s_active_e,
    const Bitmask&                    s_active_v,
    const Bitmask&                    s_owned_v,
    const Bitmask&                    s_owned_e,
    const Bitmask&                    s_owned_f,
    Bitmask&                          s_vert_tag,
    Bitmask&                          s_edge_tag,
    Bitmask&                          s_face_tag)
{
    // tage edges and vertices through faces
    tag_edges_and_vertices_through_face<blockThreads>(num_vertices,
                                                      num_edges,
                                                      num_faces,
                                                      s_ev,
                                                      s_fe,
                                                      s_active_f,
                                                      s_active_e,
                                                      s_active_v,
                                                      s_owned_f,
                                                      s_vert_tag,
                                                      s_edge_tag,
                                                      s_face_tag);
    block.sync();

    // tag faces through edges
    tag_faces_through_edges_and_vertices<blockThreads>(num_vertices,
                                                       num_edges,
                                                       num_faces,
                                                       s_ev,
                                                       s_fe,
                                                       s_active_v,
                                                       s_active_e,
                                                       s_active_f,
                                                       s_owned_v,
                                                       s_owned_e,
                                                       s_vert_tag,
                                                       s_edge_tag,
                                                       s_face_tag);
    block.sync();

    // tage edges and vertices through faces
    tag_edges_and_vertices_through_face<blockThreads>(num_vertices,
                                                      num_edges,
                                                      num_faces,
                                                      s_ev,
                                                      s_fe,
                                                      s_active_f,
                                                      s_active_e,
                                                      s_active_v,
                                                      s_owned_f,
                                                      s_vert_tag,
                                                      s_edge_tag,
                                                      s_face_tag);
    block.sync();

    // tag vertices through edges
    // and tag edges if there are owned
    tag_vertices_through_edges<blockThreads>(num_vertices,
                                             num_edges,
                                             s_ev,
                                             s_active_v,
                                             s_active_e,
                                             s_owned_e,
                                             s_vert_tag,
                                             s_edge_tag);
}


template <uint32_t blockThreads>
__inline__ __device__ void remove_idle_elements(
    cooperative_groups::thread_block& block,
    LPPair*                           s_table,
    uint32_t*                         s_patch_stash,
    LPHashTable&                      table,
    const PatchStash&                 patch_stash,
    const uint16_t                    num_elements,
    const Bitmask&                    is_owned,
    const Bitmask&                    is_active)

{
    // mesh elements in the patch that are deleted/inactive but not removed from
    // the hashtable (i.e., replaced by sentinel_pair) prevent inserting in
    // their place and need to be removed.
    //
    // remove idle elements from the hashtable by query the table from
    // global memory, write the results to shared memory buffer, then
    // copy the shared memory buffer to global memory

    __shared__ LPPair s_stash[LPHashTable::stash_size];

    fill_n<blockThreads>(s_stash, uint16_t(LPHashTable::stash_size), LPPair());
    fill_n<blockThreads>(s_table, table.get_capacity(), LPPair());
    block.sync();

    for (uint16_t e = threadIdx.x; e < num_elements; e += blockThreads) {
        if (is_active(e) && !is_owned(e)) {

            uint32_t bucket_id;
            bool     in_stash;
            LPPair pair = table.find(e, bucket_id, in_stash, nullptr, nullptr);
            assert(pair.patch_stash_id() < PatchStash::stash_size);

            // this will show up as race condition but even when multiple
            // threads try to write at the same location, they write the same
            // value
            s_patch_stash[pair.patch_stash_id()] =
                patch_stash.m_stash[pair.patch_stash_id()];
            assert(!pair.is_sentinel());
            if (in_stash) {
                s_stash[bucket_id] = pair;
            } else {
                s_table[bucket_id] = pair;
            }
        }
    }
    block.sync();

    for (uint16_t e = threadIdx.x; e < table.get_capacity();
         e += blockThreads) {
        table.m_table[e].m_pair = s_table[e].m_pair;
    }
    for (uint16_t e = threadIdx.x; e < LPHashTable::stash_size;
         e += blockThreads) {
        table.m_stash[e].m_pair = s_stash[e].m_pair;
    }
    block.sync();

#ifndef NDEBUG
    for (uint16_t e = threadIdx.x; e < num_elements; e += blockThreads) {
        if (is_active(e) && !is_owned(e)) {
            LPPair pair = table.find(e, nullptr, nullptr);
            assert(!pair.is_sentinel());
        }
    }
#endif
}
template <uint32_t blockThreads>
__global__ static void remove_surplus_elements(Context context)
{
    auto block = cooperative_groups::this_thread_block();

    const uint32_t pid = blockIdx.x;
    if (pid >= context.m_num_patches[0]) {
        return;
    }

    PatchInfo pi = context.m_patches_info[pid];

    context.m_patches_info[pid].child_id = INVALID32;

    const uint16_t num_vertices = pi.num_vertices[0];
    const uint16_t num_edges    = pi.num_edges[0];
    const uint16_t num_faces    = pi.num_faces[0];

    __shared__ uint32_t s_num_owned_vertices;
    __shared__ uint32_t s_num_owned_edges;
    __shared__ uint32_t s_num_owned_faces;

    __shared__ uint32_t s_num_vertices;
    __shared__ uint32_t s_num_edges;
    __shared__ uint32_t s_num_faces;
    if (threadIdx.x == 0) {
        s_num_owned_vertices = 0;
        s_num_owned_edges    = 0;
        s_num_owned_faces    = 0;
        s_num_vertices       = 0;
        s_num_edges          = 0;
        s_num_faces          = 0;
    }

    ShmemAllocator shrd_alloc;

    Bitmask s_owned_v = Bitmask(num_vertices, shrd_alloc);
    s_owned_v.load_async(block, pi.owned_mask_v);

    Bitmask s_owned_e = Bitmask(num_edges, shrd_alloc);
    s_owned_e.load_async(block, pi.owned_mask_e);

    Bitmask s_owned_f = Bitmask(num_faces, shrd_alloc);
    s_owned_f.load_async(block, pi.owned_mask_f);

    Bitmask s_active_v = Bitmask(num_vertices, shrd_alloc);
    s_active_v.load_async(block, pi.active_mask_v);

    Bitmask s_active_e = Bitmask(num_edges, shrd_alloc);
    s_active_e.load_async(block, pi.active_mask_e);

    Bitmask s_active_f = Bitmask(num_faces, shrd_alloc);
    s_active_f.load_async(block, pi.active_mask_f, true);


    // indicate if an edge is incident to an owned face
    Bitmask s_edge_tag = Bitmask(num_edges, shrd_alloc);
    s_edge_tag.reset(block);

    Bitmask s_vert_tag = Bitmask(num_vertices, shrd_alloc);
    s_vert_tag.reset(block);

    Bitmask s_face_tag = Bitmask(num_faces, shrd_alloc);
    s_face_tag.reset(block);

    uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(3 * num_faces);
    load_async(
        block, reinterpret_cast<uint16_t*>(pi.fe), 3 * num_faces, s_fe, false);

    uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(2 * num_edges);
    load_async(
        block, reinterpret_cast<uint16_t*>(pi.ev), 2 * num_edges, s_ev, true);

    block.sync();

    reevaluate_active_elements<blockThreads>(block,
                                             num_vertices,
                                             num_edges,
                                             num_faces,
                                             s_ev,
                                             s_fe,
                                             s_active_f,
                                             s_active_e,
                                             s_active_v,
                                             s_owned_v,
                                             s_owned_e,
                                             s_owned_f,
                                             s_vert_tag,
                                             s_edge_tag,
                                             s_face_tag);

    block.sync();
    s_vert_tag.store<blockThreads>(pi.active_mask_v);
    s_edge_tag.store<blockThreads>(pi.active_mask_e);
    s_face_tag.store<blockThreads>(pi.active_mask_f);


    // dealloc connectivity shared memory so we could use this space for loading
    // the hashtable
    shrd_alloc.dealloc<uint16_t>(3 * num_faces);
    shrd_alloc.dealloc<uint16_t>(2 * num_edges);

    uint16_t max_hash_table_cap =
        std::max(pi.lp_v.get_capacity(), pi.lp_e.get_capacity());
    max_hash_table_cap = std::max(max_hash_table_cap, pi.lp_f.get_capacity());

    LPPair* s_table = shrd_alloc.alloc<LPPair>(max_hash_table_cap);

    __shared__ uint32_t s_patch_stash[PatchStash::stash_size];
    fill_n<blockThreads>(
        s_patch_stash, uint16_t(LPHashTable::stash_size), INVALID32);

    block.sync();
    remove_idle_elements<blockThreads>(block,
                                       s_table,
                                       s_patch_stash,
                                       pi.get_lp<VertexHandle>(),
                                       pi.patch_stash,
                                       num_vertices,
                                       s_owned_v,
                                       s_vert_tag);
    block.sync();

    remove_idle_elements<blockThreads>(block,
                                       s_table,
                                       s_patch_stash,
                                       pi.get_lp<EdgeHandle>(),
                                       pi.patch_stash,
                                       num_edges,
                                       s_owned_e,
                                       s_edge_tag);
    block.sync();

    remove_idle_elements<blockThreads>(block,
                                       s_table,
                                       s_patch_stash,
                                       pi.get_lp<FaceHandle>(),
                                       pi.patch_stash,
                                       num_faces,
                                       s_owned_f,
                                       s_face_tag);
    block.sync();
    store<blockThreads>(s_patch_stash,
                        uint16_t(PatchStash::stash_size),
                        pi.patch_stash.m_stash);

    for (uint32_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        if (s_vert_tag(v) && s_owned_v(v)) {
            ::atomicAdd(&s_num_owned_vertices, uint32_t(1));
        }
        if (s_vert_tag(v)) {
            ::atomicMax(&s_num_vertices, v + 1);
        }
    }

    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        if (s_edge_tag(e) && s_owned_e(e)) {
            ::atomicAdd(&s_num_owned_edges, uint32_t(1));
        }
        if (s_edge_tag(e)) {
            ::atomicMax(&s_num_edges, e + 1);
        }
    }

    for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        if (s_face_tag(f) && s_owned_f(f)) {
            ::atomicAdd(&s_num_owned_faces, uint32_t(1));
        }
        if (s_face_tag(f)) {
            ::atomicMax(&s_num_faces, f + 1);
        }
    }

    block.sync();

    if (threadIdx.x == 0) {
        ::atomicAdd(context.m_num_vertices, s_num_owned_vertices);
        ::atomicAdd(context.m_num_edges, s_num_owned_edges);
        ::atomicAdd(context.m_num_faces, s_num_owned_faces);

        pi.num_vertices[0] = s_num_vertices;
        pi.num_edges[0]    = s_num_edges;
        pi.num_faces[0]    = s_num_faces;

        ::atomicMax(context.m_max_num_vertices, s_num_vertices);
        ::atomicMax(context.m_max_num_edges, s_num_edges);
        ::atomicMax(context.m_max_num_faces, s_num_faces);
    }

    pi.clear_dirty();
}

template <uint32_t blockThreads, typename HandleT>
__inline__ __device__ void copy_to_hashtable(
    const PatchInfo& pi,
    PatchInfo&       new_pi,
    const uint16_t   num_elements,
    const Bitmask&   s_new_p_active,
    const Bitmask&   s_new_p_owned,
    const Bitmask&   s_active,
    const Bitmask&   s_owned,
    PatchStash&      new_patch_stash,
    // PatchStash&      original_patch_stash,
    ShmemMutex& new_patch_stash_mutex)
{

    for (uint16_t v = threadIdx.x; v < num_elements; v += blockThreads) {
        // if the element is active but not owned in the new patch
        /* HandleT vh;
        if (s_active(v) && !s_owned(v) && !s_new_p_active(v)) {
            // vh = pi.find<HandleT>(v);

            uint32_t bucket_id;

            bool in_stash;

            LPPair pair = pi.get_lp<HandleT>().find(
                v, bucket_id, in_stash, nullptr, nullptr);

            assert(pair.patch_stash_id() < PatchStash::stash_size);

            vh = HandleT(pi.patch_stash.get_patch(pair),
                         {pair.local_id_in_owner_patch()});

            original_patch_stash.m_stash[pair.patch_stash_id()] = vh.patch_id();
        }*/

        if (s_new_p_active(v) && !s_new_p_owned(v)) {
            LPPair lp;
            // if the element is originally owned by the patch
            if (s_owned(v)) {
                lp = LPPair(v, v, 0);
            } else {
                /* if (!vh.is_valid()) {
                    vh = pi.find<HandleT>(v);
                }*/

                HandleT vh = pi.find<HandleT>(v);

                uint8_t st = new_patch_stash.insert_patch(
                    vh.patch_id(), new_patch_stash_mutex);

                assert(st != INVALID8);

                lp = LPPair(v, vh.local_id(), st);
            }

            new_pi.get_lp<HandleT>().insert(lp, nullptr, nullptr);
        }
    }
}

template <uint32_t blockThreads>
__inline__ __device__ void slice(Context&                          context,
                                 cooperative_groups::thread_block& block,
                                 PatchInfo&                        pi,
                                 const uint32_t                    new_patch_id,
                                 const uint16_t                    num_vertices,
                                 const uint16_t                    num_edges,
                                 const uint16_t                    num_faces,
                                 PatchStash& s_new_patch_stash,
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
                                 Bitmask&        s_new_p_owned_f)
{
    // when we move elements to the other patch, we keep their index the same
    // since the new patch has a size bigger than the existing one

    // do similar dance that we do in remove_surplus_elements in order to get
    // what elements to copy to the new patch
    // We start by considering active elements are only those that are owned
    // and then add more elements

    // filter not-owned element for s_new_p_owned_v/e/f since these masks were
    // generated during bi-assignment which considered all elements in the patch

    assert(s_owned_v.size() == num_vertices);
    assert(s_new_p_owned_v.size() == num_vertices);
    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        if (s_new_p_owned_v(v) && !s_owned_v(v)) {
            s_new_p_owned_v.reset(v, true);
        }
    }

    assert(s_new_p_owned_e.size() == num_edges);
    assert(s_owned_e.size() == num_edges);
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        if (s_new_p_owned_e(e) && !s_owned_e(e)) {
            s_new_p_owned_e.reset(e, true);
        }
    }

    assert(s_new_p_owned_f.size() == num_faces);
    assert(s_owned_f.size() == num_faces);
    for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        if (s_new_p_owned_f(f) && !s_owned_f(f)) {
            s_new_p_owned_f.reset(f, true);
        }
    }
    block.sync();

    for (uint16_t i = threadIdx.x; i < PatchStash::stash_size;
         i += blockThreads) {
        if (i == 0) {
            s_new_patch_stash.m_stash[i] = pi.patch_id;
        } else {
            s_new_patch_stash.m_stash[i] = INVALID32;
        }
        // s_original_patch_stash.m_stash[i] = INVALID32;
    }


    block.sync();

    ShmemMutex new_patch_stash_mutex;
    new_patch_stash_mutex.alloc();

    PatchInfo new_patch = context.m_patches_info[new_patch_id];

    // evluate active elements of the new patch
    reevaluate_active_elements<blockThreads>(block,
                                             num_vertices,
                                             num_edges,
                                             num_faces,
                                             s_ev,
                                             s_fe,
                                             s_active_f,
                                             s_active_e,
                                             s_active_v,
                                             s_new_p_owned_v,
                                             s_new_p_owned_e,
                                             s_new_p_owned_f,
                                             s_new_p_active_v,
                                             s_new_p_active_e,
                                             s_new_p_active_f);
    block.sync();

    // insert elements in the new patch hashtable
    copy_to_hashtable<blockThreads, VertexHandle>(pi,
                                                  new_patch,
                                                  num_vertices,
                                                  s_new_p_active_v,
                                                  s_new_p_owned_v,
                                                  s_active_v,
                                                  s_owned_v,
                                                  s_new_patch_stash,
                                                  // s_original_patch_stash,
                                                  new_patch_stash_mutex);

    copy_to_hashtable<blockThreads, EdgeHandle>(pi,
                                                new_patch,
                                                num_edges,
                                                s_new_p_active_e,
                                                s_new_p_owned_e,
                                                s_active_e,
                                                s_owned_e,
                                                s_new_patch_stash,
                                                // s_original_patch_stash,
                                                new_patch_stash_mutex);

    copy_to_hashtable<blockThreads, FaceHandle>(pi,
                                                new_patch,
                                                num_faces,
                                                s_new_p_active_f,
                                                s_new_p_owned_f,
                                                s_active_f,
                                                s_owned_f,
                                                s_new_patch_stash,
                                                // s_original_patch_stash,
                                                new_patch_stash_mutex);

    // store new patch to global memory
    if (threadIdx.x == 0) {
        new_patch.num_vertices[0] = num_vertices;
        new_patch.num_edges[0]    = num_edges;
        new_patch.num_faces[0]    = num_faces;

        context.m_patches_info[new_patch_id].patch_id = new_patch_id;
        context.m_patches_info[pi.patch_id].child_id  = new_patch_id;
    }
    // store active mask
    s_new_p_active_v.store<blockThreads>(new_patch.active_mask_v);
    s_new_p_active_e.store<blockThreads>(new_patch.active_mask_e);
    s_new_p_active_f.store<blockThreads>(new_patch.active_mask_f);

    // store owned mask
    s_new_p_owned_v.store<blockThreads>(new_patch.owned_mask_v);
    s_new_p_owned_e.store<blockThreads>(new_patch.owned_mask_e);
    s_new_p_owned_f.store<blockThreads>(new_patch.owned_mask_f);

    // ev and fe
    detail::store<blockThreads>(
        s_ev, 2 * num_edges, reinterpret_cast<uint16_t*>(new_patch.ev));
    detail::store<blockThreads>(
        s_fe, 3 * num_faces, reinterpret_cast<uint16_t*>(new_patch.fe));

    // patch stash
    block.sync();
    detail::store<blockThreads>(s_new_patch_stash.m_stash,
                                PatchStash::stash_size,
                                new_patch.patch_stash.m_stash);

    ////////////////////////////
    // now, we update this patch (that we sliced)
    block.sync();
    // reset s_new_p_active_v/e/f so we could recycle them
    s_new_p_active_v.reset(block);
    s_new_p_active_e.reset(block);
    s_new_p_active_f.reset(block);

    // if the element is active, owned by this patch and the new patch, then
    // we remove the ownership from this patch
    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        if ((s_active_v(v) && s_owned_v(v) && s_new_p_owned_v(v)) ||
            !s_active_v(v)) {
            s_owned_v.reset(v, true);
        }
    }
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        if ((s_active_e(e) && s_owned_e(e) && s_new_p_owned_e(e)) ||
            !s_active_e(e)) {
            s_owned_e.reset(e, true);
        }
    }
    for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        if ((s_active_f(f) && s_owned_f(f) && s_new_p_owned_f(f)) ||
            !s_active_f(f)) {
            s_owned_f.reset(f, true);
        }
    }
    block.sync();

    __shared__ uint8_t s_new_patch_stash_id;
    if (threadIdx.x == 0) {
        s_new_patch_stash_id = pi.patch_stash.insert_patch(new_patch_id);
        // s_new_patch_stash_id =
        //     s_original_patch_stash.insert_patch(new_patch_id);
        assert(s_new_patch_stash_id != INVALID8);
    }

    // evluate active elements of the original patch based on the update
    // element ownership. We reuse s_new_p_active_v/e/f to store this
    // updated active elements
    reevaluate_active_elements<blockThreads>(block,
                                             num_vertices,
                                             num_edges,
                                             num_faces,
                                             s_ev,
                                             s_fe,
                                             s_active_f,
                                             s_active_e,
                                             s_active_v,
                                             s_owned_v,
                                             s_owned_e,
                                             s_owned_f,
                                             s_new_p_active_v,
                                             s_new_p_active_e,
                                             s_new_p_active_f);
    block.sync();

    // store original patch's filtered patch stash

    // detail::store<blockThreads>(s_original_patch_stash.m_stash,
    //                             PatchStash::stash_size,
    //                             pi.patch_stash.m_stash);


    // store owned mask
    s_owned_v.store<blockThreads>(pi.owned_mask_v);
    s_owned_e.store<blockThreads>(pi.owned_mask_e);
    s_owned_f.store<blockThreads>(pi.owned_mask_f);

    // store active mask
    s_new_p_active_v.store<blockThreads>(pi.active_mask_v);
    s_new_p_active_e.store<blockThreads>(pi.active_mask_e);
    s_new_p_active_f.store<blockThreads>(pi.active_mask_f);


    // add to pi the new ribbon elements due to slicing the patch
    // these elements are thoese that are (still) active, not-owned but
    // owned by the new patch
    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        if (s_new_p_active_v(v) && !s_owned_v(v) && s_new_p_owned_v(v)) {
            assert(!new_patch.is_deleted(LocalVertexT(v)));
            LPPair lp(v, v, s_new_patch_stash_id);
            pi.lp_v.insert(lp, nullptr, nullptr);
        }
    }

    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        if (s_new_p_active_e(e) && !s_owned_e(e) && s_new_p_owned_e(e)) {
            assert(!new_patch.template is_deleted(LocalEdgeT(e)));
            LPPair lp(e, e, s_new_patch_stash_id);
            pi.lp_e.insert(lp, nullptr, nullptr);
        }
    }

    for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        if (s_new_p_active_f(f) && !s_owned_f(f) && s_new_p_owned_f(f)) {
            assert(!new_patch.template is_deleted(LocalFaceT(f)));
            LPPair lp(f, f, s_new_patch_stash_id);
            pi.lp_f.insert(lp, nullptr, nullptr);
        }
    }
}

template <uint32_t blockThreads>
__inline__ __device__ void bi_assignment(
    cooperative_groups::thread_block& block,
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
    Bitmask&                          s_new_p_owned_f)
{
    // assign mesh element to two partitions. the assignment partition the patch
    // into contiguous patches of almost equal size
    //
    // an element stay in the patch if its bitmask is zero

    // initially, all mesh elements stays in this patch
    s_new_p_owned_v.reset(block);
    s_new_p_owned_e.reset(block);
    s_new_p_owned_f.reset(block);
    block.sync();

    // number of faces that are assigned to 1
    __shared__ uint16_t s_num_1_faces;
    if (threadIdx.x == 0) {
        // we bootstrap the assignment by assigning a ribbon face to 1
        s_num_1_faces = 0;
        for (uint16_t f = 0; f < num_faces; ++f) {
            if (s_active_f(f) && !s_owned_f(f)) {
                s_new_p_owned_f.set(f);
                s_num_1_faces++;
                break;
            }
        }

        if (s_num_1_faces == 0) {
            // in case it is isolated patch without ribbon, then any face active
            // would work
            for (uint16_t f = 0; f < num_faces; ++f) {
                if (s_active_f(f)) {
                    s_new_p_owned_f.set(f);
                    s_num_1_faces++;
                    break;
                }
            }
        }
    }
    block.sync();

    // TODO use number of owned faces and not-owned in the new patch to guide
    //  its growth
    //  we iterate over faces twice. First, every face atomically set its
    //  three vertices if the face is set. Second, every face set itself if
    //  there are one vertex incident to it that is set. we stop when the
    //  s_num_1_faces is more than half num_faces
    uint32_t iter = 0;
    while (true) {
        iter++;
        // 1st
        for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
            if (s_active_f(f) && s_new_p_owned_f(f)) {
                const uint16_t v0(s_fv[3 * f + 0]), v1(s_fv[3 * f + 1]),
                    v2(s_fv[3 * f + 2]);
                assert(v0 < num_vertices);
                assert(v1 < num_vertices);
                assert(v2 < num_vertices);
                s_new_p_owned_v.set(v0, true);
                s_new_p_owned_v.set(v1, true);
                s_new_p_owned_v.set(v2, true);
            }
        }

        if (s_num_1_faces > num_faces / 2 || iter > 20) {
            break;
        }
        block.sync();

        // 2nd
        for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
            if (s_active_f(f) && !s_new_p_owned_f(f)) {
                const uint16_t v0(s_fv[3 * f + 0]), v1(s_fv[3 * f + 1]),
                    v2(s_fv[3 * f + 2]);
                assert(v0 < num_vertices);
                assert(v1 < num_vertices);
                assert(v2 < num_vertices);
                if (s_new_p_owned_v(v0) || s_new_p_owned_v(v1) ||
                    s_new_p_owned_v(v2)) {
                    s_new_p_owned_f.set(f, true);
                    atomicAdd(&s_num_1_faces, 1);
                }
            }
        }
        block.sync();
    }


    // finally we assign edges
    /*for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        if (s_active_e(e)) {
            const uint16_t v0(s_ev[2 * e + 0]), v1(s_ev[2 * e + 1]);
            assert(v0 < num_vertices);
            assert(v1 < num_vertices);
            if (s_new_p_owned_v(v0) && s_new_p_owned_v(v1)) {
                s_new_p_owned_e.set(e, true);
            }
        }
    }*/
}


template <uint32_t blockThreads>
__inline__ __device__ void bi_assignment_ggp(
    cooperative_groups::thread_block& block,
    const uint16_t                    num_vertices,
    const Bitmask&                    s_owned_v,
    const bool                        ignore_owned_v,
    const Bitmask&                    s_active_v,
    const uint16_t*                   s_vv_offset,
    const uint16_t*                   s_vv,
    Bitmask&                          s_assigned_v,
    Bitmask&                          s_current_frontier_v,
    Bitmask&                          s_next_frontier_v,
    Bitmask&                          s_partition_a_v,
    Bitmask&                          s_partition_b_v,
    int                               num_iter)
{
    __shared__ int      s_num_assigned_vertices;
    __shared__ int      s_num_active_vertices;
    __shared__ int      s_num_A_vertices, s_num_B_vertices;
    __shared__ uint16_t s_seed_a, s_seed_b;

    // compute the total number of active vertices (including not-owned)
    auto compute_num_active_vertices = [&]() {
        if (threadIdx.x == 0) {
            s_num_active_vertices = 0;
        }
        block.sync();


        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
            if (s_active_v(v)) {
                ::atomicAdd(&s_num_active_vertices, 1);
            }
        }
    };

    // pick two vertices and set them as the seeds
    // we do this by picking two vertices that are on the ribbon and far
    // a part. If we did not find any (because the patch is isolated) we
    // pick any two active vertices
    auto bootstrap = [&]() {
        s_num_assigned_vertices = 0;

        bool found_a(false), found_b(false);
        for (uint16_t v = 0; v < num_vertices; ++v) {
            if (s_active_v(v) /* && !s_owned_v(v)*/) {
                if (!ignore_owned_v && s_owned_v(v)) {
                    continue;
                }
                s_seed_a = v;
                found_a  = true;
                break;
            }
        }
        if (!found_a) {
            for (uint16_t v = 0; v < num_vertices; ++v) {
                if (s_active_v(v)) {
                    s_seed_a = v;
                    found_a  = true;
                    break;
                }
            }
        }
        assert(found_a);

        for (uint16_t v = num_vertices - 1; v > 1; --v) {
            if (s_active_v(v) /*&& !s_owned_v(v)*/ && v != s_seed_a) {
                if (!ignore_owned_v && s_owned_v(v)) {
                    continue;
                }
                s_seed_b = v;
                found_b  = true;
                break;
            }
        }

        if (!found_b) {
            for (uint16_t v = num_vertices - 1; v > 1; --v) {
                if (s_active_v(v) && v != s_seed_a) {
                    s_seed_b = v;
                    found_b  = true;
                    break;
                }
            }
        }
        assert(found_b);
    };


    // init region growing by populating the current frontier with the seeds
    auto init_region_growing = [&]() {
        assert(s_seed_a != s_seed_b);

        s_partition_a_v.reset(block);
        s_partition_b_v.reset(block);
        s_assigned_v.reset(block);
        s_current_frontier_v.reset(block);
        s_next_frontier_v.reset(block);
        block.sync();

        if (threadIdx.x == 0) {
            s_assigned_v.set(s_seed_a);
            s_partition_a_v.set(s_seed_a);
            s_current_frontier_v.set(s_seed_a);
            s_num_A_vertices = 1;

            s_assigned_v.set(s_seed_b);
            s_partition_b_v.set(s_seed_b);
            s_current_frontier_v.set(s_seed_b);
            s_num_B_vertices = 1;

            s_num_assigned_vertices = 2;
        }
    };


    // region growing. Starting with two seed vertices in each partition, we tag
    // other vertices such that vertices are tagged to the closer seed. We do
    // this as a scatter operation using VV
    auto region_growing = [&]() {
        int num_assigned_prv_iter = 0;
        while (s_num_assigned_vertices < s_num_active_vertices) {

            block.sync();

            for (uint16_t v = threadIdx.x; v < num_vertices;
                 v += blockThreads) {

                if (s_current_frontier_v(v)) {

                    assert(s_active_v(v));
                    assert(s_assigned_v(v));
                    assert(s_partition_a_v(v) || s_partition_b_v(v));

                    const uint16_t start = s_vv_offset[v];
                    const uint16_t end   = s_vv_offset[v + 1];

                    for (uint16_t vv = start; vv < end; ++vv) {

                        const uint16_t nv = s_vv[vv];

                        if (!s_active_v(nv)) {
                            continue;
                        }

                        if (!s_assigned_v(nv)) {
                            if (s_partition_a_v(v)) {
                                if (s_partition_a_v.try_set(nv)) {
                                    s_next_frontier_v.set(nv, true);
                                    assert(s_partition_a_v(nv));
                                    ::atomicAdd(&s_num_A_vertices, 1);
                                }
                            }
                            if (s_partition_b_v(v)) {
                                if (s_partition_b_v.try_set(nv)) {
                                    s_next_frontier_v.set(nv, true);
                                    assert(s_partition_b_v(nv));
                                    ::atomicAdd(&s_num_B_vertices, 1);
                                }
                            }
                        }
                    }
                } else if (s_active_v(v) && !s_assigned_v(v)) {
                    // for active vertices that are not connected to anything
                    // other vertices, we will add then to one of the partitions
                    // at random (kinda) and put the vertex on the frontier so
                    // we don't encounter this if condition again
                    const uint16_t start = s_vv_offset[v];
                    const uint16_t end   = s_vv_offset[v + 1];

                    // and we have to calculate the neighbors since we might be
                    // connected to in-active vertices
                    int num_neighbours = 0;

                    for (uint16_t vv = start; vv < end; ++vv) {

                        const uint16_t nv = s_vv[vv];

                        if (!s_active_v(nv)) {
                            continue;
                        }
                        num_neighbours++;
                    }
                    if (num_neighbours == 0) {
                        if (v % 2 == 0) {
                            s_partition_a_v.set(v, true);
                            ::atomicAdd(&s_num_A_vertices, 1);
                        } else {
                            s_partition_b_v.set(v, true);
                            ::atomicAdd(&s_num_B_vertices, 1);
                        }
                        s_next_frontier_v.set(v, true);
                        // printf("\n reg growing %u", v);
                    }
                }
            }

            block.sync();

            for (uint16_t v = threadIdx.x; v < num_vertices;
                 v += blockThreads) {
                if (s_next_frontier_v(v)) {
                    ::atomicAdd(&s_num_assigned_vertices, 1);
                    s_assigned_v.set(v, true);
                }
            }

            s_current_frontier_v.copy(block, s_next_frontier_v);
            block.sync();
            s_next_frontier_v.reset(block);

            if (s_num_assigned_vertices == num_assigned_prv_iter) {
                // means that we have not no made any progress in this iteration
                // probably because we have a disconnected patch
                break;
            }
            num_assigned_prv_iter = s_num_assigned_vertices;
        }

        if (s_num_assigned_vertices < s_num_active_vertices) {
            // fix for disconnected patches by assigning them to the small
            // partition
            bool is_a_bigger = s_num_A_vertices > s_num_B_vertices;
            block.sync();
            for (uint16_t v = threadIdx.x; v < num_vertices;
                 v += blockThreads) {
                if (s_active_v(v) && !s_assigned_v(v)) {
                    if (is_a_bigger) {
                        s_partition_b_v.set(v, true);
                        ::atomicAdd(&s_num_B_vertices, 1);
                    } else {
                        s_partition_a_v.set(v, true);
                        ::atomicAdd(&s_num_A_vertices, 1);
                    }
                    s_assigned_v.set(v, true);
                }
            }
        }
    };


    // vertices on the boundary might be assigned to both region, here we
    // restrict that every vertex to be assigned to one region and we make
    // this in a way that balance out both regions
    auto impose_constraints = [&]() {
        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
            if (s_partition_a_v(v) && s_partition_b_v(v)) {
                assert(s_active_v(v));
                if (atomic_read(&s_num_A_vertices) >
                    atomic_read(&s_num_B_vertices)) {
                    s_partition_a_v.reset(v, true);
                    ::atomicAdd(&s_num_A_vertices, -1);
                } else {
                    s_partition_b_v.reset(v, true);
                    ::atomicAdd(&s_num_B_vertices, -1);
                }
            }
        }
    };

    // inite the frontier for each region/partition. the frontier means vertices
    // on partition X that are connected to vertices in partition Y or a vertex
    // in the ribbon
    auto init_interior = [&]() {
        s_current_frontier_v.reset(block);
        s_assigned_v.reset(block);
        if (threadIdx.x == 0) {
            s_num_assigned_vertices = 0;
        }
        block.sync();

        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
            if (s_active_v(v)) {
                bool is_a(s_partition_a_v(v));
                // TODO: line frontier of separators, double check the logic
                bool on_frontier = (ignore_owned_v) ? false : !s_owned_v(v);

                if (!on_frontier) {
                    const uint16_t start = s_vv_offset[v];
                    const uint16_t end   = s_vv_offset[v + 1];


                    // check if the vertex is isolated
                    int num_neighbours = 0;

                    for (uint16_t vv = start; vv < end; ++vv) {
                        const uint16_t nv = s_vv[vv];
                        if (s_active_v(nv)) {
                            num_neighbours++;
                            if (is_a) {
                                if (s_partition_b_v(nv)) {
                                    on_frontier = true;
                                    break;
                                }
                            } else {
                                if (s_partition_a_v(nv)) {
                                    on_frontier = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (num_neighbours == 0) {
                        // if the vertex is isolated, then we mark it
                        // as assigned but we don't put it on the frontier
                        assert(!on_frontier);
                        s_assigned_v.set(v, true);
                        ::atomicAdd(&s_num_assigned_vertices, 1);
                    }
                }
                if (on_frontier) {
                    s_current_frontier_v.set(v, true);
                    s_assigned_v.set(v, true);
                    ::atomicAdd(&s_num_assigned_vertices, 1);
                    if (is_a) {
                        s_seed_a = v;
                    } else {
                        s_seed_b = v;
                    }
                }
            }
        }
    };


    // find the most interior vertex in each partition starting from the
    // frontier and moving to the interior. While we do this, we keep re-writing
    // s_seed_a and s_seed_b. In the last iteration in this while loop, (one of)
    // the most interior seed will be finally written
    auto compute_interior = [&]() {
        int num_assigned_prv_iter = 0;

        while (s_num_assigned_vertices < s_num_active_vertices) {
            block.sync();
            for (uint16_t v = threadIdx.x; v < num_vertices;
                 v += blockThreads) {

                if (s_current_frontier_v(v)) {

                    assert(s_assigned_v(v));
                    assert(s_partition_a_v(v) || s_partition_b_v(v));

                    bool is_a(s_partition_a_v(v));

                    const uint16_t start = s_vv_offset[v];
                    const uint16_t end   = s_vv_offset[v + 1];

                    for (uint16_t vv = start; vv < end; ++vv) {

                        const uint16_t nv = s_vv[vv];
                        if (!s_assigned_v(nv)) {
                            if (is_a) {
                                if (s_partition_a_v(nv)) {
                                    if (s_next_frontier_v.try_set(nv)) {
                                        ::atomicAdd(&s_num_assigned_vertices,
                                                    1);
                                        s_seed_a = nv;
                                    }
                                }
                            } else {
                                if (s_partition_b_v(nv)) {
                                    if (s_next_frontier_v.try_set(nv)) {
                                        ::atomicAdd(&s_num_assigned_vertices,
                                                    1);
                                        s_seed_b = nv;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            block.sync();

            for (uint16_t v = threadIdx.x; v < num_vertices;
                 v += blockThreads) {
                if (s_next_frontier_v(v)) {
                    s_assigned_v.set(v, true);
                }
            }

            s_current_frontier_v.copy(block, s_next_frontier_v);
            block.sync();
            s_next_frontier_v.reset(block);

            if (s_num_assigned_vertices == num_assigned_prv_iter) {
                // means that we have not no made any progress in this iteration
                // probably because we have a disconnected patch
                break;
            }
            num_assigned_prv_iter = s_num_assigned_vertices;
        }
    };

    if (threadIdx.x == 0) {
        bootstrap();
    }
    compute_num_active_vertices();
    block.sync();

    for (int it = 0; it < num_iter; ++it) {

        init_region_growing();
        block.sync();

        region_growing();
        block.sync();

        impose_constraints();
        block.sync();


        // if we have reached good balance, then terminate early
        float ratio = float(std::abs(s_num_A_vertices - s_num_B_vertices)) /
                      float(s_num_active_vertices);
        if (ratio < 0.1f) {
            break;
        }

        assert(s_num_A_vertices + s_num_B_vertices == s_num_active_vertices);

        init_interior();
        block.sync();

        if (s_num_assigned_vertices == 0) {
            // i.e., there is no frontier means we probably have a disconnected
            // input
            break;
        }
        compute_interior();
        block.sync();
    }

#ifndef NDEBUG
    // sanity check that an active vertex is either assigned to single partition
    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        if (s_active_v(v)) {
            if (s_partition_a_v(v)) {
                assert(!s_partition_b_v(v));
            }
            if (s_partition_b_v(v)) {
                assert(!s_partition_a_v(v));
            }
        }
    }
#endif
}


template <uint32_t blockThreads>
__global__ static void calc_num_elements(const Context context,
                                         uint32_t*     sum_num_vertices,
                                         uint32_t*     sum_num_edges,
                                         uint32_t*     sum_num_faces)
{
    auto sum_v = [&](VertexHandle& v_id) { ::atomicAdd(sum_num_vertices, 1u); };
    for_each<Op::V, blockThreads>(context, sum_v);


    auto sum_e = [&](EdgeHandle& e_id) { ::atomicAdd(sum_num_edges, 1u); };
    for_each<Op::E, blockThreads>(context, sum_e);


    auto sum_f = [&](FaceHandle& f_id) { ::atomicAdd(sum_num_faces, 1u); };
    for_each<Op::F, blockThreads>(context, sum_f);
}

template <uint32_t blockThreads>
__global__ static void check_uniqueness(const Context           context,
                                        unsigned long long int* d_check)
{
    auto block = cooperative_groups::this_thread_block();

    const uint32_t patch_id = blockIdx.x;

    if (patch_id < context.m_num_patches[0]) {

        PatchInfo patch_info = context.m_patches_info[patch_id];

        ShmemAllocator shrd_alloc;

        uint16_t* s_fe =
            shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces[0]);
        uint16_t* s_ev =
            shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges[0]);

        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * patch_info.num_edges[0],
                   s_ev,
                   false);

        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces[0],
                   s_fe,
                   true);
        block.sync();

        // make sure an edge is connecting two unique vertices
        for (uint16_t e = threadIdx.x; e < patch_info.num_edges[0];
             e += blockThreads) {

            const LocalEdgeT el(e);

            uint16_t v0 = s_ev[2 * e + 0];
            uint16_t v1 = s_ev[2 * e + 1];

            if (!patch_info.is_deleted(el)) {

                if (v0 >= patch_info.num_vertices[0] ||
                    v1 >= patch_info.num_vertices[0] || v0 == v1) {
                    // printf("\n 1 unqiuness = %u", patch_id);
                    ::atomicAdd(d_check, 1);
                }

                if (patch_info.is_deleted(LocalVertexT(v0)) ||
                    patch_info.is_deleted(LocalVertexT(v1))) {
                    // printf("\n 2 unqiuness = %u", patch_id);
                    ::atomicAdd(d_check, 1);
                }
            }
        }

        // make sure a face is formed by three unique edges and these edges
        // gives three unique vertices
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces[0];
             f += blockThreads) {

            const LocalFaceT fl(f);

            if (!patch_info.is_deleted(fl)) {
                uint16_t e0, e1, e2;
                flag_t   d0(0), d1(0), d2(0);
                Context::unpack_edge_dir(s_fe[3 * f + 0], e0, d0);
                Context::unpack_edge_dir(s_fe[3 * f + 1], e1, d1);
                Context::unpack_edge_dir(s_fe[3 * f + 2], e2, d2);

                if (e0 >= patch_info.num_edges[0] ||
                    e1 >= patch_info.num_edges[0] ||
                    e2 >= patch_info.num_edges[0] || e0 == e1 || e0 == e2 ||
                    e1 == e2) {
                    // printf("\n 3 unqiuness = %u", patch_id);
                    ::atomicAdd(d_check, 1);
                }

                if (patch_info.is_deleted(LocalEdgeT(e0)) ||
                    patch_info.is_deleted(LocalEdgeT(e1)) ||
                    patch_info.is_deleted(LocalEdgeT(e2))) {
                    // printf("\n 4 unqiuness = %u, f= %u", patch_id, f);
                    ::atomicAdd(d_check, 1);
                }

                uint16_t v0, v1, v2;
                v0 = s_ev[(2 * e0) + (1 * d0)];
                v1 = s_ev[(2 * e1) + (1 * d1)];
                v2 = s_ev[(2 * e2) + (1 * d2)];


                if (v0 >= patch_info.num_vertices[0] ||
                    v1 >= patch_info.num_vertices[0] ||
                    v2 >= patch_info.num_vertices[0] || v0 == v1 || v0 == v2 ||
                    v1 == v2) {
                    // printf("\n 5 unqiuness = %u", patch_id);
                    ::atomicAdd(d_check, 1);
                }

                if (patch_info.is_deleted(LocalVertexT(v0)) ||
                    patch_info.is_deleted(LocalVertexT(v1)) ||
                    patch_info.is_deleted(LocalVertexT(v2))) {
                    // printf("\n 6 unqiuness = %u, f=%u", patch_id, f);
                    ::atomicAdd(d_check, 1);
                }
            }
        }
    }
}


template <uint32_t blockThreads>
__global__ static void check_not_owned(const Context           context,
                                       unsigned long long int* d_check)
{
    auto block = cooperative_groups::this_thread_block();

    const uint32_t patch_id = blockIdx.x;

    if (patch_id < context.m_num_patches[0]) {

        const PatchInfo patch_info = context.m_patches_info[patch_id];

        ShmemAllocator shrd_alloc;
        uint16_t*      s_fe =
            shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces[0]);
        uint16_t* s_ev =
            shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges[0]);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * patch_info.num_edges[0],
                   s_ev,
                   false);

        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces[0],
                   s_fe,
                   true);
        block.sync();


        // for every not-owned face, check that its three edges (possibly
        // not-owned) are the same as those in the face's owner patch
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces[0];
             f += blockThreads) {
            const LocalFaceT fl(f);
            if (!patch_info.is_deleted(fl) && !patch_info.is_owned(fl)) {

                uint16_t e0, e1, e2;
                flag_t   d0(0), d1(0), d2(0);
                uint32_t p0(patch_id), p1(patch_id), p2(patch_id);
                Context::unpack_edge_dir(s_fe[3 * f + 0], e0, d0);
                Context::unpack_edge_dir(s_fe[3 * f + 1], e1, d1);
                Context::unpack_edge_dir(s_fe[3 * f + 2], e2, d2);

                // if the edge is not owned, grab its local index in the owner
                // patch
                auto get_owned_e = [&](uint16_t&       e,
                                       uint32_t&       p,
                                       const PatchInfo pi) {
                    EdgeHandle eh =
                        context.get_owner_handle(EdgeHandle(pi.patch_id, {e}));

                    e = eh.local_id();
                    p = eh.patch_id();
                };

                get_owned_e(e0, p0, patch_info);
                get_owned_e(e1, p1, patch_info);
                get_owned_e(e2, p2, patch_info);

                // get f's three edges from its owner patch

                // face handle of this face (f) in its owner patch
                FaceHandle f_owned = context.get_owner_handle(
                    FaceHandle(patch_info.patch_id, fl));
                PatchInfo owner_patch_info =
                    context.m_patches_info[f_owned.patch_id()];

                // the owner patch should have indicate that the owned face is
                // owned by it
                if (!owner_patch_info.is_owned(
                        LocalFaceT(f_owned.local_id()))) {
                    // printf("\n 1 owned = %u", patch_id);
                    ::atomicAdd(d_check, 1);
                }

                // If a face is deleted, it should also be deleted in the other
                // patches that have it as not-owned
                if (owner_patch_info.is_deleted(
                        LocalFaceT(f_owned.local_id()))) {
                    // printf("\n 2 owned = %u", patch_id);
                    ::atomicAdd(d_check, 1);
                } else {
                    uint16_t ew0, ew1, ew2;
                    flag_t   dw0(0), dw1(0), dw2(0);
                    uint32_t pw0(f_owned.patch_id()), pw1(f_owned.patch_id()),
                        pw2(f_owned.patch_id());
                    Context::unpack_edge_dir(
                        owner_patch_info.fe[3 * f_owned.local_id() + 0].id,
                        ew0,
                        dw0);
                    Context::unpack_edge_dir(
                        owner_patch_info.fe[3 * f_owned.local_id() + 1].id,
                        ew1,
                        dw1);
                    Context::unpack_edge_dir(
                        owner_patch_info.fe[3 * f_owned.local_id() + 2].id,
                        ew2,
                        dw2);

                    get_owned_e(ew0, pw0, owner_patch_info);
                    get_owned_e(ew1, pw1, owner_patch_info);
                    get_owned_e(ew2, pw2, owner_patch_info);

                    if (e0 != ew0 || p0 != pw0 ||  //
                        e1 != ew1 || p1 != pw1 ||  //
                        e2 != ew2 || p2 != pw2) {
                        /*if (e0 != ew0 || p0 != pw0) {
                            printf(
                                "\n 3A owned patch= %u, f=%u, fw(%u, %u), "
                                "(p0=%u, e0=%u, pw0=%u, ew0=%u)",
                                patch_info.patch_id,
                                f,
                                f_owned.patch_id(),
                                f_owned.local_id(),
                                p0,
                                e0,
                                pw0,
                                ew0);
                        }

                        if (e1 != ew1 || p1 != pw1) {
                            printf(
                                "\n 3B owned patch= %u, f=%u, fw(%u, %u), "
                                "(p1=%u, e1=%u, pw1=%u, ew1=%u)",
                                patch_info.patch_id,
                                f,
                                f_owned.patch_id(),
                                f_owned.local_id(),
                                p1,
                                e1,
                                pw1,
                                ew1);
                        }

                        if (e2 != ew2 || p2 != pw2) {
                            printf(
                                "\n 3C owned patch= %u, f=%u, fw(%u, %u), "
                                "(p2=%u, e2=%u, pw2=%u, ew2=%u)",
                                patch_info.patch_id,
                                f,
                                f_owned.patch_id(),
                                f_owned.local_id(),
                                p2,
                                e2,
                                pw2,
                                ew2);
                        }*/
                        ::atomicAdd(d_check, 1);
                    }

                    if (d0 != dw0 || d1 != dw1 || d2 != dw2) {
                        // printf("\n 4 owned = %u", patch_id);
                        ::atomicAdd(d_check, 1);
                    }
                }
            }
        }

        // for every not-owned edge, check its two vertices (possibly
        // not-owned) are the same as those in the edge's owner patch
        for (uint16_t e = threadIdx.x; e < patch_info.num_edges[0];
             e += blockThreads) {

            const LocalEdgeT el(e);
            if (!patch_info.is_deleted(el) && !patch_info.is_owned(el)) {

                uint16_t v0 = s_ev[2 * e + 0];
                uint16_t v1 = s_ev[2 * e + 1];
                uint32_t p0(patch_id), p1(patch_id);

                auto get_owned_v =
                    [&](uint16_t& v, uint32_t& p, const PatchInfo pi) {
                        VertexHandle vh = context.get_owner_handle(
                            VertexHandle(pi.patch_id, {v}));

                        v = vh.local_id();
                        p = vh.patch_id();
                    };

                get_owned_v(v0, p0, patch_info);
                get_owned_v(v1, p1, patch_info);

                // get e's two vertices from its owner patch
                EdgeHandle e_owned = context.get_owner_handle(
                    EdgeHandle(patch_info.patch_id, el));

                PatchInfo owner_patch_info =
                    context.m_patches_info[e_owned.patch_id()];


                // the owner patch should have indicate that the owned face is
                // owned by it
                if (!owner_patch_info.is_owned(
                        LocalEdgeT(e_owned.local_id()))) {
                    // printf("\n 5 owned = %u", patch_id);
                    ::atomicAdd(d_check, 1);
                }

                // If an edge is deleted, it should also be deleted in the other
                // patches that have it as not-owned
                if (owner_patch_info.is_deleted(
                        LocalEdgeT(e_owned.local_id()))) {
                    // printf("\n 6 owned = %u", patch_id);
                    ::atomicAdd(d_check, 1);
                } else {

                    auto [vw0, vw1] =
                        owner_patch_info.get_edge_vertices(e_owned.local_id());

                    uint32_t pw0(e_owned.patch_id()), pw1(e_owned.patch_id());

                    get_owned_v(vw0, pw0, owner_patch_info);
                    get_owned_v(vw1, pw1, owner_patch_info);

                    /*if (v0 != vw0 || p0 != pw0 || v1 != vw1 || p1 != pw1) {
                        printf(
                            "\n 7 owned p %u, e= %u, v0= (%u, %u), v0_o= %u "
                            "v1= (%u, %u), v1_o= %u, ew= (%u,%u), vw0= (%u, "
                            "%u), vw0_0= %u, vw1= (%u, %u), vw0_1= %u",
                            patch_id,
                            e,
                            v0,
                            p0,
                            s_ev[2 * e + 0],
                            v1,
                            p1,
                            s_ev[2 * e + 1],
                            e_owned.local_id(),
                            e_owned.patch_id(),
                            vw0,
                            pw0,
                            owner_patch_info.ev[2 * e_owned.local_id() + 0].id,
                            vw1,
                            pw1,
                            owner_patch_info.ev[2 * e_owned.local_id() + 1].id);
                        ::atomicAdd(d_check, 1);
                    }*/
                }
            }
        }
    }
}


template <uint32_t blockThreads>
__global__ static void check_ribbon_edges(const Context           context,
                                          unsigned long long int* d_check)
{
    auto block = cooperative_groups::this_thread_block();

    const uint32_t patch_id = blockIdx.x;

    if (patch_id < context.m_num_patches[0]) {
        PatchInfo patch_info = context.m_patches_info[patch_id];

        ShmemAllocator shrd_alloc;
        uint16_t*      s_fe =
            shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces[0]);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces[0],
                   s_fe,
                   true);
        uint16_t* s_mark_edges =
            shrd_alloc.alloc<uint16_t>(patch_info.num_edges[0]);

        for (uint16_t e = threadIdx.x; e < patch_info.num_edges[0];
             e += blockThreads) {
            s_mark_edges[e] = 0;
        }

        block.sync();

        // Check that each owned edge is incident to at least one owned
        // not-deleted face. We do that by iterating over faces, each face
        // (atomically) mark its incident edges only if they are owned. Then we
        // check the marked edges where we expect all owned edges to be marked.
        // If there is an edge that is owned but not marked, then this edge is
        // not incident to any owned faces
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces[0];
             f += blockThreads) {
            const LocalFaceT fl(f);

            if (!patch_info.is_deleted(fl) && patch_info.is_owned(fl)) {

                uint16_t e0 = s_fe[3 * f + 0] >> 1;
                uint16_t e1 = s_fe[3 * f + 1] >> 1;
                uint16_t e2 = s_fe[3 * f + 2] >> 1;

                auto mark_if_owned = [&](uint16_t edge) {
                    if (patch_info.is_owned(LocalEdgeT(edge))) {
                        atomicAdd(s_mark_edges + edge, uint16_t(1));
                    }
                };

                mark_if_owned(e0);
                mark_if_owned(e1);
                mark_if_owned(e2);
            }
        }
        block.sync();
        for (uint16_t e = threadIdx.x; e < patch_info.num_edges[0];
             e += blockThreads) {
            const LocalEdgeT el(e);
            if (patch_info.is_owned(el) && !patch_info.is_deleted(el)) {
                if (s_mark_edges[e] == 0) {
                    // printf("\n ribbon edge = %u, %u, v0= %u, v1= %u",
                    //        patch_id,
                    //        e,
                    //        patch_info.ev[2 * e + 0].id,
                    //        patch_info.ev[2 * e + 1].id);
                    ::atomicAdd(d_check, 1);
                }
            }
        }
    }
}


template <uint32_t blockThreads>
__global__ static void compute_vf(const Context               context,
                                  VertexAttribute<FaceHandle> output)
{
    using namespace rxmesh;

    auto store_lambda = [&](VertexHandle& v_id, FaceIterator& iter) {
        for (uint32_t i = 0; i < iter.size(); ++i) {
            output(v_id, i) = iter[i];
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VF>(block, shrd_alloc, store_lambda);
}


template <uint32_t blockThreads>
__global__ static void compute_max_valence(const Context context,
                                           uint32_t*     d_max_valence)
{
    using namespace rxmesh;

    auto max_valence = [&](VertexHandle& v_id, VertexIterator& iter) {
        ::atomicMax(d_max_valence, iter.size());
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, max_valence);
}

template <uint32_t blockThreads>
__global__ static void check_ribbon_faces(const Context               context,
                                          VertexAttribute<FaceHandle> global_vf,
                                          unsigned long long int*     d_check)
{
    auto block = cooperative_groups::this_thread_block();

    const uint32_t patch_id = blockIdx.x;

    if (patch_id < context.m_num_patches[0]) {
        PatchInfo patch_info = context.m_patches_info[patch_id];

        ShmemAllocator shrd_alloc;

        uint16_t* s_fv =
            shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces[0]);
        uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(std::max(
            3 * patch_info.num_faces[0], 1 + patch_info.num_vertices[0]));
        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(
            std::max(2 * patch_info.num_edges[0], 3 * patch_info.num_faces[0]));
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * patch_info.num_edges[0],
                   s_ev,
                   false);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces[0],
                   s_fv,
                   true);
        block.sync();


        // compute FV
        f_v<blockThreads>(patch_info.num_edges[0],
                          s_ev,
                          patch_info.num_faces[0],
                          s_fv,
                          patch_info.active_mask_f);
        block.sync();

        // copy FV
        for (uint16_t i = threadIdx.x; i < 3 * patch_info.num_faces[0];
             i += blockThreads) {
            s_fe[i] = s_fv[i];
        }
        block.sync();

        // compute (local) VF by transposing FV
        uint16_t* s_vf_offset = &s_fe[0];
        uint16_t* s_vf_value  = &s_ev[0];
        block_mat_transpose<3u, blockThreads>(patch_info.num_faces[0],
                                              patch_info.num_vertices[0],
                                              s_fe,
                                              s_ev,
                                              patch_info.active_mask_f,
                                              0);
        block.sync();

        // For every incident vertex V to an owned face, check if VF of V
        // using global_VF can be retrieved from local_VF
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces[0];
             f += blockThreads) {

            const LocalFaceT fl(f);

            // Only if the face is owned, we do the check
            if (!patch_info.is_deleted(fl) && patch_info.is_owned(fl)) {

                // for the three vertices incident to this face
                for (uint16_t k = 0; k < 3; ++k) {
                    uint16_t v_id = s_fv[3 * f + k];

                    // get the vertex handle so we can index the attributes
                    assert(!patch_info.is_deleted(LocalVertexT(v_id)));

                    const VertexHandle vh = context.get_owner_handle(
                        VertexHandle(patch_id, {v_id}));

                    // for every incident face to this vertex
                    for (uint16_t i = 0; i < global_vf.get_num_attributes();
                         ++i) {

                        const FaceHandle fvh_global = global_vf(vh, i);

                        if (fvh_global.is_valid()) {

                            // look for the face incident to the vertex in local
                            // VF
                            assert(s_vf_offset[v_id + 1] > s_vf_offset[v_id]);
                            bool found = false;
                            for (uint16_t j = s_vf_offset[v_id];
                                 j < s_vf_offset[v_id + 1];
                                 ++j) {

                                assert(!patch_info.is_deleted(
                                    LocalFaceT(s_vf_value[j])));

                                const FaceHandle fh = context.get_owner_handle(
                                    FaceHandle(patch_id, {s_vf_value[j]}));

                                if (fvh_global == fh) {
                                    found = true;
                                    break;
                                }
                            }

                            if (!found) {
                                // printf(
                                //     "\n T=%u, p = %u, #F=%u, #F_owned= %u, "
                                //     "#E=%u, #E_owned= %u, #V=%u, #V_owned=
                                //     %u, " "f= %u, fvh_global=%u, %u, v_id=
                                //     %u, " "vh=%u,%u, s_vf =%u, %u",
                                //     threadIdx.x,
                                //     patch_id,
                                //     patch_info.num_faces[0],
                                //     patch_info.get_num_owned<FaceHandle>(),
                                //     patch_info.num_edges[0],
                                //     patch_info.get_num_owned<EdgeHandle>(),
                                //     patch_info.num_vertices[0],
                                //     patch_info.get_num_owned<VertexHandle>(),
                                //     f,
                                //     fvh_global.patch_id(),
                                //     fvh_global.local_id(),
                                //     v_id,
                                //     vh.patch_id(),
                                //     vh.local_id(),
                                //     s_vf_offset[v_id],
                                //     s_vf_offset[v_id + 1]);
                                ::atomicAdd(d_check, 1);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace detail


void RXMeshDynamic::save(std::string filename)
{
    /*if (m_patcher->m_num_patches != get_num_patches()) {
        RXMESH_ERROR(
            "RXMeshDynamic:save() does not support changing number of "
            "patches in the mesh");
    }

    m_patcher->m_max_num_patches = get_num_patches();

    m_patcher->m_num_vertices = get_num_vertices();
    m_patcher->m_vertex_patch.resize(m_patcher->m_num_vertices);

    m_patcher->m_num_edges = get_num_edges();
    m_patcher->m_edge_patch.resize(m_patcher->m_num_edges);

    m_patcher->m_num_faces = get_num_faces();
    m_patcher->m_face_patch.resize(m_patcher->m_num_faces);


    for_each_vertex(HOST, [&](VertexHandle vh) {
        m_patcher->m_vertex_patch[linear_id(vh)] = vh.patch_id();
    });

    for_each_edge(HOST, [&](EdgeHandle eh) {
        m_patcher->m_edge_patch[linear_id(eh)] = eh.patch_id();
    });

    for_each_face(HOST, [&](FaceHandle fh) {
        m_patcher->m_face_patch[linear_id(fh)] = fh.patch_id();
    });


    m_patcher->m_patches_offset.resize(get_num_patches(), 0);
    std::fill(m_patcher->m_patches_offset.begin(),
              m_patcher->m_patches_offset.end(),
              0);
    m_patcher->m_patches_val.resize(get_num_faces());

    for_each_face(
        HOST,
        [&](FaceHandle fh) { m_patcher->m_patches_offset[fh.patch_id()]++; },
        NULL,
        false);

    std::inclusive_scan(m_patcher->m_patches_offset.begin(),
                        m_patcher->m_patches_offset.end(),
                        m_patcher->m_patches_offset.begin());

    std::vector<int> offset(get_num_patches(), 0);

    for_each_face(
        HOST,
        [&](FaceHandle fh) {
            uint32_t p_offset =
                (fh.patch_id() == 0) ?
                    0 :
                    m_patcher->m_patches_offset[fh.patch_id() - 1];
            m_patcher->m_patches_val[p_offset + (offset[fh.patch_id()]++)] =
                linear_id(fh);
        },
        NULL,
        false);

    // update m_ribbon_ext_val and m_ribbon_ext_offset
    m_patcher->m_ribbon_ext_offset.resize(get_num_patches(), 0);
    std::fill(m_patcher->m_ribbon_ext_offset.begin(),
              m_patcher->m_ribbon_ext_offset.end(),
              0);
    m_patcher->m_ribbon_ext_val.resize(get_num_faces(), 0);


    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        uint16_t num_not_owned_faces =
            m_h_patches_info[p].num_faces[0] -
            m_h_patches_info[p].get_num_owned<FaceHandle>();
        m_patcher->m_ribbon_ext_offset[p] = num_not_owned_faces;
    }

    std::inclusive_scan(m_patcher->m_ribbon_ext_offset.begin(),
                        m_patcher->m_ribbon_ext_offset.end(),
                        m_patcher->m_ribbon_ext_offset.begin());


    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        uint16_t offset = 0;
        uint32_t p_offset =
            (p == 0) ? 0 : m_patcher->m_ribbon_ext_offset[p - 1];

        for (uint16_t f = 0; f < m_h_patches_info[p].num_faces[0]; ++f) {
            LocalFaceT fl(f);
            if (!m_h_patches_info[p].is_owned(fl) &&
                !m_h_patches_info[p].is_deleted(fl)) {
                FaceHandle fh  = get_owner_handle<FaceHandle>({p, fl});
                uint32_t   fid = linear_id(fh);
                m_patcher->m_ribbon_ext_val[p_offset + offset++] = fid;
            }
        }
    }*/


    RXMesh::save(filename);
}


bool RXMeshDynamic::validate()
{
    CUDA_ERROR(cudaDeviceSynchronize());
    RXMESH_TRACE("RXMeshDynamic validation started");

    uint32_t num_patches;
    CUDA_ERROR(cudaMemcpy(&num_patches,
                          m_rxmesh_context.m_num_patches,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    unsigned long long int* d_check;
    CUDA_ERROR(cudaMalloc((void**)&d_check, sizeof(unsigned long long int)));

    assert(num_patches == get_num_patches());

    auto is_okay = [&]() {
        unsigned long long int h_check(0);
        CUDA_ERROR(cudaMemcpy(&h_check,
                              d_check,
                              sizeof(unsigned long long int),
                              cudaMemcpyDeviceToHost));
        if (h_check != 0) {
            return false;
        } else {
            return true;
        }
    };

    // check that the sum of owned vertices, edges, and faces per patch is equal
    // to the number of vertices, edges, and faces respectively
    auto check_num_mesh_elements = [&]() -> bool {
        uint32_t *d_sum_num_vertices, *d_sum_num_edges, *d_sum_num_faces;
        thrust::device_vector<uint32_t> d_sum_vertices(1, 0);
        thrust::device_vector<uint32_t> d_sum_edges(1, 0);
        thrust::device_vector<uint32_t> d_sum_faces(1, 0);

        constexpr uint32_t block_size = 256;
        const uint32_t     grid_size  = num_patches;

        detail::calc_num_elements<block_size>
            <<<grid_size, block_size>>>(m_rxmesh_context,
                                        d_sum_vertices.data().get(),
                                        d_sum_edges.data().get(),
                                        d_sum_faces.data().get());

        uint32_t num_vertices, num_edges, num_faces;
        CUDA_ERROR(cudaMemcpy(&num_vertices,
                              m_rxmesh_context.m_num_vertices,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(&num_edges,
                              m_rxmesh_context.m_num_edges,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(&num_faces,
                              m_rxmesh_context.m_num_faces,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        uint32_t sum_num_vertices, sum_num_edges, sum_num_faces;
        thrust::copy(
            d_sum_vertices.begin(), d_sum_vertices.end(), &sum_num_vertices);
        thrust::copy(d_sum_edges.begin(), d_sum_edges.end(), &sum_num_edges);
        thrust::copy(d_sum_faces.begin(), d_sum_faces.end(), &sum_num_faces);

        if (num_vertices != sum_num_vertices || num_edges != sum_num_edges ||
            num_faces != sum_num_faces) {
            return false;
        } else {
            return true;
        }
    };

    // check that each edge is composed of two unique vertices and each face is
    // composed of three unique edges that give three unique vertices.
    auto check_uniqueness = [&]() -> bool {
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long int)));
        constexpr uint32_t block_size = 256;
        const uint32_t     grid_size  = num_patches;
        const uint32_t     dynamic_smem =
            rxmesh::ShmemAllocator::default_alignment * 2 +
            (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
            (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

        detail::check_uniqueness<block_size>
            <<<grid_size, block_size, dynamic_smem>>>(m_rxmesh_context,
                                                      d_check);

        return is_okay();
    };

    // check that every not-owned mesh elements' connectivity (faces and
    // edges) is equivalent to their connectivity in their owner patch.
    // if the mesh element is deleted in the owner patch, no check is done
    auto check_not_owned = [&]() -> bool {
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long int)));
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long int)));

        constexpr uint32_t block_size = 256;
        const uint32_t     grid_size  = num_patches;
        const uint32_t     dynamic_smem =
            ShmemAllocator::default_alignment * 2 +
            (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
            (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

        detail::check_not_owned<block_size>
            <<<grid_size, block_size, dynamic_smem>>>(m_rxmesh_context,
                                                      d_check);
        return is_okay();
    };

    // check if the ribbon construction is complete i.e., 1) each owned edge is
    // incident to an owned face, and 2) VF of the three vertices of an owned
    // face is inside the patch
    auto check_ribbon = [&]() {
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long int)));
        constexpr uint32_t block_size = 512;
        const uint32_t     grid_size  = num_patches;
        uint32_t           dynamic_smem =
            ShmemAllocator::default_alignment * 3 +
            (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
            this->m_max_edges_per_patch * sizeof(uint16_t);

        detail::check_ribbon_edges<block_size>
            <<<grid_size, block_size, dynamic_smem>>>(m_rxmesh_context,
                                                      d_check);

        if (!is_okay()) {
            RXMESH_WARN("RXMeshDynamic::validate() check_ribbon_edges failed");
            return false;
            // CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long
            // int)));
        }

        uint32_t* d_max_valence;
        CUDA_ERROR(cudaMalloc((void**)&d_max_valence, sizeof(uint32_t)));
        CUDA_ERROR(cudaMemset(d_max_valence, 0, sizeof(uint32_t)));

        LaunchBox<block_size> launch_box;
        RXMeshStatic::prepare_launch_box(
            {Op::VV},
            launch_box,
            (void*)detail::compute_max_valence<block_size>);
        detail::compute_max_valence<block_size>
            <<<launch_box.blocks, block_size, launch_box.smem_bytes_dyn>>>(
                m_rxmesh_context, d_max_valence);


        uint32_t h_max_valence = 0;
        CUDA_ERROR(cudaMemcpy(&h_max_valence,
                              d_max_valence,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        GPU_FREE(d_max_valence);

        auto vf_global = this->add_vertex_attribute<FaceHandle>(
            "vf", h_max_valence, rxmesh::DEVICE);
        vf_global->reset(FaceHandle(), rxmesh::DEVICE);


        RXMeshStatic::prepare_launch_box(
            {Op::VF}, launch_box, (void*)detail::compute_vf<block_size>);

        detail::compute_vf<block_size>
            <<<launch_box.blocks, block_size, launch_box.smem_bytes_dyn>>>(
                m_rxmesh_context, *vf_global);

        dynamic_smem =
            ShmemAllocator::default_alignment * 3 +
            2 * (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
            std::max(3 * this->m_max_faces_per_patch,
                     2 * this->m_max_edges_per_patch) *
                sizeof(uint16_t);

        detail::check_ribbon_faces<block_size>
            <<<grid_size, block_size, dynamic_smem>>>(
                m_rxmesh_context, *vf_global, d_check);

        remove_attribute("vf");
        return is_okay();
    };


    // Every mesh element in the hash table of a patch should be mapped to an
    // owner that lives in a different patch. Otherwise, the mesh element is
    // actually duplicated. We check on this here. We also check that when we
    // query the hashtable for a (not-owned) mesh element, what we get is a mesh
    // element that is owned by the owner patch
    auto check_hashtable = [&]() {
        for (uint32_t p = 0; p < get_num_patches(); ++p) {
            PatchInfo pi = m_h_patches_info[p];

            for (uint32_t v = 0; v < pi.num_vertices[0]; v++) {
                const LocalVertexT vl(v);
                if (!pi.is_deleted(vl) && !pi.is_owned(vl)) {
                    const VertexHandle vh = pi.find<VertexHandle>(v);
                    if (!vh.is_valid()) {
                        return false;
                    }
                    if (vh.patch_id() == p) {
                        return false;
                    }
                    if (vh.patch_id() > get_num_patches()) {
                        return false;
                    }
                    if (m_h_patches_info[vh.patch_id()].is_deleted(
                            LocalVertexT(vh.local_id()))) {
                        return false;
                    }
                    if (!m_h_patches_info[vh.patch_id()].is_owned(
                            LocalVertexT(vh.local_id()))) {
                        return false;
                    }
                }
            }


            for (uint32_t e = 0; e < pi.num_edges[0]; e++) {
                const LocalEdgeT el(e);
                if (!pi.is_deleted(el) && !pi.is_owned(el)) {
                    const EdgeHandle eh = pi.find<EdgeHandle>(e);
                    if (!eh.is_valid()) {
                        return false;
                    }
                    if (eh.patch_id() == p) {
                        return false;
                    }
                    if (eh.patch_id() > get_num_patches()) {
                        return false;
                    }
                    if (m_h_patches_info[eh.patch_id()].is_deleted(
                            LocalEdgeT(eh.local_id()))) {
                        return false;
                    }
                    if (!m_h_patches_info[eh.patch_id()].is_owned(
                            LocalEdgeT(eh.local_id()))) {
                        return false;
                    }
                }
            }


            for (uint32_t f = 0; f < pi.num_faces[0]; f++) {
                const LocalFaceT fl(f);
                if (!pi.is_deleted(fl) && !pi.is_owned(fl)) {
                    const FaceHandle fh = pi.find<FaceHandle>(f);
                    if (!fh.is_valid()) {
                        return false;
                    }
                    if (fh.patch_id() == p) {
                        return false;
                    }
                    if (fh.patch_id() > get_num_patches()) {
                        return false;
                    }
                    if (m_h_patches_info[fh.patch_id()].is_deleted(
                            LocalFaceT(fh.local_id()))) {
                        return false;
                    }
                    if (!m_h_patches_info[fh.patch_id()].is_owned(
                            LocalFaceT(fh.local_id()))) {
                        return false;
                    }
                }
            }
        }

        return true;
    };

    // check if a patch p has q in its patch stash, then q also has p in its
    // patch stash
    auto patch_stash_inclusion = [&]() {
        for (uint32_t p = 0; p < get_num_patches(); ++p) {
            for (uint8_t p_sh = 0; p_sh < PatchStash::stash_size; ++p_sh) {
                uint32_t q = m_h_patches_info[p].patch_stash.get_patch(p_sh);
                if (q != INVALID32) {
                    bool found = false;
                    for (uint8_t q_sh = 0; q_sh < PatchStash::stash_size;
                         ++q_sh) {
                        if (m_h_patches_info[q].patch_stash.get_patch(q_sh) ==
                            p) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    // check that a patch is only encountered once in patch stash
    auto unique_patch_stash = [&]() {
        for (uint32_t p = 0; p < get_num_patches(); ++p) {
            for (uint8_t p_sh = 0; p_sh < PatchStash::stash_size; ++p_sh) {
                uint32_t q = m_h_patches_info[p].patch_stash.get_patch(p_sh);
                if (q != INVALID32) {
                    bool duplicated = false;

                    for (uint8_t pp_sh = 0; pp_sh < PatchStash::stash_size;
                         ++pp_sh) {
                        if (pp_sh == p_sh) {
                            continue;
                        }
                        if (m_h_patches_info[p].patch_stash.get_patch(pp_sh) ==
                            q) {
                            duplicated = true;
                            break;
                        }
                    }
                    if (duplicated) {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    bool success = true;
    if (!check_num_mesh_elements()) {
        RXMESH_ERROR(
            "RXMeshDynamic::validate() check_num_mesh_elements failed");
        success = false;
    }

    if (!check_hashtable()) {
        RXMESH_ERROR("RXMeshDynamic::validate() check_hashtable failed");
        success = false;
    }

    if (!check_uniqueness()) {
        RXMESH_ERROR("RXMeshDynamic::validate() check_uniqueness failed");
        success = false;
    }

    if (!check_not_owned()) {
        RXMESH_ERROR("RXMeshDynamic::validate() check_not_owned failed");
        success = false;
    }

    if (!check_ribbon()) {
        RXMESH_ERROR("RXMeshDynamic::validate() check_ribbon failed");
        success = false;
    }

    /*if (!patch_stash_inclusion()) {
        RXMESH_ERROR("RXMeshDynamic::validate() patch_stash_inclusion failed");
        success = false;
    }*/

    if (!unique_patch_stash()) {
        RXMESH_ERROR("RXMeshDynamic::validate() unique_patch_stash failed");
        success = false;
    }

    CUDA_ERROR(cudaFree(d_check));

    RXMESH_TRACE("RXMeshDynamic validation finished");
    return success;
}

void RXMeshDynamic::cleanup()
{
    CUDA_ERROR(cudaMemcpy(&m_num_patches,
                          m_rxmesh_context.m_num_patches,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    constexpr uint32_t block_size = 256;
    const uint32_t     grid_size  = get_num_patches();

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

    CUDA_ERROR(
        cudaMemset(m_rxmesh_context.m_num_vertices, 0, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(m_rxmesh_context.m_num_edges, 0, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(m_rxmesh_context.m_num_faces, 0, sizeof(uint32_t)));

    CUDA_ERROR(
        cudaMemset(m_rxmesh_context.m_max_num_vertices, 0, sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMemset(m_rxmesh_context.m_max_num_edges, 0, sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMemset(m_rxmesh_context.m_max_num_faces, 0, sizeof(uint32_t)));


    uint32_t dyn_shmem = 0;

    dyn_shmem += 3 * detail::mask_num_bytes(this->m_max_vertices_per_patch) +
                 3 * ShmemAllocator::default_alignment;

    dyn_shmem += 3 * detail::mask_num_bytes(this->m_max_edges_per_patch) +
                 3 * ShmemAllocator::default_alignment;

    dyn_shmem += 3 * detail::mask_num_bytes(this->m_max_faces_per_patch) +
                 3 * ShmemAllocator::default_alignment;

    uint32_t hash_table_shmem =
        std::max(sizeof(LPPair) * max_lp_hashtable_capacity<LocalEdgeT>(),
                 sizeof(LPPair) * max_lp_hashtable_capacity<LocalVertexT>());
    hash_table_shmem = std::max(
        hash_table_shmem,
        uint32_t(sizeof(LPPair) * max_lp_hashtable_capacity<LocalFaceT>()));
    hash_table_shmem += ShmemAllocator::default_alignment;

    uint32_t connect_shmem =
        2 * ShmemAllocator::default_alignment +
        (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
        (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

    dyn_shmem += std::max(hash_table_shmem, connect_shmem);

    detail::hashtable_calibration<block_size>
        <<<grid_size, block_size>>>(this->m_rxmesh_context);

    detail::remove_surplus_elements<block_size>
        <<<grid_size, block_size, dyn_shmem>>>(this->m_rxmesh_context);

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
}

void RXMeshDynamic::update_host()
{
    RXMESH_TRACE("RXMeshDynamic updating host started");

    auto resize_masks = [&](uint16_t   size,
                            uint16_t&  capacity,
                            uint32_t*& active_mask,
                            uint32_t*& owned_mask) {
        if (size > capacity) {
            capacity = size;
            free(active_mask);
            free(owned_mask);
            active_mask = (uint32_t*)malloc(detail::mask_num_bytes(size));
            owned_mask  = (uint32_t*)malloc(detail::mask_num_bytes(size));
        }
    };

    uint32_t num_patches = 0;
    CUDA_ERROR(cudaMemcpy(&num_patches,
                          m_rxmesh_context.m_num_patches,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    if (num_patches > get_max_num_patches()) {
        RXMESH_ERROR(
            "RXMeshDynamic::update_host() number of patches is bigger than the "
            "maximum expect number of patches");
    }
    m_num_patches = num_patches;

    for (uint32_t p = 0; p < m_num_patches; ++p) {
        PatchInfo d_patch;
        CUDA_ERROR(cudaMemcpy(&d_patch,
                              m_d_patches_info + p,
                              sizeof(PatchInfo),
                              cudaMemcpyDeviceToHost));

        assert(d_patch.patch_id == p);

        CUDA_ERROR(cudaMemcpy(m_h_patches_info[p].num_vertices,
                              d_patch.num_vertices,
                              sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_h_patches_info[p].num_edges,
                              d_patch.num_edges,
                              sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_h_patches_info[p].num_faces,
                              d_patch.num_faces,
                              sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));

        // resize topology (don't update capacity here)
        if (m_h_patches_info[p].num_edges[0] >
            m_h_patches_info[p].edges_capacity[0]) {
            free(m_h_patches_info[p].ev);
            m_h_patches_info[p].ev = (LocalVertexT*)malloc(
                m_h_patches_info[p].num_edges[0] * 2 * sizeof(LocalVertexT));
        }

        if (m_h_patches_info[p].num_faces[0] >
            m_h_patches_info[p].faces_capacity[0]) {
            free(m_h_patches_info[p].fe);
            m_h_patches_info[p].fe = (LocalEdgeT*)malloc(
                m_h_patches_info[p].num_faces[0] * 3 * sizeof(LocalEdgeT));
        }

        // copy topology
        CUDA_ERROR(cudaMemcpy(
            m_h_patches_info[p].ev,
            d_patch.ev,
            2 * m_h_patches_info[p].num_edges[0] * sizeof(LocalVertexT),
            cudaMemcpyDeviceToHost));

        CUDA_ERROR(cudaMemcpy(
            m_h_patches_info[p].fe,
            d_patch.fe,
            3 * m_h_patches_info[p].num_faces[0] * sizeof(LocalEdgeT),
            cudaMemcpyDeviceToHost));

        // resize mask (update capacity)
        resize_masks(m_h_patches_info[p].num_vertices[0],
                     m_h_patches_info[p].vertices_capacity[0],
                     m_h_patches_info[p].active_mask_v,
                     m_h_patches_info[p].owned_mask_v);

        resize_masks(m_h_patches_info[p].num_edges[0],
                     m_h_patches_info[p].edges_capacity[0],
                     m_h_patches_info[p].active_mask_e,
                     m_h_patches_info[p].owned_mask_e);

        resize_masks(m_h_patches_info[p].num_faces[0],
                     m_h_patches_info[p].faces_capacity[0],
                     m_h_patches_info[p].active_mask_f,
                     m_h_patches_info[p].owned_mask_f);

        // copy masks
        CUDA_ERROR(cudaMemcpy(
            m_h_patches_info[p].active_mask_v,
            d_patch.active_mask_v,
            detail::mask_num_bytes(m_h_patches_info[p].num_vertices[0]),
            cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(
            m_h_patches_info[p].owned_mask_v,
            d_patch.owned_mask_v,
            detail::mask_num_bytes(m_h_patches_info[p].num_vertices[0]),
            cudaMemcpyDeviceToHost));


        CUDA_ERROR(
            cudaMemcpy(m_h_patches_info[p].active_mask_e,
                       d_patch.active_mask_e,
                       detail::mask_num_bytes(m_h_patches_info[p].num_edges[0]),
                       cudaMemcpyDeviceToHost));
        CUDA_ERROR(
            cudaMemcpy(m_h_patches_info[p].owned_mask_e,
                       d_patch.owned_mask_e,
                       detail::mask_num_bytes(m_h_patches_info[p].num_edges[0]),
                       cudaMemcpyDeviceToHost));

        CUDA_ERROR(
            cudaMemcpy(m_h_patches_info[p].active_mask_f,
                       d_patch.active_mask_f,
                       detail::mask_num_bytes(m_h_patches_info[p].num_faces[0]),
                       cudaMemcpyDeviceToHost));
        CUDA_ERROR(
            cudaMemcpy(m_h_patches_info[p].owned_mask_f,
                       d_patch.owned_mask_f,
                       detail::mask_num_bytes(m_h_patches_info[p].num_faces[0]),
                       cudaMemcpyDeviceToHost));


        // copy patch stash
        CUDA_ERROR(cudaMemcpy(m_h_patches_info[p].patch_stash.m_stash,
                              d_patch.patch_stash.m_stash,
                              PatchStash::stash_size * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        // dirty
        CUDA_ERROR(cudaMemcpy(m_h_patches_info[p].dirty,
                              d_patch.dirty,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

        // child id
        m_h_patches_info[p].child_id = d_patch.child_id;

        // should slice
        m_h_patches_info[p].should_slice = d_patch.should_slice;

        // copy lp hashtable
        m_h_patches_info[p].lp_v.move(d_patch.lp_v);
        m_h_patches_info[p].lp_e.move(d_patch.lp_e);
        m_h_patches_info[p].lp_f.move(d_patch.lp_f);
    }


    CUDA_ERROR(cudaMemcpy(&this->m_num_vertices,
                          m_rxmesh_context.m_num_vertices,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(&this->m_num_edges,
                          m_rxmesh_context.m_num_edges,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(&this->m_num_faces,
                          m_rxmesh_context.m_num_faces,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

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

    // count and update num_owned and it prefix sum
    m_h_vertex_prefix[0] = 0;
    m_h_edge_prefix[0]   = 0;
    m_h_face_prefix[0]   = 0;
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        m_h_num_owned_v[p] = m_h_patches_info[p].get_num_owned<VertexHandle>();
        m_h_vertex_prefix[p + 1] = m_h_vertex_prefix[p] + m_h_num_owned_v[p];

        m_h_num_owned_e[p] = m_h_patches_info[p].get_num_owned<EdgeHandle>();
        m_h_edge_prefix[p + 1] = m_h_edge_prefix[p] + m_h_num_owned_e[p];

        m_h_num_owned_f[p] = m_h_patches_info[p].get_num_owned<FaceHandle>();
        m_h_face_prefix[p + 1] = m_h_face_prefix[p] + m_h_num_owned_f[p];
    }

    if (m_h_vertex_prefix[m_num_patches] != this->m_num_vertices) {
        RXMESH_ERROR(
            "RXMeshDynamic::update_host error in updating host. m_num_vertices "
            "{} does not match m_h_vertex_prefix calculation {}",
            this->m_num_vertices,
            m_h_vertex_prefix[m_num_patches]);
    }
    this->m_num_vertices = m_h_vertex_prefix[m_num_patches];
    if (m_h_edge_prefix[m_num_patches] != this->m_num_edges) {
        RXMESH_ERROR(
            "RXMeshDynamic::update_host error in updating host. m_num_edges "
            "{} does not match m_h_edge_prefix calculation {}",
            this->m_num_edges,
            m_h_edge_prefix[m_num_patches]);
    }
    this->m_num_edges = m_h_edge_prefix[m_num_patches];
    if (m_h_face_prefix[m_num_patches] != this->m_num_faces) {
        RXMESH_ERROR(
            "RXMeshDynamic::update_host error in updating host. m_num_faces "
            "{} does not match m_h_face_prefix calculation {}",
            this->m_num_faces,
            m_h_face_prefix[m_num_patches]);
    }
    this->m_num_faces = m_h_face_prefix[m_num_patches];

    const uint32_t patches_1_bytes = (m_num_patches + 1) * sizeof(uint32_t);
    CUDA_ERROR(cudaMemcpy(m_d_vertex_prefix,
                          m_h_vertex_prefix,
                          patches_1_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_edge_prefix,
                          m_h_edge_prefix,
                          patches_1_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_face_prefix,
                          m_h_face_prefix,
                          patches_1_bytes,
                          cudaMemcpyHostToDevice));

    this->calc_max_elements();

    RXMESH_TRACE("RXMeshDynamic updating host finished");
}

void RXMeshDynamic::update_polyscope(std::string new_name)
{
#if USE_POLYSCOPE
    // for polyscope, we just remove the mesh and re-add it since polyscope does
    // not support changing the mesh topology
    // if (this->m_polyscope_mesh_name.find("updated") != std::string::npos) {
    // polyscope::removeSurfaceMesh(this->m_polyscope_mesh_name, true);
    //}
    if (new_name.empty()) {
        this->m_polyscope_mesh_name = this->m_polyscope_mesh_name + "updated";
    } else {
        this->m_polyscope_mesh_name = new_name;
    }
    this->register_polyscope();
#endif
}


template __device__ void detail::slice<256>(
    Context&,
    cooperative_groups::thread_block&,
    PatchInfo&,
    const uint32_t,
    const uint16_t,
    const uint16_t,
    const uint16_t,
    PatchStash&,
    // PatchStash&,
    Bitmask&,
    Bitmask&,
    Bitmask&,
    const Bitmask&,
    const Bitmask&,
    const Bitmask&,
    const uint16_t*,
    const uint16_t*,
    Bitmask&,
    Bitmask&,
    Bitmask&,
    Bitmask&,
    Bitmask&,
    Bitmask&);

template __device__ void detail::bi_assignment<256>(
    cooperative_groups::thread_block&,
    const uint16_t,
    const uint16_t,
    const uint16_t,
    const Bitmask&,
    const Bitmask&,
    const Bitmask&,
    const Bitmask&,
    const Bitmask&,
    const Bitmask&,
    const uint16_t*,
    const uint16_t*,
    Bitmask&,
    Bitmask&,
    Bitmask&);

template __device__ void detail::bi_assignment_ggp<256>(
    cooperative_groups::thread_block&,
    const uint16_t,
    const Bitmask&,
    const bool,
    const Bitmask&,
    const uint16_t*,
    const uint16_t*,
    Bitmask&,
    Bitmask&,
    Bitmask&,
    Bitmask&,
    Bitmask&,
    int);
}  // namespace rxmesh
