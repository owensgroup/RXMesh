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
                                                 PatchInfo&    pi)
{
    // TODO cleanup patch stash
    // TODO load the hashtable in shared memory
    // TODO cleanup the hash table for stall elements
    using LocalT = typename HandleT::LocalT;

    const uint16_t num_elements = *(pi.get_num_elements<HandleT>());

    const uint16_t num_elements_up =
        ROUND_UP_TO_NEXT_MULTIPLE(num_elements, blockThreads);

    for (uint16_t i = threadIdx.x; i < num_elements_up; i += blockThreads) {
        HandleT handle;
        bool    replace = false;

        // int probe = 0;

        if (i < num_elements) {

            if (!pi.is_owned(LocalT(i)) && !pi.is_deleted(LocalT(i))) {

                // This is the same implementation in
                // Context::get_owner_handle()

                uint32_t owner = pi.patch_id;
                uint16_t lid   = i;

                LPPair lp = pi.get_lp<HandleT>().find(lid);

                assert(!lp.is_sentinel());

                owner = pi.patch_stash.get_patch(lp);

                assert(owner != INVALID32);

                // This only happen when the element i resides in the cavity of
                // the owner where it will be cleaned up later in
                // remove_surplus_elements
                if (!context.m_patches_info[owner].is_deleted(
                        LocalT(lp.local_id_in_owner_patch()))) {

                    // assert(!context.m_patches_info[owner].is_deleted(
                    //    LocalT(lp.local_id_in_owner_patch())));


                    while (!context.m_patches_info[owner].is_owned(
                        LocalT(lp.local_id_in_owner_patch()))) {

                        // probe++;

                        replace = true;

                        lp = context.m_patches_info[owner]
                                 .get_lp<HandleT>()
                                 .find(lp.local_id_in_owner_patch());

                        assert(!lp.is_sentinel());

                        owner =
                            context.m_patches_info[owner].patch_stash.get_patch(
                                lp);

                        if (context.m_patches_info[owner].is_deleted(
                                LocalT(lp.local_id_in_owner_patch()))) {
                            replace = false;
                            // printf("\n probe = %d, p= %u, type= %s, owner=
                            // %u",
                            //       probe,
                            //       pi.patch_id,
                            //       LocalT::name(),
                            //       owner);
                            break;
                        }
                        assert(!context.m_patches_info[owner].is_deleted(
                            LocalT(lp.local_id_in_owner_patch())));
                    }

                    handle = HandleT(owner, lp.local_id_in_owner_patch());
                }
            }
        }

        __syncthreads();

        if (replace) {

            uint8_t o = pi.patch_stash.insert_patch(handle.patch_id());

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

    PatchInfo pi = context.m_patches_info[pid];

    hashtable_calibration<blockThreads, VertexHandle>(context, pi);
    hashtable_calibration<blockThreads, EdgeHandle>(context, pi);
    hashtable_calibration<blockThreads, FaceHandle>(context, pi);
}


template <uint32_t blockThreads>
__global__ static void remove_surplus_elements(const Context context)
{
    auto block = cooperative_groups::this_thread_block();

    const uint32_t pid = blockIdx.x;
    if (pid >= context.m_num_patches[0]) {
        return;
    }

    PatchInfo pi = context.m_patches_info[pid];

    const uint16_t num_vertices = pi.num_vertices[0];
    const uint16_t num_edges    = pi.num_edges[0];
    const uint16_t num_faces    = pi.num_faces[0];

    ShmemAllocator shrd_alloc;

    uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(3 * num_faces);
    load_async(
        block, reinterpret_cast<uint16_t*>(pi.fe), 3 * num_faces, s_fe, false);

    uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(2 * num_edges);
    load_async(
        block, reinterpret_cast<uint16_t*>(pi.ev), 2 * num_edges, s_ev, false);

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

    block.sync();

    auto tag_edges_and_vertices_through_face = [&]() {
        // mark edges that are incident to owned faces
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


    tag_edges_and_vertices_through_face();
    block.sync();

    // tag faces through edges
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
    block.sync();

    tag_edges_and_vertices_through_face();
    block.sync();


    // tag vertices through edges
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

    block.sync();
    s_vert_tag.store<blockThreads>(pi.active_mask_v);
    s_edge_tag.store<blockThreads>(pi.active_mask_e);
    s_face_tag.store<blockThreads>(pi.active_mask_f);
}

template <uint32_t blockThreads>
__inline__ __device__ void bi_assignment2(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
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
    const uint16_t*                   s_fe,
    Bitmask&                          s_patch_v,
    Bitmask&                          s_patch_e,
    Bitmask&                          s_patch_f,
    const int                         max_num_iter = 10)
{
    // initially, all vertices/edges/face belongs to the same (this) patch
    // so, we only assigne vertices/edges/faces to the other (split) patch
    // this assignment is indicated by setting a bit in s_patch_v/e/f

    __shared__ int s_seed, s_num_1_faces, s_num_boundary_faces;


    Bitmask s_boundary_f(num_faces, shrd_alloc);
    Bitmask s_boundary_e(num_edges, shrd_alloc);

    s_patch_v.reset(block);
    s_patch_e.reset(block);
    s_patch_f.reset(block);
    block.sync();

    if (threadIdx.x == 0) {
        // pick two seeds that are active and owned faces
        for (uint16_t f = 0; f < num_faces; ++f) {
            if (s_active_f(f) && s_owned_f(f)) {
                s_seed = f;
                break;
            }
        }
    }
    block.sync();

    int num_iter = 0;

    while (num_iter < max_num_iter) {
        // seeding
        if (threadIdx.x == 0) {
            s_patch_f.set(uint16_t(s_seed), true);
            s_num_1_faces = 1;
        }
        block.sync();

        // cluster seed propagation
        while (s_num_1_faces < num_faces / 2) {
            // 1st
            for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
                if (s_active_f(f) && s_patch_f(f)) {
                    const uint16_t e0(s_fe[3 * f + 0] >> 1),
                        e1(s_fe[3 * f + 1] >> 1), e2(s_fe[3 * f + 2] >> 1);
                    s_patch_e.set(e0, true);
                    s_patch_e.set(e1, true);
                    s_patch_e.set(e2, true);
                }
            }

            block.sync();

            // 2nd
            for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
                if (s_active_f(f)) {
                    const uint16_t e0(s_fe[3 * f + 0] >> 1),
                        e1(s_fe[3 * f + 1] >> 1), e2(s_fe[3 * f + 2] >> 1);
                    int sum = int(s_patch_e(e0)) + int(s_patch_e(e1)) +
                              int(s_patch_e(e2));
                    if (sum >= 1) {
                        s_patch_f.set(f, true);
                        ::atomicAdd(&s_num_1_faces, int(1));
                    }
                }
            }
            block.sync();
        }


        // interior
        // find the most interior face and set it as seed

        s_boundary_f.reset(block);
        s_boundary_e.reset(block);
        if (threadIdx.x == 0) {
            s_num_boundary_faces = 0;
        }
        block.sync();


        // set boundary faces
        for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
            if (s_patch_f(f)) {
                const uint16_t e0(s_fe[3 * f + 0] >> 1),
                    e1(s_fe[3 * f + 1] >> 1), e2(s_fe[3 * f + 2] >> 1);
                int sum = int(s_patch_e(e0)) + int(s_patch_e(e1)) +
                          int(s_patch_e(e2));
                if (sum == 1 || sum == 2) {
                    s_boundary_f.set(f, true);
                    ::atomicAdd(&s_num_boundary_faces, int(1));
                    atomicExch(&s_seed, int(f));
                }
            }
        }
        block.sync();

        while (s_num_boundary_faces != s_num_1_faces) {
            assert(s_num_boundary_faces < s_num_1_faces);
            block.sync();
            // set boundary edges
            for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
                if (s_boundary_f(f)) {
                    const uint16_t e0(s_fe[3 * f + 0] >> 1),
                        e1(s_fe[3 * f + 1] >> 1), e2(s_fe[3 * f + 2] >> 1);
                    s_boundary_e.set(e0, true);
                    s_boundary_e.set(e1, true);
                    s_boundary_e.set(e2, true);
                }
            }
            block.sync();

            // set boundary faces
            for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
                if (s_patch_f(f)) {
                    const uint16_t e0(s_fe[3 * f + 0] >> 1),
                        e1(s_fe[3 * f + 1] >> 1), e2(s_fe[3 * f + 2] >> 1);
                    if (s_boundary_e(e0) || s_boundary_e(e1) ||
                        s_boundary_e(e2)) {
                        s_boundary_f.set(f, true);
                        ::atomicAdd(&s_num_boundary_faces, int(1));
                        atomicExch(&s_seed, int(f));
                    }
                }
            }
            block.sync();
        }

        block.sync();
        s_patch_e.reset(block);
        s_patch_f.reset(block);
        block.sync();

        num_iter++;
    }

    // finally we assign vertices such that each edge assign its two vertices
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        if (s_active_e(e) && s_patch_e(e)) {
            const uint16_t v0(s_ev[2 * e + 0]), v1(s_ev[2 * e + 1]);
            s_patch_v.set(v0, true);
            s_patch_v.set(v1, true);
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
    const uint16_t*                   s_fe,
    Bitmask&                          s_patch_v,
    Bitmask&                          s_patch_e,
    Bitmask&                          s_patch_f)
{
    // assign mesh element to two partitions. the assignment partition the patch
    // into contiguous patches of almost equal size
    //
    // an element stay in the patch if its bitmask is zero

    // initially, all mesh elements stays in this patch
    s_patch_v.reset(block);
    s_patch_e.reset(block);
    s_patch_f.reset(block);
    block.sync();

    // number of faces that are assigned to 1
    __shared__ uint16_t s_num_1_faces;
    if (threadIdx.x == 0) {
        // we bootstrap the assignment by assigning a ribbon face to 1
        for (uint16_t f = 0; f < num_faces; ++f) {
            if (s_active_f(f) && !s_owned_f(f)) {
                s_patch_f.set(f);
            }
        }
        s_num_1_faces = 1;
    }
    block.sync();


    // we iterate over faces twice. First, every face atomically set its
    // three edges if the face is set. Second, every face set itself if there
    // are two edges incident to it that are set. we stop when the s_num_1_faces
    // is more than half num_faces
    while (s_num_1_faces < num_faces / 2) {

        // 1st
        for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
            if (s_active_f(f) && s_patch_f(f)) {
                const uint16_t e0(s_fe[3 * f + 0] >> 1),
                    e1(s_fe[3 * f + 1] >> 1), e2(s_fe[3 * f + 2] >> 1);
                s_patch_e.set(e0, true);
                s_patch_e.set(e1, true);
                s_patch_e.set(e2, true);
            }
        }

        block.sync();

        // 2nd
        for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
            if (s_active_f(f)) {
                const uint16_t e0(s_fe[3 * f + 0] >> 1),
                    e1(s_fe[3 * f + 1] >> 1), e2(s_fe[3 * f + 2] >> 1);

                if (s_patch_e(e0) || s_patch_e(e1) || s_patch_e(e2)) {
                    s_patch_f.set(f, true);
                    atomicAdd(&s_num_1_faces, 1);
                }
            }
        }
        block.sync();
    }


    // finally we assign vertices such that each edge assign its two vertices
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        if (s_active_e(e) && s_patch_e(e)) {
            const uint16_t v0(s_ev[2 * e + 0]), v1(s_ev[2 * e + 1]);
            s_patch_v.set(v0, true);
            s_patch_v.set(v1, true);
        }
    }
}

template <uint32_t blockThreads>
__global__ static void slice_patches(const Context  context,
                                     const uint32_t current_num_patches,
                                     const uint32_t num_faces_threshold,
                                     rxmesh::FaceAttribute<int>   f_attr,
                                     rxmesh::EdgeAttribute<int>   e_attr,
                                     rxmesh::VertexAttribute<int> v_attr)
{
    // ev, fe, active_v/e/f, owned_v/e/f, patch_v/e/f
    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    const uint32_t pid = blockIdx.x;
    if (pid >= current_num_patches) {
        return;
    }

    PatchInfo pi = context.m_patches_info[pid];

    const uint16_t num_vertices = pi.num_vertices[0];
    const uint16_t num_edges    = pi.num_edges[0];
    const uint16_t num_faces    = pi.num_faces[0];


    auto alloc_masks = [&](uint16_t        num_elements,
                           Bitmask&        owned,
                           Bitmask&        active,
                           Bitmask&        patch,
                           const uint32_t* g_owned,
                           const uint32_t* g_active) {
        owned  = Bitmask(num_elements, shrd_alloc);
        active = Bitmask(num_elements, shrd_alloc);
        patch  = Bitmask(num_elements, shrd_alloc);

        owned.reset(block);
        active.reset(block);

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


    if (num_faces >= num_faces_threshold) {
        Bitmask s_owned_v, s_owned_e, s_owned_f;
        Bitmask s_active_v, s_active_e, s_active_f;
        Bitmask s_patch_v, s_patch_e, s_patch_f;

        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(2 * num_edges);
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(pi.ev),
                           2 * num_edges,
                           s_ev,
                           false);
        uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(3 * num_faces);
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(pi.fe),
                           3 * num_faces,
                           s_fe,
                           true);

        alloc_masks(num_vertices,
                    s_owned_v,
                    s_active_v,
                    s_patch_v,
                    pi.owned_mask_v,
                    pi.active_mask_v);
        alloc_masks(num_edges,
                    s_owned_e,
                    s_active_e,
                    s_patch_e,
                    pi.owned_mask_e,
                    pi.active_mask_e);
        alloc_masks(num_faces,
                    s_owned_f,
                    s_active_f,
                    s_patch_f,
                    pi.owned_mask_f,
                    pi.active_mask_f);


        bi_assignment<blockThreads>(block,
                                    // shrd_alloc,
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
                                    s_fe,
                                    s_patch_v,
                                    s_patch_e,
                                    s_patch_f);
        block.sync();

        if (pi.patch_id == 1) {
            for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
                FaceHandle fh(pi.patch_id, f);
                if (!s_owned_f(f)) {
                    fh = pi.find<FaceHandle>(f);
                }

                if (s_patch_f(fh.local_id())) {
                    f_attr(fh) = 1;
                } else {
                    f_attr(fh) = 2;
                }
            }
            detail::for_each_face(pi, [&](const FaceHandle fh) {
                if (s_patch_f(fh.local_id())) {
                    f_attr(fh) = 1;
                } else {
                    f_attr(fh) = 2;
                }
            });

            detail::for_each_edge(pi, [&](const EdgeHandle eh) {
                if (s_patch_e(eh.local_id())) {
                    e_attr(eh) = 1;
                } else {
                    e_attr(eh) = 2;
                }
            });

            detail::for_each_vertex(pi, [&](const VertexHandle vh) {
                if (s_patch_v(vh.local_id())) {
                    v_attr(vh) = 1;
                } else {
                    v_attr(vh) = 2;
                }
            });
        }
    }
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
                    uint16_t vw0 =
                        owner_patch_info.ev[2 * e_owned.local_id() + 0].id;
                    uint16_t vw1 =
                        owner_patch_info.ev[2 * e_owned.local_id() + 1].id;
                    uint32_t pw0(e_owned.patch_id()), pw1(e_owned.patch_id());

                    get_owned_v(vw0, pw0, owner_patch_info);
                    get_owned_v(vw1, pw1, owner_patch_info);

                    if (v0 != vw0 || p0 != pw0 || v1 != vw1 || p1 != pw1) {
                        // printf("\n 7 owned = %u", patch_id);
                        ::atomicAdd(d_check, 1);
                    }
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
                    // printf("\n ribbon edge = %u, %u", patch_id, e);
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
        uint16_t*      s_fv =
            shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces[0]);
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

                        const auto fvh_global = global_vf(vh, i);

                        if (fvh_global.is_valid()) {

                            // look for the face incident to the vertex in local
                            // VF
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
                                /*printf(
                                    "\n T=%u, ribbon face = %u, f= %u, v_id= "
                                    "%u ",
                                    threadIdx.x,
                                    patch_id,
                                    f,
                                    v_id);*/
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
    if (m_patcher->m_num_patches != get_num_patches()) {
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
    }


    RXMesh::save(filename);
}


bool RXMeshDynamic::validate()
{
    bool cached_quite = this->m_quite;
    this->m_quite     = true;

    CUDA_ERROR(cudaDeviceSynchronize());

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
            return false;
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


    // every mesh element in the ribbon of a patch should be mapped to an owner
    // that lives in a different. otherwise, the ribbon element is actually
    // duplicated. We check on this here
    auto check_unique_ribbon = [&]() {
        for (uint32_t p = 0; p < get_num_patches(); ++p) {
            PatchInfo pi = m_h_patches_info[p];

            for (uint32_t v = 0; v < pi.num_vertices[0]; v++) {
                const LocalVertexT vl(v);
                if (!pi.is_deleted(vl) && !pi.is_owned(vl)) {
                    const VertexHandle vh =
                        get_owner_handle<VertexHandle>({p, vl});
                    if (vh.patch_id() == p) {
                        return false;
                    }
                }
            }


            for (uint32_t e = 0; e < pi.num_edges[0]; e++) {
                const LocalEdgeT el(e);
                if (!pi.is_deleted(el) && !pi.is_owned(el)) {
                    const EdgeHandle eh = get_owner_handle<EdgeHandle>({p, el});
                    if (eh.patch_id() == p) {
                        return false;
                    }
                }
            }


            for (uint32_t f = 0; f < pi.num_faces[0]; f++) {
                const LocalFaceT fl(f);
                if (!pi.is_deleted(fl) && !pi.is_owned(fl)) {
                    const FaceHandle fh = get_owner_handle<FaceHandle>({p, fl});
                    if (fh.patch_id() == p) {
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

    if (!check_uniqueness()) {
        RXMESH_ERROR("RXMeshDynamic::validate() check_uniqueness failed");
        success = false;
    }

    if (!check_not_owned()) {
        RXMESH_ERROR("RXMeshDynamic::validate() check_not_owned failed");
        success = false;
    }

    if (!check_unique_ribbon()) {
        RXMESH_ERROR("RXMeshDynamic::validate() check_unique_ribbon failed");
        success = false;
    }

    if (!check_ribbon()) {
        RXMESH_ERROR("RXMeshDynamic::validate() check_ribbon failed");
        success = false;
    }

    CUDA_ERROR(cudaFree(d_check));

    this->m_quite = cached_quite;

    return success;
}

void RXMeshDynamic::cleanup()
{
    constexpr uint32_t block_size = 256;
    const uint32_t     grid_size  = get_num_patches();

    CUDA_ERROR(cudaMemcpy(&this->m_max_faces_per_patch,
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

    uint32_t dyn_shmem = 2 * ShmemAllocator::default_alignment +
                         (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
                         (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

    dyn_shmem += 3 * detail::mask_num_bytes(this->m_max_vertices_per_patch) +
                 3 * ShmemAllocator::default_alignment;

    dyn_shmem += 3 * detail::mask_num_bytes(this->m_max_edges_per_patch) +
                 3 * ShmemAllocator::default_alignment;

    dyn_shmem += 3 * detail::mask_num_bytes(this->m_max_faces_per_patch) +
                 3 * ShmemAllocator::default_alignment;

    detail::hashtable_calibration<block_size>
        <<<grid_size, block_size>>>(this->m_rxmesh_context);

    detail::remove_surplus_elements<block_size>
        <<<grid_size, block_size, dyn_shmem>>>(this->m_rxmesh_context);
}

void RXMeshDynamic::slice_patches(const uint32_t num_faces_threshold,
                                  rxmesh::FaceAttribute<int>&   f_attr,
                                  rxmesh::EdgeAttribute<int>&   e_attr,
                                  rxmesh::VertexAttribute<int>& v_attr)
{
    constexpr uint32_t block_size = 256;
    const uint32_t     grid_size  = get_num_patches();


    // ev, fe
    uint32_t dyn_shmem = 2 * ShmemAllocator::default_alignment +
                         (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
                         (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

    // active_v/e/f, owned_v/e/f, patch_v/e/f
    dyn_shmem += 3 * detail::mask_num_bytes(this->m_max_vertices_per_patch) +
                 3 * ShmemAllocator::default_alignment;

    dyn_shmem += 3 * detail::mask_num_bytes(this->m_max_edges_per_patch) +
                 3 * ShmemAllocator::default_alignment;

    dyn_shmem += 3 * detail::mask_num_bytes(this->m_max_faces_per_patch) +
                 3 * ShmemAllocator::default_alignment;

    detail::slice_patches<block_size>
        <<<grid_size, block_size, dyn_shmem>>>(this->m_rxmesh_context,
                                               get_num_patches(),
                                               num_faces_threshold,
                                               f_attr,
                                               e_attr,
                                               v_attr);
}

void RXMeshDynamic::update_host()
{
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
    if (num_patches != m_num_patches) {
        RXMESH_ERROR(
            "RXMeshDynamic::update_host() does support changing number of "
            "patches in the mesh");
    }

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

    if (m_h_edge_prefix[m_num_patches] != this->m_num_edges) {
        RXMESH_ERROR(
            "RXMeshDynamic::update_host error in updating host. m_num_edges "
            "{} does not match m_h_edge_prefix calculation {}",
            this->m_num_faces,
            m_h_face_prefix[m_num_patches]);
    }

    if (m_h_face_prefix[m_num_patches] != this->m_num_faces) {
        RXMESH_ERROR(
            "RXMeshDynamic::update_host error in updating host. m_num_faces "
            "{} does not match m_h_face_prefix calculation {}",
            this->m_num_edges,
            m_h_edge_prefix[m_num_patches]);
    }

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
}

void RXMeshDynamic::update_polyscope()
{
#if USE_POLYSCOPE
    // for polyscope, we just remove the mesh and re-add it since polyscope does
    // not support changing the mesh topology
    // if (this->m_polyscope_mesh_name.find("updated") != std::string::npos) {
    // polyscope::removeSurfaceMesh(this->m_polyscope_mesh_name, true);
    //}
    this->m_polyscope_mesh_name = this->m_polyscope_mesh_name + "updated";
    this->register_polyscope();
#endif
}

}  // namespace rxmesh
