#pragma once

#include <cooperative_groups.h>

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/util/meta.h"

#include "rxmesh/attribute.h"


namespace rxmesh {

/**
 * @brief create, process, and manipulate cavities. A block would normally
 * process a single patch in which it may create more than one cavity. This
 * class creates, processes, and manipulates all cavities created by a block
 * The patch being processed by the block is referred to as P
 * A neighbor patch to P is referred to as Q
 */
template <uint32_t blockThreads, CavityOp cop>
struct Cavity
{
    __device__ __inline__ Cavity()
        : m_s_num_cavities(nullptr),
          m_s_cavity_size(nullptr),
          m_s_cavity_id_v(nullptr),
          m_s_cavity_id_e(nullptr),
          m_s_cavity_id_f(nullptr),
          m_s_cavity_edge_loop(nullptr),
          m_s_ev(nullptr),
          m_s_fe(nullptr),
          m_s_num_vertices(nullptr),
          m_s_num_edges(nullptr),
          m_s_num_faces(nullptr),
          m_s_cavity_edge_loop(nullptr)

    {
    }

    //##
    __device__ __inline__ Cavity(cooperative_groups::thread_block& block,
                                 Context&                          context,
                                 ShmemAllocator&                   shrd_alloc)
        : m_context(context)
    {
        __shared__ uint32_t patch_id;
        __shared__ uint32_t s_init_timestamp;
        __shared__ uint32_t smem[DIVIDE_UP(blockThreads, 32)];
        m_s_active_cavity_bitmask = Bitmask(blockThreads, smem);

        __shared__ uint16_t counts[3];
        m_s_num_vertices = counts + 0;
        m_s_num_edges    = counts + 1;
        m_s_num_faces    = counts + 2;

        __shared__ bool readd[1];
        m_s_readd_to_queue = readd;
        if (threadIdx.x == 0) {
            m_s_readd_to_queue[0] = false;
            patch_id              = m_context.m_patch_scheduler.pop();
            if (patch_id != INVALID32) {
                m_s_num_vertices[0] =
                    m_context.m_patches_info[patch_id].num_vertices[0];
                m_s_num_edges[0] =
                    m_context.m_patches_info[patch_id].num_edges[0];
                m_s_num_faces[0] =
                    m_context.m_patches_info[patch_id].num_faces[0];
                s_init_timestamp =
                    atomic_read(m_context.m_patches_info[patch_id].timestamp);
            }
        }
        block.sync();

        if (patch_id == INVALID32) {
            return;
        }

        m_patch_info     = m_context.m_patches_info[patch_id];
        m_init_timestamp = s_init_timestamp;


        // TODO we don't to store the cavity IDs for all elements. we can
        // optimize this based on the give CavityOp
        const uint16_t vert_cap = m_patch_info.vertices_capacity[0];
        const uint16_t edge_cap = m_patch_info.edges_capacity[0];
        const uint16_t face_cap = m_patch_info.faces_capacity[0];

        m_s_num_cavities     = shrd_alloc.alloc<int>(1);
        m_s_cavity_id_v      = shrd_alloc.alloc<uint16_t>(vert_cap);
        m_s_owner_v          = m_s_cavity_id_v;
        m_s_cavity_id_e      = shrd_alloc.alloc<uint16_t>(edge_cap);
        m_s_owner_e          = m_s_cavity_id_e;
        m_s_cavity_id_f      = shrd_alloc.alloc<uint16_t>(face_cap);
        m_s_owner_f          = m_s_cavity_id_f;
        m_s_cavity_edge_loop = shrd_alloc.alloc<uint16_t>(m_s_num_edges[0]);

        auto alloc_masks = [&](uint16_t        num_elements,
                               Bitmask&        owned,
                               Bitmask&        active,
                               Bitmask&        ownership,
                               Bitmask&        added_to_lp,
                               Bitmask&        in_cavity,
                               const uint32_t* g_owned,
                               const uint32_t* g_active) {
            owned       = Bitmask(num_elements, shrd_alloc);
            active      = Bitmask(num_elements, shrd_alloc);
            ownership   = Bitmask(num_elements, shrd_alloc);
            added_to_lp = Bitmask(num_elements, shrd_alloc);
            in_cavity   = Bitmask(num_elements, shrd_alloc);

            owned.reset(block);
            active.reset(block);
            ownership.reset(block);
            added_to_lp.reset(block);
            in_cavity.reset(block);

            detail::load_async(reinterpret_cast<const char*>(g_owned),
                               owned.num_bytes(),
                               reinterpret_cast<char*>(owned.m_bitmask),
                               false);
            detail::load_async(reinterpret_cast<const char*>(g_active),
                               active.num_bytes(),
                               reinterpret_cast<char*>(active.m_bitmask),
                               false);

            ownership.reset(block);
        };


        // vertices masks
        alloc_masks(vert_cap,
                    m_s_owned_mask_v,
                    m_s_active_mask_v,
                    m_s_ownership_change_mask_v,
                    m_s_added_to_lp_v,
                    m_s_in_cavity_v,
                    m_patch_info.owned_mask_v,
                    m_patch_info.active_mask_v);
        m_s_migrate_mask_v      = Bitmask(vert_cap, shrd_alloc);
        m_s_owned_cavity_bdry_v = Bitmask(vert_cap, shrd_alloc);
        m_s_ribbonize_v         = Bitmask(vert_cap, shrd_alloc);
        m_s_src_mask_v = Bitmask(context.m_max_num_vertices[0], shrd_alloc);
        m_s_src_connect_mask_v =
            Bitmask(context.m_max_num_vertices[0], shrd_alloc);


        // edges masks
        alloc_masks(edge_cap,
                    m_s_owned_mask_e,
                    m_s_active_mask_e,
                    m_s_ownership_change_mask_e,
                    m_s_added_to_lp_e,
                    m_s_in_cavity_e,
                    m_patch_info.owned_mask_e,
                    m_patch_info.active_mask_e);
        const uint16_t max_edge_cap = static_cast<uint16_t>(
            context.m_capacity_factor *
            static_cast<float>(context.m_max_num_edges[0]));
        m_s_src_mask_e = Bitmask(std::max(max_edge_cap, edge_cap), shrd_alloc);
        m_s_src_connect_mask_e =
            Bitmask(context.m_max_num_edges[0], shrd_alloc);

        // faces masks
        alloc_masks(face_cap,
                    m_s_owned_mask_f,
                    m_s_active_mask_f,
                    m_s_ownership_change_mask_f,
                    m_s_added_to_lp_f,
                    m_s_in_cavity_f,
                    m_patch_info.owned_mask_f,
                    m_patch_info.active_mask_f);

        m_s_patches_to_lock_mask = Bitmask(PatchStash::stash_size, shrd_alloc);
        m_s_locked_patches_mask  = Bitmask(PatchStash::stash_size, shrd_alloc);

        if (threadIdx.x == 0) {
            m_s_num_cavities[0] = 0;
        }

        init_cavity_id(vert_cap, edge_cap, face_cap);

        m_s_patches_to_lock_mask.reset(block);
        m_s_active_cavity_bitmask.set(block);
        cooperative_groups::wait(block);
        block.sync();
    }

    /**
     * @brief init cavity id arrays
     * ##
     */
    __device__ __inline__ void init_cavity_id(const uint16_t vert_cap,
                                              const uint16_t edge_cap,
                                              const uint16_t face_cap)
    {
        // TODO fix the bank conflict
        for (uint16_t v = threadIdx.x; v < vert_cap; v += blockThreads) {
            m_s_cavity_id_v[v] = INVALID16;
        }

        for (uint16_t e = threadIdx.x; e < edge_cap; e += blockThreads) {
            m_s_cavity_id_e[e] = INVALID16;
        }

        for (uint16_t f = threadIdx.x; f < face_cap; f += blockThreads) {
            m_s_cavity_id_f[f] = INVALID16;
        }
    }

    /**
     * @brief create new cavity
     * ##
     */
    template <typename HandleT>
    __device__ __inline__ void add(HandleT handle)
    {
        if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                      cop == CavityOp::VE || cop == CavityOp::VF) {
            static_assert(std::is_same_v<HandleT, VertexHandle>,
                          "Cavity::get_handle() since Cavity's template "
                          "parameter operation is Op::V/Op::VV/Op::VE/Op::VF, "
                          "get_handle() should take VertexHandle as an input");
        }

        if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                      cop == CavityOp::EE || cop == CavityOp::EF) {
            static_assert(std::is_same_v<HandleT, EdgeHandle>,
                          "Cavity::get_handle() since Cavity's template "
                          "parameter operation is Op::E/Op::EV/Op::EE/Op::EF, "
                          "get_handle() should take EdgeHandle as an input");
        }

        if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                      cop == CavityOp::FE || cop == CavityOp::FF) {
            static_assert(std::is_same_v<HandleT, FaceHandle>,
                          "Cavity::get_handle() since Cavity's template "
                          "parameter operation is Op::F/Op::FV/Op::FE/Op::FF, "
                          "get_handle() should take FaceHandle as an input");
        }

        int id = ::atomicAdd(m_s_num_cavities, 1);

        // there is no race condition in here since each thread is assigned to
        // one element
        if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                      cop == CavityOp::VE || cop == CavityOp::VF) {
            assert(m_s_active_mask_v(handle.local_id()));
            assert(m_s_owned_mask_v(handle.local_id()));
            m_s_cavity_id_v[handle.local_id()] = id;
        }

        if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                      cop == CavityOp::EE || cop == CavityOp::EF) {
            assert(m_s_active_mask_e(handle.local_id()));
            assert(m_s_owned_mask_e(handle.local_id()));
            m_s_cavity_id_e[handle.local_id()] = id;
        }


        if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                      cop == CavityOp::FE || cop == CavityOp::FF) {
            assert(m_s_active_mask_f(handle.local_id()));
            assert(m_s_owned_mask_f(handle.local_id()));
            m_s_cavity_id_f[handle.local_id()] = id;
        }
    }

    /**
     * @brief delete elements by applying the cop operation
     */
    __device__ __inline__ bool process(cooperative_groups::thread_block& block,
                                       ShmemAllocator&               shrd_alloc/*,
                                       rxmesh::VertexAttribute<int>& v_attr,
                                       rxmesh::EdgeAttribute<int>&   e_attr,
                                       rxmesh::FaceAttribute<int>    f_attr*/)
    {
        m_s_cavity_size = shrd_alloc.alloc<int>(m_s_num_cavities[0] + 1);
        for (uint16_t i = threadIdx.x; i < m_s_num_cavities[0] + 1;
             i += blockThreads) {
            m_s_cavity_size[i] = 0;
        }

        // make sure the timestamp is the same after locking the patch
        // if (!is_same_timestamp(block)) {
        //    push();
        //    return false;
        //}

        // load mesh FE and EV
        load_mesh_async(block, shrd_alloc);
        block.sync();

        // need to make sure we have the same timestamp since if it is different
        // we could just quite and save a lot of work but also since the
        // topology could have changed and now when we operate on it, we may
        // encounter errors/illegal memory read/failed assertions
        if (!is_same_timestamp(block)) {
            push();
            return false;
        }

        // Expand cavities by marking incident elements
        if constexpr (cop == CavityOp::V) {
            mark_edges_through_vertices();
            block.sync();
            mark_faces_through_edges();
            block.sync();
        }

        if constexpr (cop == CavityOp::E) {
            mark_faces_through_edges();
            block.sync();
        }

        // Repair for conflicting cavities
        deactivate_conflicting_cavities();
        block.sync();

        // Clear bitmask for elements in the (active) cavity to indicate that
        // they are deleted (but only in shared memory)
        // TODO optimize this based on CavityOp
        clear_bitmask_if_in_cavity();
        block.sync();

        // construct cavity boundary loop
        construct_cavities_edge_loop(block);
        block.sync();

        // sort each cavity edge loop
        sort_cavities_edge_loop();
        block.sync();

        // init cavity_id_v/e/f which points at the same place as owner_v/e/f
        init_cavity_id(m_patch_info.vertices_capacity[0],
                       m_patch_info.edges_capacity[0],
                       m_patch_info.faces_capacity[0]);
        block.sync();

        // cache owner_v/e/f
        load_owner();
        block.sync();

        if (!migrate(block /*, v_attr, e_attr, f_attr*/)) {
            push();
            return false;
        }

        block.sync();

        change_ownership(block);
        block.sync();

        post_migration_cleanup(block);

        if (threadIdx.x == 0) {
            m_patch_info.num_vertices[0] = m_s_num_vertices[0];
            m_patch_info.num_edges[0]    = m_s_num_edges[0];
            m_patch_info.num_faces[0]    = m_s_num_faces[0];
        }

        block.sync();
        return true;
    }


    /**
     * @brief load mesh FE and EV into shared memory
     * ##
     */
    __device__ __inline__ void load_mesh_async(
        cooperative_groups::thread_block& block,
        ShmemAllocator&                   shrd_alloc)
    {
        m_s_ev = shrd_alloc.alloc<uint16_t>(2 * m_patch_info.edges_capacity[0]);
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(m_patch_info.ev),
                           2 * m_s_num_edges[0],
                           m_s_ev,
                           false);
        m_s_fe = shrd_alloc.alloc<uint16_t>(3 * m_patch_info.faces_capacity[0]);
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(m_patch_info.fe),
                           3 * m_s_num_faces[0],
                           m_s_fe,
                           true);
    }

    /**
     * @brief propagate the cavity tag from vertices to their incident edges
     * ##
     */
    __device__ __inline__ void mark_edges_through_vertices()
    {
        for (uint16_t e = threadIdx.x; e < m_s_num_edges[0];
             e += blockThreads) {
            if (m_s_active_mask_e(e)) {

                // vertices tag
                const uint16_t v0 = m_s_ev[2 * e + 0];
                const uint16_t v1 = m_s_ev[2 * e + 1];

                const uint16_t c0 = m_s_cavity_id_v[v0];
                const uint16_t c1 = m_s_cavity_id_v[v1];

                mark_element(m_s_cavity_id_e, e, c0);
                mark_element(m_s_cavity_id_e, e, c1);
            }
        }
    }

    /**
     * @brief propagate the cavity tag from edges to their incident faces
     * ##
     */
    __device__ __inline__ void mark_faces_through_edges()
    {
        for (uint16_t f = threadIdx.x; f < m_s_num_faces[0];
             f += blockThreads) {
            if (m_s_active_mask_f(f)) {

                // edges tag
                const uint16_t e0 = m_s_fe[3 * f + 0] >> 1;
                const uint16_t e1 = m_s_fe[3 * f + 1] >> 1;
                const uint16_t e2 = m_s_fe[3 * f + 2] >> 1;

                const uint16_t c0 = m_s_cavity_id_e[e0];
                const uint16_t c1 = m_s_cavity_id_e[e1];
                const uint16_t c2 = m_s_cavity_id_e[e2];

                mark_element(m_s_cavity_id_f, f, c0);
                mark_element(m_s_cavity_id_f, f, c1);
                mark_element(m_s_cavity_id_f, f, c2);
            }
        }
    }


    /**
     * @brief deactivate the cavities that has been marked as inactivate in the
     * bitmask (m_s_active_cavity_bitmask) by reverting all mesh element ID
     * assigned to these cavities to be INVALID16
     * ##
     */
    __device__ __inline__ void deactivate_conflicting_cavities()
    {
        deactivate_conflicting_cavities(m_s_num_vertices[0], m_s_cavity_id_v);

        deactivate_conflicting_cavities(m_s_num_edges[0], m_s_cavity_id_e);

        deactivate_conflicting_cavities(m_s_num_faces[0], m_s_cavity_id_f);
    }

    /**
     * @brief revert the element cavity ID to INVALID16 if the element's cavity
     * ID is a cavity that has been marked as inactive in
     * m_s_active_cavity_bitmask
     * ##
     */
    __device__ __inline__ void deactivate_conflicting_cavities(
        const uint16_t num_elements,
        uint16_t*      element_cavity_id)
    {
        for (uint16_t i = threadIdx.x; i < num_elements; i += blockThreads) {
            const uint32_t c = element_cavity_id[i];
            if (c != INVALID16) {
                if (!m_s_active_cavity_bitmask(c)) {
                    element_cavity_id[i] = INVALID16;
                }
            }
        }
    }

    /**
     * @brief mark element and inactivate cavities if there is a conflict. Each
     * element should be marked by one cavity. In case of conflict, the cavity
     * with min id wins. If the element has been marked previously with cavity
     * of higher ID, this higher ID cavity will be deactivated. If the element
     * has been already been marked with a cavity of lower ID, the current
     * cavity (cavity_id) will be deactivated
     * This function assumes no other thread is trying to update element_id's
     * cavity ID
     * ##
     */
    __device__ __inline__ void mark_element(uint16_t*      element_cavity_id,
                                            const uint16_t element_id,
                                            const uint16_t cavity_id)
    {
        if (cavity_id != INVALID16) {
            const uint16_t prv_element_cavity_id =
                element_cavity_id[element_id];


            if (prv_element_cavity_id == cavity_id) {
                return;
            }

            if (prv_element_cavity_id == INVALID16) {
                element_cavity_id[element_id] = cavity_id;
                return;
            }

            if (prv_element_cavity_id > cavity_id) {
                // deactivate previous element cavity ID
                m_s_active_cavity_bitmask.reset(prv_element_cavity_id, true);
                element_cavity_id[element_id] = cavity_id;
            }

            if (prv_element_cavity_id < cavity_id) {
                // deactivate cavity ID
                m_s_active_cavity_bitmask.reset(cavity_id, true);
            }
        }
    }


    /**
     * @brief clear the bit corresponding to an element in the active bitmask if
     * the element is in a cavity. Apply this for vertices, edges and face
     * ##
     */
    __device__ __inline__ void clear_bitmask_if_in_cavity()
    {
        clear_bitmask_if_in_cavity(m_s_active_mask_v,
                                   m_s_in_cavity_v,
                                   m_s_cavity_id_v,
                                   m_s_num_vertices[0]);
        clear_bitmask_if_in_cavity(m_s_active_mask_e,
                                   m_s_in_cavity_e,
                                   m_s_cavity_id_e,
                                   m_s_num_edges[0]);
        clear_bitmask_if_in_cavity(m_s_active_mask_f,
                                   m_s_in_cavity_f,
                                   m_s_cavity_id_f,
                                   m_s_num_faces[0]);
    }
    /**
     * @brief clear the bit corresponding to an element in the bitmask if the
     * element is in a cavity
     * ##
     */
    __device__ __inline__ void clear_bitmask_if_in_cavity(
        Bitmask&        active_bitmask,
        Bitmask&        in_cavity,
        const uint16_t* element_cavity_id,
        const uint16_t  num_elements)
    {
        for (uint16_t b = threadIdx.x; b < num_elements; b += blockThreads) {
            if (element_cavity_id[b] != INVALID16) {
                active_bitmask.reset(b, true);
                in_cavity.set(b, true);
                assert(!active_bitmask(b));
            }
        }
    }

    /**
     * @brief construct the cavities boundary loop
     * ##
     */
    template <uint32_t itemPerThread = 5>
    __device__ __inline__ void construct_cavities_edge_loop(
        cooperative_groups::thread_block& block)
    {

        assert(itemPerThread * blockThreads >= m_s_num_faces[0]);

        // Trace faces on the border of the cavity i.e., having an edge on the
        // cavity boundary loop. These faces will add how many of their edges
        // are on the boundary loop. We then do scan and then populate the
        // boundary loop
        uint16_t local_offset[itemPerThread];

        auto index = [&](uint16_t i) {
            // return itemPerThread * threadIdx.x + i;
            return threadIdx.x + blockThreads * i;
        };

        for (uint16_t i = 0; i < itemPerThread; ++i) {
            uint16_t f = index(i);

            local_offset[i] = INVALID16;

            uint16_t face_cavity = INVALID16;
            if (f < m_s_num_faces[0]) {
                face_cavity = m_s_cavity_id_f[f];
            }

            // if the face is inside a cavity
            // we could check on if the face is deleted but we only mark faces
            // that are not deleted so no need to double check this
            if (face_cavity != INVALID16) {
                const uint16_t c0 = m_s_cavity_id_e[m_s_fe[3 * f + 0] >> 1];
                const uint16_t c1 = m_s_cavity_id_e[m_s_fe[3 * f + 1] >> 1];
                const uint16_t c2 = m_s_cavity_id_e[m_s_fe[3 * f + 2] >> 1];

                // the edge tag is supposed to be the same as the face tag
                assert(c0 == INVALID16 || c0 == face_cavity);
                assert(c1 == INVALID16 || c1 == face_cavity);
                assert(c2 == INVALID16 || c2 == face_cavity);

                // count how many edges this face contribute to the cavity
                // boundary loop
                int num_edges_on_boundary = 0;
                num_edges_on_boundary += (c0 == INVALID16);
                num_edges_on_boundary += (c1 == INVALID16);
                num_edges_on_boundary += (c2 == INVALID16);

                // it is a face on the boundary only if it has 1 or 2 edges
                // tagged with the (same) cavity id. If it is three edges, then
                // this face is in the interior of the cavity
                if (num_edges_on_boundary == 1 || num_edges_on_boundary == 2) {
                    local_offset[i] = ::atomicAdd(m_s_cavity_size + face_cavity,
                                                  num_edges_on_boundary);
                }
            }
        }
        block.sync();

        // scan
        detail::cub_block_exclusive_sum<int, blockThreads>(m_s_cavity_size,
                                                           m_s_num_cavities[0]);
        block.sync();

        for (uint16_t i = 0; i < itemPerThread; ++i) {
            if (local_offset[i] != INVALID16) {

                uint16_t f = index(i);

                const uint16_t face_cavity = m_s_cavity_id_f[f];

                int num_added = 0;

                const uint16_t e0 = m_s_fe[3 * f + 0];
                const uint16_t e1 = m_s_fe[3 * f + 1];
                const uint16_t e2 = m_s_fe[3 * f + 2];

                const uint16_t c0 = m_s_cavity_id_e[e0 >> 1];
                const uint16_t c1 = m_s_cavity_id_e[e1 >> 1];
                const uint16_t c2 = m_s_cavity_id_e[e2 >> 1];


                auto check_and_add = [&](const uint16_t c, const uint16_t e) {
                    if (c == INVALID16) {
                        uint16_t offset = m_s_cavity_size[face_cavity] +
                                          local_offset[i] + num_added;
                        m_s_cavity_edge_loop[offset] = e;
                        num_added++;
                    }
                };

                check_and_add(c0, e0);
                check_and_add(c1, e1);
                check_and_add(c2, e2);
            }
        }
        block.sync();
    }


    /**
     * @brief sort cavity edge loop
     * ##
     */
    __device__ __inline__ void sort_cavities_edge_loop()
    {

        // TODO need to increase the parallelism in this part. It should be at
        // least one warp processing one cavity
        for (uint16_t c = threadIdx.x; c < m_s_num_cavities[0];
             c += blockThreads) {

            // Specify the starting edge of the cavity before sorting everything
            // TODO this may be tuned for different CavityOp's
            static_assert(cop == CavityOp::E);

            uint16_t cavity_edge_src_vertex;
            for (uint16_t e = 0; e < m_s_num_edges[0]; ++e) {
                if (m_s_cavity_id_e[e] == c) {
                    cavity_edge_src_vertex = m_s_ev[2 * e];
                    break;
                }
            }

            const uint16_t start = m_s_cavity_size[c];
            const uint16_t end   = m_s_cavity_size[c + 1];

            for (uint16_t e = start; e < end; ++e) {
                uint32_t edge = m_s_cavity_edge_loop[e];

                assert(m_s_active_mask_e((edge >> 1)));
                if (get_cavity_vertex(c, e - start).local_id() ==
                    cavity_edge_src_vertex) {
                    uint16_t temp               = m_s_cavity_edge_loop[start];
                    m_s_cavity_edge_loop[start] = edge;
                    m_s_cavity_edge_loop[e]     = temp;
                    break;
                }
            }


            for (uint16_t e = start; e < end; ++e) {
                uint16_t edge;
                uint8_t  dir;
                Context::unpack_edge_dir(m_s_cavity_edge_loop[e], edge, dir);
                uint16_t end_vertex = m_s_ev[2 * edge + 1];
                if (dir) {
                    end_vertex = m_s_ev[2 * edge];
                }

                for (uint16_t i = e + 1; i < end; ++i) {
                    uint32_t ee = m_s_cavity_edge_loop[i] >> 1;
                    uint32_t v0 = m_s_ev[2 * ee + 0];
                    uint32_t v1 = m_s_ev[2 * ee + 1];

                    assert(m_s_active_mask_v(v0));
                    assert(m_s_active_mask_v(v1));
                    if (v0 == end_vertex || v1 == end_vertex) {
                        uint16_t temp = m_s_cavity_edge_loop[e + 1];
                        m_s_cavity_edge_loop[e + 1] = m_s_cavity_edge_loop[i];
                        m_s_cavity_edge_loop[i]     = temp;
                        break;
                    }
                }
            }
        }
    }


    /**
     * @brief apply a lambda function on each cavity to fill it in with edges
     * and then faces
     * ##
     */
    template <typename FillInT>
    __device__ __inline__ void for_each_cavity(
        cooperative_groups::thread_block& block,
        FillInT                           FillInFunc)
    {
        // TODO need to increase the parallelism in this part. It should be at
        // least one warp processing one cavity
        for (uint16_t c = threadIdx.x; c < m_s_num_cavities[0];
             c += blockThreads) {
            const uint16_t size = get_cavity_size(c);
            if (size > 0) {
                FillInFunc(c, size);
            }
        }

        block.sync();
    }

    /**
     * @brief return number of cavities in this patch
     * ##
     */
    __device__ __inline__ int get_num_cavities() const
    {
        return m_s_num_cavities[0];
    }

    /**
     * @brief return the size of the c-th cavity. The size is the number of
     * edges surrounding the cavity
     * ##
     */
    __device__ __inline__ uint16_t get_cavity_size(uint16_t c) const
    {
        return m_s_cavity_size[c + 1] - m_s_cavity_size[c];
    }

    /**
     * @brief get an edge handle to the i-th edges in the c-th cavity
     * ##
     */
    __device__ __inline__ DEdgeHandle get_cavity_edge(uint16_t c,
                                                      uint16_t i) const
    {
        assert(c < m_s_num_cavities[0]);
        assert(i < get_cavity_size(c));
        return DEdgeHandle(m_patch_info.patch_id,
                           m_s_cavity_edge_loop[m_s_cavity_size[c] + i]);
    }


    /**
     * @brief get a vertex handle to the i-th vertex in the c-th cavity
     * ##
     */
    __device__ __inline__ VertexHandle get_cavity_vertex(uint16_t c,
                                                         uint16_t i) const
    {
        assert(c < m_s_num_cavities[0]);
        assert(i < get_cavity_size(c));

        uint16_t edge;
        flag_t   dir;
        Context::unpack_edge_dir(
            m_s_cavity_edge_loop[m_s_cavity_size[c] + i], edge, dir);

        const uint16_t v0 = m_s_ev[2 * edge];
        const uint16_t v1 = m_s_ev[2 * edge + 1];

        assert(m_s_active_mask_v(v0));
        assert(m_s_active_mask_v(v1));
        return VertexHandle(m_patch_info.patch_id, ((dir == 0) ? v0 : v1));
    }

    /**
     * @brief should be called by a single thread
     * ##
     */
    __device__ __inline__ VertexHandle add_vertex()
    {
        // First try to reuse a vertex in the cavity or a deleted vertex
        uint16_t v_id = add_element(m_s_active_mask_v, m_s_num_vertices[0]);

        if (v_id == INVALID16) {
            // if this fails, then add a new vertex to the mesh
            v_id = atomicAdd(m_s_num_vertices, 1);
            assert(v_id < m_patch_info.vertices_capacity[0]);
        }

        assert(!m_s_active_mask_v(v_id));
        // m_s_active_mask_v.set(v_id, true);
        m_s_owned_mask_v.set(v_id, true);
        return {m_patch_info.patch_id, v_id};
    }


    /**
     * @brief should be called by a single thread
     * ##
     */
    __device__ __inline__ DEdgeHandle add_edge(const VertexHandle src,
                                               const VertexHandle dest)
    {
        assert(src.patch_id() == m_patch_info.patch_id);
        assert(dest.patch_id() == m_patch_info.patch_id);

        // First try to reuse an edge in the cavity or a deleted edge
        uint16_t e_id = add_element(m_s_active_mask_e, m_s_num_edges[0]);
        if (e_id == INVALID16) {
            // if this fails, then add a new edge to the mesh
            e_id = atomicAdd(m_s_num_edges, 1);
            assert(e_id < m_patch_info.edges_capacity[0]);
        }

        assert(m_s_active_mask_v(src.local_id()));
        assert(m_s_active_mask_v(dest.local_id()));

        m_s_ev[2 * e_id + 0] = src.local_id();
        m_s_ev[2 * e_id + 1] = dest.local_id();
        // m_s_active_mask_e.set(e_id, true);
        m_s_owned_mask_e.set(e_id, true);
        return {m_patch_info.patch_id, e_id, 0};
    }


    /**
     * @brief should be called by a single thread
     * ##
     */
    __device__ __inline__ FaceHandle add_face(const DEdgeHandle e0,
                                              const DEdgeHandle e1,
                                              const DEdgeHandle e2)
    {
        assert(e0.patch_id() == m_patch_info.patch_id);
        assert(e1.patch_id() == m_patch_info.patch_id);
        assert(e2.patch_id() == m_patch_info.patch_id);

        // First try to reuse a face in the cavity or a deleted face
        uint16_t f_id = add_element(m_s_active_mask_f, m_s_num_faces[0]);

        if (f_id == INVALID16) {
            // if this fails, then add a new face to the mesh
            f_id = atomicAdd(m_s_num_faces, 1);
            assert(f_id < m_patch_info.faces_capacity[0]);
        }

        m_s_fe[3 * f_id + 0] = e0.local_id();
        m_s_fe[3 * f_id + 1] = e1.local_id();
        m_s_fe[3 * f_id + 2] = e2.local_id();


        assert(m_s_active_mask_e(e0.get_edge_handle().local_id()));
        assert(m_s_active_mask_e(e1.get_edge_handle().local_id()));
        assert(m_s_active_mask_e(e2.get_edge_handle().local_id()));

        // m_s_active_mask_f.set(f_id, true);
        m_s_owned_mask_f.set(f_id, true);

        return {m_patch_info.patch_id, f_id};
    }


    /**
     * @brief store owned and active bitmasks to global memory
     * @return
     */
    __device__ __inline__ void store_bitmasks()
    {
        detail::store<blockThreads>(m_s_owned_mask_v.m_bitmask,
                                    DIVIDE_UP(m_s_num_vertices[0], 32),
                                    m_patch_info.owned_mask_v);

        detail::store<blockThreads>(m_s_active_mask_v.m_bitmask,
                                    DIVIDE_UP(m_s_num_vertices[0], 32),
                                    m_patch_info.active_mask_v);

        detail::store<blockThreads>(m_s_owned_mask_e.m_bitmask,
                                    DIVIDE_UP(m_s_num_edges[0], 32),
                                    m_patch_info.owned_mask_e);

        detail::store<blockThreads>(m_s_active_mask_e.m_bitmask,
                                    DIVIDE_UP(m_s_num_edges[0], 32),
                                    m_patch_info.active_mask_e);

        detail::store<blockThreads>(m_s_owned_mask_f.m_bitmask,
                                    DIVIDE_UP(m_s_num_faces[0], 32),
                                    m_patch_info.owned_mask_f);

        detail::store<blockThreads>(m_s_active_mask_f.m_bitmask,
                                    DIVIDE_UP(m_s_num_faces[0], 32),
                                    m_patch_info.active_mask_f);
    }

    /**
     * @brief cleanup by moving data from shared memory to global memory
     */
    __device__ __inline__ void cleanup(cooperative_groups::thread_block& block)
    {
        // cleanup the hashtable by removing the vertices/edges/faces that has
        // changed their ownership to be in this patch (p) and thus should not
        // be in the hashtable
        for (uint32_t vp = threadIdx.x; vp < m_s_num_vertices[0];
             vp += blockThreads) {
            if (m_s_ownership_change_mask_v(vp)) {
                m_s_readd_to_queue[0] = true;
                m_patch_info.lp_v.remove(vp);
            }
        }

        for (uint32_t ep = threadIdx.x; ep < m_s_num_edges[0];
             ep += blockThreads) {
            if (m_s_ownership_change_mask_e(ep)) {
                m_s_readd_to_queue[0] = true;
                m_patch_info.lp_e.remove(ep);
            }
        }

        for (uint32_t fp = threadIdx.x; fp < m_s_num_faces[0];
             fp += blockThreads) {
            if (m_s_ownership_change_mask_f(fp)) {
                m_s_readd_to_queue[0] = true;
                m_patch_info.lp_f.remove(fp);
            }
        }

        ::atomicMax(m_context.m_max_num_vertices, m_s_num_vertices[0]);
        ::atomicMax(m_context.m_max_num_edges, m_s_num_edges[0]);
        ::atomicMax(m_context.m_max_num_faces, m_s_num_faces[0]);

        detail::store<blockThreads>(
            m_s_ev,
            2 * m_s_num_edges[0],
            reinterpret_cast<uint16_t*>(m_patch_info.ev));

        detail::store<blockThreads>(
            m_s_fe,
            3 * m_s_num_faces[0],
            reinterpret_cast<uint16_t*>(m_patch_info.fe));

        store_bitmasks();

        block.sync();

        // readd the patch to the queue if there is ownership change
        if (threadIdx.x == 0 && m_s_readd_to_queue[0]) {
            m_context.m_patch_scheduler.push(m_patch_info.patch_id);
        }

        unlock_patches_and_update_timestamp(block);

        block.sync();
    }


    /**
     * @brief update an attribute such that it can be used after the topology
     * changes
     */
    template <typename AttributeT>
    __device__ __inline__ void update_attributes(
        cooperative_groups::thread_block& block,
        AttributeT&                       attribute)
    {
        using HandleT = typename AttributeT::HandleType;
        using Type    = typename AttributeT::Type;

        const uint32_t p = m_patch_info.patch_id;


        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            for (uint16_t vp = threadIdx.x; vp < m_s_num_vertices[0];
                 vp += blockThreads) {
                if (m_s_ownership_change_mask_v(vp)) {
                    assert(m_s_owned_mask_v(vp));
                    assert(m_s_active_mask_v(vp) || m_s_in_cavity_v(vp));

                    const HandleT handle =
                        convert_to_handle<HandleT>(m_s_owner_v[vp]);

                    assert(handle.patch_id() != p);
                    assert(handle.patch_id() != INVALID32);
                    assert(handle.local_id() != INVALID16);

                    const uint32_t num_attr = attribute.get_num_attributes();
                    for (uint32_t attr = 0; attr < num_attr; ++attr) {
                        attribute(m_patch_info.patch_id, vp, attr) =
                            attribute(handle, attr);
                    }
                }
            }
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            for (uint16_t ep = threadIdx.x; ep < m_s_num_edges[0];
                 ep += blockThreads) {
                if (m_s_ownership_change_mask_e(ep)) {
                    assert(m_s_owned_mask_e(ep));
                    assert(m_s_active_mask_e(ep) || m_s_in_cavity_e(ep));

                    const HandleT handle =
                        convert_to_handle<HandleT>(m_s_owner_e[ep]);

                    assert(handle.patch_id() != p);
                    assert(handle.patch_id() != INVALID32);
                    assert(handle.local_id() != INVALID16);

                    const uint32_t num_attr = attribute.get_num_attributes();
                    for (uint32_t attr = 0; attr < num_attr; ++attr) {
                        attribute(m_patch_info.patch_id, ep, attr) =
                            attribute(handle, attr);
                    }
                }
            }
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            for (uint16_t fp = threadIdx.x; fp < m_s_num_faces[0];
                 fp += blockThreads) {
                if (m_s_ownership_change_mask_f(fp)) {
                    assert(m_s_owned_mask_f(fp));
                    assert(m_s_active_mask_f(fp) || m_s_in_cavity_f(fp));

                    const HandleT handle =
                        convert_to_handle<HandleT>(m_s_owner_f[fp]);

                    assert(handle.patch_id() != p);
                    assert(handle.patch_id() != INVALID32);
                    assert(handle.local_id() != INVALID16);

                    const uint32_t num_attr = attribute.get_num_attributes();
                    for (uint32_t attr = 0; attr < num_attr; ++attr) {
                        attribute(m_patch_info.patch_id, fp, attr) =
                            attribute(handle, attr);
                    }
                }
            }
        }

        block.sync();
    }

    /**
     * @brief find the index of the next element to add. First search within the
     * cavity and find the first element that has its cavity set to cavity_id.
     * If nothing found, search for the first element that has its bitmask set
     * to 0.
     * ##
     */
    __device__ __inline__ uint16_t add_element(Bitmask        active_bitmask,
                                               const uint16_t num_elements)
    {

        /*for (uint16_t i = 0; i < num_elements; ++i) {
            if (atomicCAS(element_cavity_id + i, cavity_id, INVALID16) ==
                cavity_id) {
                return i;
            }
        }*/

        for (uint16_t i = 0; i < num_elements; ++i) {
            if (active_bitmask.try_set(i)) {
                return i;
            }
        }

        return INVALID16;
    }

    //##
    template <typename HandleT>
    __device__ __inline void load_owner(const uint32_t patch_id,
                                        const uint16_t num_elements,
                                        uint16_t*      s_owner)
    {
        using LocalT = typename HandleT::LocalT;

        for (uint16_t v = threadIdx.x; v < num_elements; v += blockThreads) {
            const LocalT vl(v);
            if (!m_patch_info.is_deleted(vl) && !m_patch_info.is_owned(vl)) {


                LPPair lp = m_patch_info.get_lp<HandleT>().find(v);

                assert(!lp.is_sentinel());

                uint32_t owner = m_patch_info.patch_stash.get_patch(lp);

                assert(!m_context.m_patches_info[owner].is_deleted(
                    LocalT(lp.local_id_in_owner_patch())));


                while (!m_context.m_patches_info[owner].is_owned(
                    LocalT(lp.local_id_in_owner_patch()))) {

                    lp = m_context.m_patches_info[owner].get_lp<HandleT>().find(
                        lp.local_id_in_owner_patch());

                    assert(!lp.is_sentinel());

                    owner =
                        m_context.m_patches_info[owner].patch_stash.get_patch(
                            lp);

                    assert(!m_context.m_patches_info[owner].is_deleted(
                        LocalT(lp.local_id_in_owner_patch())));
                }


                assert(m_patch_info.patch_stash.get_patch(lp) != INVALID32);
                assert(owner != patch_id);

                s_owner[v] = lp.value();
            }
        }
    }

    //##
    __device__ __inline void load_owner()
    {
        const uint32_t p = m_patch_info.patch_id;
        load_owner<VertexHandle>(p, m_s_num_vertices[0], m_s_owner_v);
        load_owner<EdgeHandle>(p, m_s_num_edges[0], m_s_owner_e);
        load_owner<FaceHandle>(p, m_s_num_faces[0], m_s_owner_f);
    }


    /**
     * @brief change vertices, edges, and faces ownership as marked in
     * m_s_ownership_change_mask
     */
    __device__ __inline__ void change_ownership(
        cooperative_groups::thread_block& block)
    {
        change_ownership<VertexHandle>(block,
                                       m_s_num_vertices[0],
                                       m_s_ownership_change_mask_v,
                                       m_s_owner_v,
                                       m_s_owned_mask_v);

        change_ownership<EdgeHandle>(block,
                                     m_s_num_edges[0],
                                     m_s_ownership_change_mask_e,
                                     m_s_owner_e,
                                     m_s_owned_mask_e);

        change_ownership<FaceHandle>(block,
                                     m_s_num_faces[0],
                                     m_s_ownership_change_mask_f,
                                     m_s_owner_f,
                                     m_s_owned_mask_f);
    }

    /**
     * @brief change ownership for mesh elements of type HandleT marked in
     * ownership_change. We can remove these mesh elements from the
     * hashtable, but we delay this (do it in cleanup) since we need to get
     * these mesh elements' original owner patch in update_attributes()
     */
    template <typename HandleT>
    __device__ __inline__ void change_ownership(
        cooperative_groups::thread_block& block,
        const uint16_t                    num_elements,
        const Bitmask&                    s_ownership_change,
        const uint16_t*                   s_owner,
        Bitmask&                          s_owned_bitmask)
    {
        for (uint32_t vp = threadIdx.x; vp < num_elements; vp += blockThreads) {

            if (s_ownership_change(vp)) {
                // get handle to the owner
                // we don't check if the element is deleted since vp could have
                // just been added to this patch and thus it is not active in
                // global memory

                assert(!m_patch_info.is_owned(HandleT::LocalT(vp)));

                const HandleT h = convert_to_handle<HandleT>(s_owner[vp]);

                assert(h.patch_id() != INVALID32);
                assert(h.local_id() != INVALID16);

                const uint32_t q  = h.patch_id();
                const uint16_t vq = h.local_id();

                // set the bitmask of this element in shared memory
                s_owned_bitmask.set(vp, true);

                // m_patch_info.get_lp<HandleT>().remove(vp);

                assert(q != m_patch_info.patch_id);
                assert(!m_context.m_patches_info[q].is_deleted(
                    HandleT::LocalT(vq)));

                assert(
                    m_context.m_patches_info[q].is_owned(HandleT::LocalT(vq)));

                // add this patch (p) to the owner's patch stash
                const uint8_t stash_id =
                    m_context.m_patches_info[q].patch_stash.insert_patch(
                        m_patch_info.patch_id);

                // clear the bitmask of the owner's patch
                detail::bitmask_clear_bit(
                    vq,
                    m_context.m_patches_info[q].get_owned_mask<HandleT>(),
                    true);

                // add an LP entry in the owner's patch
                LPPair lp(vq, vp, stash_id);
                if (!m_context.m_patches_info[q].get_lp<HandleT>().insert(lp)) {
                    assert(false);
                }
            }
        }
    }


    /**
     * @brief migrate edges and face incident to vertices in the bitmask to this
     * m_patch_info from a neighbor_patch
     * ##
     */
    __device__ __inline__ bool migrate(
        cooperative_groups::thread_block& block);

    /**
     * @brief unlock/release lock for the patches stored in
     * m_s_patches_to_lock_mask. This functionally additionally update the
     * timestamp of the locked patches and this patch
     */
    __device__ __inline__ bool unlock_patches_and_update_timestamp(
        cooperative_groups::thread_block& block)
    {
        if (threadIdx.x == 0) {
            for (uint8_t i = 0; i < PatchStash::stash_size; ++i) {
                if (m_s_locked_patches_mask(i)) {
                    uint32_t p = m_patch_info.patch_stash.get_patch(i);
                    m_context.m_patches_info[p].update_timestamp();
                    m_context.m_patches_info[p].lock.release_lock();
                }
            }

            m_patch_info.update_timestamp();
            m_patch_info.lock.release_lock();
        }
    }

    /**
     * @brief push to the current patch to the scheduler
     * @return
     * ##
     */
    __device__ __inline__ void push()
    {
        if (threadIdx.x == 0) {
            m_context.m_patch_scheduler.push(m_patch_info.patch_id);
        }
    }


    /**
     * @brief given a neighbor patch (q), migrate vertices (and edges and faces
     * connected to these vertices) marked in migrate_mask_v to the patch
     * used by this cavity (p)
     * ##
     */
    __device__ __inline__ bool migrate_from_patch(
        cooperative_groups::thread_block& block,
        const uint32_t                    q,
        const Bitmask&                    migrate_mask_v,
        const bool                        change_ownership);

    /**
     * @brief give a neighbor patch q and a vertex in it q_vertex, find the copy
     * of q_vertex in this patch. If it does not exist, create such a copy.
     * ##
     */
    template <typename FuncT>
    __device__ __inline__ LPPair migrate_vertex(
        const uint32_t q,
        const uint16_t q_num_vertices,
        const uint16_t q_vertex,
        const bool     require_ownership_change,
        PatchInfo&     q_patch_info,
        FuncT          should_migrate)
    {
        LPPair ret;
        if (q_vertex < q_num_vertices &&
            !q_patch_info.is_deleted(LocalVertexT(q_vertex))) {

            if (should_migrate(q_vertex)) {
                uint16_t vq = q_vertex;
                uint32_t o  = q;
                uint16_t vp = find_copy_vertex(vq, o);

                assert(
                    !m_context.m_patches_info[o].is_deleted(LocalVertexT(vq)));
                assert(m_context.m_patches_info[o].is_owned(LocalVertexT(vq)));

                if (vp == INVALID16) {

                    vp = atomicAdd(m_s_num_vertices, 1u);

                    assert(vp < m_patch_info.vertices_capacity[0]);

                    // activate the vertex in the bit mask
                    m_s_active_mask_v.set(vp, true);

                    // since it is owned by some other patch
                    m_s_owned_mask_v.reset(vp, true);

                    // mark that we have added vp to the hashtable
                    m_s_added_to_lp_v.set(vp, true);

                    // insert the patch in the patch stash and return its
                    // id in the stash
                    const uint8_t owner_stash_id =
                        m_patch_info.patch_stash.insert_patch(o);
                    assert(owner_stash_id != INVALID8);
                    ret = LPPair(vp, vq, owner_stash_id);

                    m_s_patches_to_lock_mask.set(owner_stash_id, true);
                } else if (o != q && o != m_patch_info.patch_id) {
                    uint8_t st = m_patch_info.patch_stash.find_patch_index(o);
                    assert(st != INVALID8);
                    m_s_patches_to_lock_mask.set(st, true);
                }

                if (require_ownership_change && !m_s_owned_mask_v(vp)) {
                    m_s_ownership_change_mask_v.set(vp, true);
                    if (!ret.is_sentinel()) {
                        m_s_owner_v[vp] = ret.value();
                    }
                }
            }
        }
        return ret;
    }

    /**
     * @brief give a neighbor patch q and an edge in it q_edge, find the copy
     * of q_edge in this patch. If it does not exist, create such a copy.
     * ##
     */
    template <typename FuncT>
    __device__ __inline__ LPPair migrate_edge(
        const uint32_t q,
        const uint16_t q_num_edges,
        const uint16_t q_edge,
        const bool     require_ownership_change,
        PatchInfo&     q_patch_info,
        FuncT          should_migrate)
    {
        LPPair ret;

        if (q_edge < q_num_edges &&
            !q_patch_info.is_deleted(LocalEdgeT(q_edge))) {

            // edge v0q--v1q where o0 (defined below) is owner
            // patch of v0q and o1 (defined below) is owner
            // patch for v1q
            uint16_t v0q = q_patch_info.ev[2 * q_edge + 0].id;
            uint16_t v1q = q_patch_info.ev[2 * q_edge + 1].id;

            if (should_migrate(q_edge, v0q, v1q)) {

                // check on if e already exist in p
                uint16_t eq = q_edge;
                uint32_t o  = q;
                uint16_t ep = find_copy_edge(eq, o);

                assert(!m_context.m_patches_info[o].is_deleted(LocalEdgeT(eq)));
                assert(m_context.m_patches_info[o].is_owned(LocalEdgeT(eq)));

                if (ep == INVALID16) {
                    ep = atomicAdd(m_s_num_edges, 1u);
                    assert(ep < m_patch_info.edges_capacity[0]);


                    // We assume that the owner patch is q and will
                    // fix this later
                    uint32_t o0(q), o1(q);

                    // vq -> mapped to its local index in owner
                    // patch o-> mapped to the owner patch vp->
                    // mapped to the corresponding local index in p
                    uint16_t v0p = find_copy_vertex(v0q, o0);
                    uint16_t v1p = find_copy_vertex(v1q, o1);

                    assert(!m_context.m_patches_info[o0].is_deleted(
                        LocalVertexT(v0q)));
                    assert(m_context.m_patches_info[o0].is_owned(
                        LocalVertexT(v0q)));

                    assert(!m_context.m_patches_info[o1].is_deleted(
                        LocalVertexT(v1q)));
                    assert(m_context.m_patches_info[o1].is_owned(
                        LocalVertexT(v1q)));


                    // since any vertex in m_s_src_mask_v has been
                    // added already to p, then we should find the
                    // copy otherwise there is something wrong
                    assert(v0p != INVALID16);
                    assert(v1p != INVALID16);


                    m_s_ev[2 * ep + 0] = v0p;
                    m_s_ev[2 * ep + 1] = v1p;

                    // activate the edge in the bitmask
                    m_s_active_mask_e.set(ep, true);

                    // since it is owned by some other patch
                    m_s_owned_mask_e.reset(ep, true);

                    // mark that we have added ep to the hashtable
                    m_s_added_to_lp_e.set(ep, true);

                    const uint8_t owner_stash_id =
                        m_patch_info.patch_stash.insert_patch(o);
                    assert(owner_stash_id != INVALID8);
                    ret = LPPair(ep, eq, owner_stash_id);

                    m_s_patches_to_lock_mask.set(owner_stash_id, true);
                } else if (o != q && o != m_patch_info.patch_id) {
                    uint8_t st = m_patch_info.patch_stash.find_patch_index(o);
                    assert(st != INVALID8);
                    m_s_patches_to_lock_mask.set(st, true);
                }

                if (require_ownership_change && !m_s_owned_mask_e(ep)) {
                    m_s_ownership_change_mask_e.set(ep, true);
                    if (!ret.is_sentinel()) {
                        m_s_owner_e[ep] = ret.value();
                    }
                }
            }
        }


        return ret;
    }


    /**
     * @brief give a neighbor patch q and a face in it q_face, find the copy
     * of q_face in this patch. If it does not exist, create such a copy.
     * ##
     */
    template <typename FuncT>
    __device__ __inline__ LPPair migrate_face(
        const uint32_t q,
        const uint16_t q_num_faces,
        const uint16_t q_face,
        const bool     require_ownership_change,
        PatchInfo&     q_patch_info,
        FuncT          should_migrate)
    {
        LPPair ret;

        if (q_face < q_num_faces &&
            !q_patch_info.is_deleted(LocalFaceT(q_face))) {

            uint16_t e0q, e1q, e2q;
            flag_t   d0, d1, d2;
            Context::unpack_edge_dir(
                q_patch_info.fe[3 * q_face + 0].id, e0q, d0);
            Context::unpack_edge_dir(
                q_patch_info.fe[3 * q_face + 1].id, e1q, d1);
            Context::unpack_edge_dir(
                q_patch_info.fe[3 * q_face + 2].id, e2q, d2);

            // If any of these three edges are participant in
            // the src bitmask
            if (should_migrate(q_face, e0q, e1q, e2q)) {

                // check on if e already exist in p
                uint16_t fq = q_face;
                uint32_t o  = q;
                uint16_t fp = find_copy_face(fq, o);


                assert(!m_context.m_patches_info[o].is_deleted(LocalFaceT(fq)));
                assert(m_context.m_patches_info[o].is_owned(LocalFaceT(fq)));

                if (fp == INVALID16) {
                    fp = atomicAdd(m_s_num_faces, 1u);

                    assert(fp < m_patch_info.faces_capacity[0]);

                    uint32_t o0(q), o1(q), o2(q);

                    // eq -> mapped it to its local index in owner
                    // patch o-> mapped to the owner patch ep->
                    // mapped to the corresponding local index in p
                    const uint16_t e0p = find_copy_edge(e0q, o0);
                    const uint16_t e1p = find_copy_edge(e1q, o1);
                    const uint16_t e2p = find_copy_edge(e2q, o2);

                    assert(!m_context.m_patches_info[o0].is_deleted(
                        LocalEdgeT(e0q)));
                    assert(
                        m_context.m_patches_info[o0].is_owned(LocalEdgeT(e0q)));

                    assert(!m_context.m_patches_info[o1].is_deleted(
                        LocalEdgeT(e1q)));
                    assert(
                        m_context.m_patches_info[o1].is_owned(LocalEdgeT(e1q)));

                    assert(!m_context.m_patches_info[o2].is_deleted(
                        LocalEdgeT(e2q)));
                    assert(
                        m_context.m_patches_info[o2].is_owned(LocalEdgeT(e2q)));

                    // since any edge in m_s_src_mask_e has been
                    // added already to p, then we should find the
                    // copy otherwise there is something wrong
                    assert(e0p != INVALID16);
                    assert(e1p != INVALID16);
                    assert(e2p != INVALID16);

                    m_s_fe[3 * fp + 0] = (e0p << 1) | d0;
                    m_s_fe[3 * fp + 1] = (e1p << 1) | d1;
                    m_s_fe[3 * fp + 2] = (e2p << 1) | d2;

                    // activate the face in the bitmask
                    m_s_active_mask_f.set(fp, true);

                    // since it is owned by some other patch
                    m_s_owned_mask_f.reset(fp, true);

                    // mark that we have added fp to the hashtable
                    m_s_added_to_lp_f.set(fp, true);

                    const uint8_t owner_stash_id =
                        m_patch_info.patch_stash.insert_patch(o);
                    assert(owner_stash_id != INVALID8);
                    ret = LPPair(fp, fq, owner_stash_id);

                    m_s_patches_to_lock_mask.set(owner_stash_id, true);
                } else if (o != q && o != m_patch_info.patch_id) {
                    uint8_t st = m_patch_info.patch_stash.find_patch_index(o);
                    assert(st != INVALID8);
                    m_s_patches_to_lock_mask.set(st, true);
                }

                if (require_ownership_change && !m_s_owned_mask_f(fp)) {
                    m_s_ownership_change_mask_f.set(fp, true);
                    if (!ret.is_sentinel()) {
                        m_s_owner_f[fp] = ret.value();
                    }
                }
            }
        }

        return ret;
    }


    /**
     * @brief cleanup neighbor patches after migration
     */
    __device__ __inline__ void post_migration_cleanup(
        cooperative_groups::thread_block& block)
    {

        // we prepare vertex/edge/face bitmask for elements in P such that if
        // these elements are seen in another patch q, it means that we can
        // safely delete them from q

        // we are reusing these bitmask since we no longer need it
        // here we only change their names to improve readability
        assert(m_s_migrate_mask_v.m_size >= m_s_num_vertices[0]);

        Bitmask vertex_incident_to_not_owned_face =
            Bitmask(m_s_migrate_mask_v.m_size, m_s_migrate_mask_v.m_bitmask);

        // here boundary vertex means a vertex incident to not-owed face
        assert(m_s_owned_cavity_bdry_v.m_size >= m_s_num_vertices[0]);
        Bitmask vertex_incident_to_boundary_vertex = Bitmask(
            m_s_owned_cavity_bdry_v.m_size, m_s_owned_cavity_bdry_v.m_bitmask);

        // means that one end of the edge is a boundary vertex
        assert(m_s_src_mask_e.m_size >= m_s_num_edges[0]);
        Bitmask edge_incident_to_boundary_vertex =
            Bitmask(m_s_src_mask_e.m_size, m_s_src_mask_e.m_bitmask);

        // means that one of the face's three vertices is a boundary vertex
        assert(m_s_src_connect_mask_e.m_size >= m_s_num_faces[0]);
        Bitmask face_incident_to_boundary_vertex = Bitmask(
            m_s_src_connect_mask_e.m_size, m_s_src_connect_mask_e.m_bitmask);

        classify_elements_for_post_migtation_cleanup(
            block,
            vertex_incident_to_not_owned_face,
            vertex_incident_to_boundary_vertex,
            edge_incident_to_boundary_vertex,
            face_incident_to_boundary_vertex);

        for (uint32_t p = 0; p < PatchStash::stash_size; ++p) {
            const uint32_t q = m_patch_info.patch_stash.get_patch(p);
            if (q != INVALID32) {
                auto q_patch_info = m_context.m_patches_info[q];
                post_migration_cleanup<FaceHandle>(
                    block,
                    m_patch_info.patch_id,
                    q,
                    q_patch_info,
                    m_s_in_cavity_f,
                    face_incident_to_boundary_vertex);
                post_migration_cleanup<EdgeHandle>(
                    block,
                    m_patch_info.patch_id,
                    q,
                    q_patch_info,
                    m_s_in_cavity_e,
                    edge_incident_to_boundary_vertex);
                post_migration_cleanup<VertexHandle>(
                    block,
                    m_patch_info.patch_id,
                    q,
                    q_patch_info,
                    m_s_in_cavity_v,
                    vertex_incident_to_boundary_vertex);
            }
        }
    }
    /**
     * @brief clean up patch q from any elements that resides in p's cavity
     * @param q neighbor patch to cleanup
     */
    template <typename HandleT>
    __device__ __inline__ void post_migration_cleanup(
        cooperative_groups::thread_block& block,
        const uint32_t                    p,
        const uint32_t                    q,
        PatchInfo                         q_patch_info,
        const Bitmask&                    in_cavity,
        const Bitmask&                    p_flag)
    {

        using LocalT = typename HandleT::LocalT;

        uint16_t q_num_elements = q_patch_info.get_num_elements<HandleT>()[0];

        for (uint16_t v = threadIdx.x; v < q_num_elements; v += blockThreads) {
            auto local = LocalT(v);
            if (!q_patch_info.is_deleted(local) &&
                !q_patch_info.is_owned(local)) {

                HandleT owner = get_owner_handle<HandleT>(p, q_patch_info, v);

                if (owner.patch_id() == p) {
                    if (in_cavity(owner.local_id()) ||
                        !p_flag(owner.local_id())) {
                        detail::bitmask_clear_bit(
                            v, q_patch_info.get_active_mask<HandleT>(), true);
                        // TODO we now don't remove things for the hashtable
                        // we can do it at the end where if an element is
                        // deleted (by reading its bitmask) we delete it from
                        // the hashtable which is a patch-local operation and
                        // can be done at the end by launching a seperate kernel
                        // q_patch_info.get_lp<HandleT>().remove(v);
                    }
                }
            }
        }
    }


    /**
     * @brief similar to Context::get_owner_handle() but using shared memory
     * bitmask for
     */
    template <typename HandleT>
    __device__ __inline__ HandleT get_owner_handle(const uint32_t  p,
                                                   const PatchInfo patch_info,
                                                   const uint16_t  lid,
                                                   const bool sh_owned = true)
    {
        using LocalT = typename HandleT::LocalT;

        LPPair lp = patch_info.get_lp<HandleT>().find(lid);
        assert(!lp.is_sentinel());
        uint32_t owner = patch_info.patch_stash.get_patch(lp);

        while (true) {
            if (owner == p) {
                if constexpr (std::is_same_v<HandleT, VertexHandle>) {
                    assert(m_s_in_cavity_v(lp.local_id_in_owner_patch()) ||
                           m_s_active_mask_v(lp.local_id_in_owner_patch()));

                    if (m_s_owned_mask_v(lp.local_id_in_owner_patch())) {
                        break;
                    } else {
                        return convert_to_handle<HandleT>(
                            m_s_owner_v[lp.local_id_in_owner_patch()]);
                    }
                }
                if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
                    assert(m_s_in_cavity_e(lp.local_id_in_owner_patch()) ||
                           m_s_active_mask_e(lp.local_id_in_owner_patch()));


                    if (m_s_owned_mask_e(lp.local_id_in_owner_patch())) {
                        break;
                    } else {
                        return convert_to_handle<HandleT>(
                            m_s_owner_e[lp.local_id_in_owner_patch()]);
                    }
                }
                if constexpr (std::is_same_v<HandleT, FaceHandle>) {
                    assert(m_s_in_cavity_f(lp.local_id_in_owner_patch()) ||
                           m_s_active_mask_f(lp.local_id_in_owner_patch()));

                    if (m_s_owned_mask_f(lp.local_id_in_owner_patch())) {
                        break;
                    } else {
                        return convert_to_handle<HandleT>(
                            m_s_owner_f[lp.local_id_in_owner_patch()]);
                    }
                }
            } else {
#ifdef DEBUG
                bool owner_locked = false;
                if (m_context.m_patches_info[owner].is_deleted(
                        LocalT(lp.local_id_in_owner_patch()))) {
                    // if the element is deleted, then we should ensure
                    // that this patch is locked since then this would
                    // mean that this block has deleted it earlier which
                    // is okay

                    for (uint8_t i = 0; i < m_s_patches_to_lock_mask.size();
                         ++i) {
                        if (m_s_patches_to_lock_mask(i)) {
                            uint32_t pp = m_patch_info.patch_stash(i);
                            if (pp == owner) {
                                owner_locked = true;
                                break;
                            }
                        }
                    }
                    assert(owner_locked);
                }

                assert(owner_locked ||
                       !m_context.m_patches_info[owner].is_deleted(
                           LocalT(lp.local_id_in_owner_patch())));

#endif

                if (m_context.m_patches_info[owner].is_owned(
                        LocalT(lp.local_id_in_owner_patch()))) {
                    break;
                }
            }

            lp = m_context.m_patches_info[owner].get_lp<HandleT>().find(
                lp.local_id_in_owner_patch());

            assert(!lp.is_sentinel());

            owner = m_context.m_patches_info[owner].patch_stash.get_patch(lp);
        }

        return HandleT(owner, lp.local_id_in_owner_patch());
    }

    /**
     * @brief classify mesh elements in P such that if they are seen in a
     * different patch Q, then they could be deleted from Q
     * @return
     */
    __device__ __inline__ void classify_elements_for_post_migtation_cleanup(
        cooperative_groups::thread_block& block,
        Bitmask&                          vertex_incident_to_not_owned_face,
        Bitmask&                          vertex_incident_to_boundary_vertex,
        Bitmask&                          edge_incident_to_boundary_vertex,
        Bitmask&                          face_incident_to_boundary_vertex)
    {
        vertex_incident_to_not_owned_face.reset(block);
        block.sync();
        vertex_incident_to_boundary_vertex.reset(block);
        edge_incident_to_boundary_vertex.reset(block);
        face_incident_to_boundary_vertex.reset(block);

        // loop over not-owned faces in P and set a bit for the face's three
        // vertices in vertex_incident_to_not_owned_face
        const uint16_t num_faces = m_s_num_faces[0];
        for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
            if (!m_s_owned_mask_f(f) && m_s_active_mask_f(f)) {
                for (int i = 0; i < 3; i++) {
                    uint16_t edge = m_s_fe[3 * f + i];
                    assert((edge >> 1) < m_s_num_edges[0]);
                    flag_t e_dir(0);
                    Context::unpack_edge_dir(edge, edge, e_dir);
                    uint16_t e_id = (2 * edge) + (1 * e_dir);
                    assert(e_id < m_s_num_edges[0] * 2);
                    uint16_t vertex = m_s_ev[e_id];
                    assert(vertex < m_s_num_vertices[0]);
                    vertex_incident_to_not_owned_face.set(vertex, true);
                }
            }
        }
        block.sync();

        // loop over edges and set the vertex bit if it's connected to a vertex
        // that is incident to a not-owned face in
        // vertex_incident_to_boundary_vertex
        const uint16_t num_edges = m_s_num_edges[0];
        for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
            if (m_s_active_mask_e(e)) {
                const uint16_t v0 = m_s_ev[2 * e + 0];
                const uint16_t v1 = m_s_ev[2 * e + 1];
                if (vertex_incident_to_not_owned_face(v0) ||
                    vertex_incident_to_not_owned_face(v1)) {
                    vertex_incident_to_boundary_vertex.set(v0, true);
                    vertex_incident_to_boundary_vertex.set(v1, true);
                }
            }
        }
        block.sync();

        // loop over edges and set the edge bit if it is connected to a vertex
        // that is incident to boundary vertex in
        // edge_incident_to_boundary_vertex
        for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
            if (m_s_active_mask_e(e)) {
                uint16_t v0 = m_s_ev[2 * e + 0];
                uint16_t v1 = m_s_ev[2 * e + 1];
                if (vertex_incident_to_boundary_vertex(v0) &&
                    vertex_incident_to_boundary_vertex(v1)) {
                    edge_incident_to_boundary_vertex.set(e, true);
                }
            }
        }
        block.sync();

        // loop over edges and set the edge bit if it is connected to a vertex
        // that is incident to boundary vertex in
        // edge_incident_to_boundary_vertex
        for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
            if (m_s_active_mask_f(f)) {
                bool flag = true;
                for (int i = 0; i < 3; i++) {
                    const uint16_t edge = m_s_fe[3 * f + i] >> 1;
                    flag &= edge_incident_to_boundary_vertex(edge);
                }
                if (flag) {
                    face_incident_to_boundary_vertex.set(f, true);
                }
            }
        }
        block.sync();
    }


    /**
     * @brief given a local face in a patch, find its corresponding local
     * index in the patch associated with this cavity i.e., m_patch_info.
     * If the given face (local_id) is not owned by the given patch, they will
     * be mapped to their owner patch and local index in the owner patch
     * ##
     */
    __device__ __inline__ uint16_t find_copy_face(uint16_t& local_id,
                                                  uint32_t& patch)
    {
        return find_copy<FaceHandle>(local_id,
                                     patch,
                                     m_s_num_faces[0],
                                     m_s_owned_mask_f,
                                     m_s_active_mask_f,
                                     m_s_in_cavity_f,
                                     m_s_owner_f);
    }

    /**
     * @brief given a local edge in a patch, find its corresponding local
     * index in the patch associated with this cavity i.e., m_patch_info.
     * If the given edge (local_id) is not owned by the given patch, they will
     * be mapped to their owner patch and local index in the owner patch
     * ##
     */
    __device__ __inline__ uint16_t find_copy_edge(uint16_t& local_id,
                                                  uint32_t& patch)
    {
        return find_copy<EdgeHandle>(local_id,
                                     patch,
                                     m_s_num_edges[0],
                                     m_s_owned_mask_e,
                                     m_s_active_mask_e,
                                     m_s_in_cavity_e,
                                     m_s_owner_e);
    }

    /**
     * @brief given a local vertex in a patch, find its corresponding local
     * index in the patch associated with this cavity i.e., m_patch_info.
     * If the given vertex (local_id) is not owned by the given patch, they will
     * be mapped to their owner patch and local index in the owner patch.
     * ##
     */
    __device__ __inline__ uint16_t find_copy_vertex(uint16_t& local_id,
                                                    uint32_t& patch)
    {
        return find_copy<VertexHandle>(local_id,
                                       patch,
                                       m_s_num_vertices[0],
                                       m_s_owned_mask_v,
                                       m_s_active_mask_v,
                                       m_s_in_cavity_v,
                                       m_s_owner_v);
    }


    /**
     * @brief find a copy of mesh element from a src_patch in a dest_patch i.e.,
     * the lid lives in src_patch and we want to find the corresponding local
     * index in dest_patch
     * ##
     */
    template <typename HandleT>
    __device__ __inline__ uint16_t find_copy(
        uint16_t&       lid,
        uint32_t&       src_patch,
        const uint16_t  dest_patch_num_elements,
        const Bitmask&  dest_patch_owned_mask,
        const Bitmask&  dest_patch_active_mask,
        const Bitmask&  dest_in_cavity,
        const uint16_t* dest_owner)
    {
        // first check if lid is owned by src_patch. If not, then map it to its
        // owner patch and local index in it
        assert(!m_context.m_patches_info[src_patch].is_deleted(
            HandleT::LocalT(lid)));

        auto owner = m_context.get_owner_handle(HandleT(src_patch, {lid}));

        src_patch = owner.patch_id();
        lid       = owner.local_id();


        // if the owner src_patch is the same as the patch associated with this
        // cavity, the lid is the local index we are looking for
        if (src_patch == m_patch_info.patch_id) {
            return lid;
        }

        // otherwise, we do a search over the not-owned elements in the dest
        // patch. For every not-owned element, we map it to its owner patch and
        // check against lid-src_patch pair
        for (uint16_t i = 0; i < dest_patch_num_elements; ++i) {
            const uint16_t lp = dest_owner[i];

            if (lp != INVALID16) {
                const HandleT handle = convert_to_handle<HandleT>(lp);
                if (handle.patch_id() == src_patch &&
                    handle.local_id() == lid) {
                    return i;
                }
            }

            /*if (!dest_patch_owned_mask(i) &&
                (dest_patch_active_mask(i) || dest_in_cavity(i))) {
                // we disable check0 since the element might have been just
                // added in the patch in shared memory and not visible to global
                // memory yet

                auto handle = m_context.get_owner_handle<HandleT>(
                    {m_patch_info.patch_id, {i}},
                    nullptr,
                    nullptr,
                    false,
                    true);
                if (handle.patch_id() == src_patch &&
                    handle.local_id() == lid) {
                    return i;
                }
            }*/
        }
        return INVALID16;
    }

    /**
     * @brief check if the current timestamp is the same as the timestamp we
     * read during the constructor
     */
    __device__ __inline__ bool is_same_timestamp(
        cooperative_groups::thread_block& block) const
    {
        __shared__ uint32_t s_current_timestamp;
        if (threadIdx.x == 0) {
            s_current_timestamp = atomic_read(m_patch_info.timestamp);
        }
        block.sync();

        return s_current_timestamp == m_init_timestamp;
    }

    template <typename HandleT>
    __device__ __inline__ HandleT convert_to_handle(const uint16_t lp)
    {
        const uint8_t st =
            static_cast<uint8_t>((lp >> LPPair::LIDOwnerNumBits));
        const uint32_t owner = m_patch_info.patch_stash.m_stash[st];

        const uint16_t local_id_in_owner =
            detail::extract_low_bits<LPPair::LIDOwnerNumBits>(lp);

        return {owner, {local_id_in_owner}};
    }

    int *m_s_num_cavities, *m_s_cavity_size;

    Bitmask m_s_active_cavity_bitmask;
    Bitmask m_s_owned_mask_v, m_s_owned_mask_e, m_s_owned_mask_f;
    Bitmask m_s_active_mask_v, m_s_active_mask_e, m_s_active_mask_f;
    Bitmask m_s_migrate_mask_v;
    Bitmask m_s_src_mask_v, m_s_src_mask_e;
    Bitmask m_s_src_connect_mask_v, m_s_src_connect_mask_e;
    Bitmask m_s_ownership_change_mask_v, m_s_ownership_change_mask_e,
        m_s_ownership_change_mask_f;
    Bitmask m_s_owned_cavity_bdry_v;
    Bitmask m_s_ribbonize_v;
    Bitmask m_s_patches_to_lock_mask;
    Bitmask m_s_locked_patches_mask;
    Bitmask m_s_added_to_lp_v, m_s_added_to_lp_e, m_s_added_to_lp_f;
    Bitmask m_s_in_cavity_v, m_s_in_cavity_e, m_s_in_cavity_f;

    bool* m_s_readd_to_queue;

    uint16_t *m_s_ev, *m_s_fe;
    uint16_t *m_s_cavity_id_v, *m_s_cavity_id_e, *m_s_cavity_id_f;
    uint16_t *m_s_owner_v, *m_s_owner_e, *m_s_owner_f;
    uint16_t *m_s_num_vertices, *m_s_num_edges, *m_s_num_faces;
    uint16_t* m_s_cavity_edge_loop;
    PatchInfo m_patch_info;
    Context   m_context;
    uint32_t  m_init_timestamp;
};

}  // namespace rxmesh

#include "rxmesh/cavity_impl.cuh"