namespace rxmesh {

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ CavityManager<blockThreads, cop>::CavityManager(
    cooperative_groups::thread_block& block,
    Context&                          context,
    ShmemAllocator&                   shrd_alloc)
    : m_context(context)
{
    __shared__ uint32_t patch_id;
    __shared__ uint32_t smem[DIVIDE_UP(blockThreads, 32)];

    // assuming the max number of cavities created here is equal to the
    // number of threads in the block
    m_s_active_cavity_bitmask = Bitmask(blockThreads, smem);

    __shared__ uint16_t counts[3];
    m_s_num_vertices = counts + 0;
    m_s_num_edges    = counts + 1;
    m_s_num_faces    = counts + 2;

    __shared__ bool readd[1];
    m_s_readd_to_queue = readd;

    __shared__ int num_cavities[1];
    m_s_num_cavities = num_cavities;

    if (threadIdx.x == 0) {
        m_s_readd_to_queue[0] = false;
        m_s_num_cavities[0]   = 0;

        // get a patch
        patch_id = m_context.m_patch_scheduler.pop();

        // try to lock the patch
        if (patch_id != INVALID32) {
            bool locked = m_context.m_patches_info[patch_id].lock.acquire_lock(
                blockIdx.x);
            if (!locked) {
                // if we can not, we add it again to the queue
                m_context.m_patch_scheduler.push(patch_id);

                // and signal other threads to also exit
                patch_id = INVALID32;
            }
        }

        if (patch_id != INVALID32) {
            m_s_num_vertices[0] =
                m_context.m_patches_info[patch_id].num_vertices[0];
            m_s_num_edges[0] = m_context.m_patches_info[patch_id].num_edges[0];
            m_s_num_faces[0] = m_context.m_patches_info[patch_id].num_faces[0];
        }
    }
    block.sync();

    if (patch_id == INVALID32) {
        return;
    }

    m_patch_info = m_context.m_patches_info[patch_id];

    m_vert_cap = m_patch_info.vertices_capacity[0];
    m_edge_cap = m_patch_info.edges_capacity[0];
    m_face_cap = m_patch_info.faces_capacity[0];

    m_s_cavity_id_v      = shrd_alloc.alloc<uint16_t>(m_vert_cap);
    m_s_owner_v          = m_s_cavity_id_v;
    m_s_cavity_id_e      = shrd_alloc.alloc<uint16_t>(m_edge_cap);
    m_s_owner_e          = m_s_cavity_id_e;
    m_s_cavity_id_f      = shrd_alloc.alloc<uint16_t>(m_face_cap);
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
    alloc_masks(m_vert_cap,
                m_s_owned_mask_v,
                m_s_active_mask_v,
                m_s_ownership_change_mask_v,
                m_s_added_to_lp_v,
                m_s_in_cavity_v,
                m_patch_info.owned_mask_v,
                m_patch_info.active_mask_v);
    m_s_migrate_mask_v      = Bitmask(m_vert_cap, shrd_alloc);
    m_s_owned_cavity_bdry_v = Bitmask(m_vert_cap, shrd_alloc);
    m_s_ribbonize_v         = Bitmask(m_vert_cap, shrd_alloc);
    m_s_src_mask_v         = Bitmask(context.m_max_num_vertices[0], shrd_alloc);
    m_s_src_connect_mask_v = Bitmask(context.m_max_num_vertices[0], shrd_alloc);


    // edges masks
    alloc_masks(m_edge_cap,
                m_s_owned_mask_e,
                m_s_active_mask_e,
                m_s_ownership_change_mask_e,
                m_s_added_to_lp_e,
                m_s_in_cavity_e,
                m_patch_info.owned_mask_e,
                m_patch_info.active_mask_e);
    const uint16_t max_edge_cap =
        static_cast<uint16_t>(context.m_capacity_factor *
                              static_cast<float>(context.m_max_num_edges[0]));
    m_s_src_mask_e = Bitmask(std::max(max_edge_cap, m_edge_cap), shrd_alloc);
    m_s_src_connect_mask_e = Bitmask(context.m_max_num_edges[0], shrd_alloc);

    // faces masks
    alloc_masks(m_face_cap,
                m_s_owned_mask_f,
                m_s_active_mask_f,
                m_s_ownership_change_mask_f,
                m_s_added_to_lp_f,
                m_s_in_cavity_f,
                m_patch_info.owned_mask_f,
                m_patch_info.active_mask_f);

    m_s_patches_to_lock_mask = Bitmask(PatchStash::stash_size, shrd_alloc);
    m_s_locked_patches_mask  = Bitmask(PatchStash::stash_size, shrd_alloc);

    fill_n<blockThreads>(m_s_cavity_id_v, m_vert_cap, uint16_t(INVALID16));
    fill_n<blockThreads>(m_s_cavity_id_e, m_edge_cap, uint16_t(INVALID16));
    fill_n<blockThreads>(m_s_cavity_id_f, m_face_cap, uint16_t(INVALID16));


    m_s_patches_to_lock_mask.reset(block);
    m_s_active_cavity_bitmask.set(block);
    cooperative_groups::wait(block);
    block.sync();
}


template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ void CavityManager<blockThreads, cop>::create(
    HandleT seed)
{
    if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                  cop == CavityOp::VE || cop == CavityOp::VF) {
        static_assert(std::is_same_v<HandleT, VertexHandle>,
                      "CavityManager::create() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::V/CavityOp::VV/CavityOp::VE/CavityOp::VF, "
                      "create() should take VertexHandle as an input");
    }

    if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                  cop == CavityOp::EE || cop == CavityOp::EF) {
        static_assert(std::is_same_v<HandleT, EdgeHandle>,
                      "CavityManager::create() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::E/CavityOp::EV/CavityOp::EE/CavityOp::EF, "
                      "create() should take EdgeHandle as an input");
    }

    if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                  cop == CavityOp::FE || cop == CavityOp::FF) {
        static_assert(std::is_same_v<HandleT, FaceHandle>,
                      "CavityManager::create() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::F/CavityOp::FV/CavityOp::FE/CavityOp::FF, "
                      "create() should take FaceHandle as an input");
    }


    int id = ::atomicAdd(m_s_num_cavities, 1);

    // there is no race condition in here since each thread is assigned to
    // one element
    if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                  cop == CavityOp::VE || cop == CavityOp::VF) {
        assert(m_s_active_mask_v(seed.local_id()));
        assert(m_s_owned_mask_v(seed.local_id()));
        m_s_cavity_id_v[seed.local_id()] = id;
    }

    if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                  cop == CavityOp::EE || cop == CavityOp::EF) {
        assert(m_s_active_mask_e(seed.local_id()));
        assert(m_s_owned_mask_e(seed.local_id()));
        m_s_cavity_id_e[seed.local_id()] = id;
    }


    if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                  cop == CavityOp::FE || cop == CavityOp::FF) {
        assert(m_s_active_mask_f(seed.local_id()));
        assert(m_s_owned_mask_f(seed.local_id()));
        m_s_cavity_id_f[seed.local_id()] = id;
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool CavityManager<blockThreads, cop>::prologue(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc)
{
    // allocate memory for the cavity prefix sum
    m_s_cavity_size_prefix = shrd_alloc.alloc<int>(m_s_num_cavities[0] + 1);
    for (uint16_t i = threadIdx.x; i < m_s_num_cavities[0] + 1;
         i += blockThreads) {
        m_s_cavity_size_prefix[i] = 0;
    }


    // load mesh FE and EV
    load_mesh_async(block, shrd_alloc);
    block.sync();

    // propagate the cavity ID
    propagate(block);
    block.sync();

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
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::epilogue(
    cooperative_groups::thread_block& block)
{
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::load_mesh_async(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc)
{
    m_s_ev = shrd_alloc.alloc<uint16_t>(2 * m_edge_cap);
    detail::load_async(block,
                       reinterpret_cast<uint16_t*>(m_patch_info.ev),
                       2 * m_s_num_edges[0],
                       m_s_ev,
                       false);
    m_s_fe = shrd_alloc.alloc<uint16_t>(3 * m_face_cap);
    detail::load_async(block,
                       reinterpret_cast<uint16_t*>(m_patch_info.fe),
                       3 * m_s_num_faces[0],
                       m_s_fe,
                       true);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::propagate(
    cooperative_groups::thread_block& block)
{
    // Expand cavities by marking incident elements
    if constexpr (cop == CavityOp::V) {
        mark_edges_through_vertices();
        block.sync();
        mark_faces_through_edges();
    }

    if constexpr (cop == CavityOp::E) {
        mark_faces_through_edges();
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::mark_edges_through_vertices()
{
    for (uint16_t e = threadIdx.x; e < m_s_num_edges[0]; e += blockThreads) {
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


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::mark_faces_through_edges()
{
    for (uint16_t f = threadIdx.x; f < m_s_num_faces[0]; f += blockThreads) {
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


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::mark_element(
    uint16_t*      element_cavity_id,
    const uint16_t element_id,
    const uint16_t cavity_id)
{
    if (cavity_id != INVALID16) {
        const uint16_t prv_element_cavity_id = element_cavity_id[element_id];


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


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::deactivate_conflicting_cavities()
{
    deactivate_conflicting_cavities(m_s_num_vertices[0], m_s_cavity_id_v);

    deactivate_conflicting_cavities(m_s_num_edges[0], m_s_cavity_id_e);

    deactivate_conflicting_cavities(m_s_num_faces[0], m_s_cavity_id_f);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::deactivate_conflicting_cavities(
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


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::clear_bitmask_if_in_cavity()
{
    clear_bitmask_if_in_cavity(m_s_active_mask_v,
                               m_s_in_cavity_v,
                               m_s_cavity_id_v,
                               m_s_num_vertices[0]);
    clear_bitmask_if_in_cavity(
        m_s_active_mask_e, m_s_in_cavity_e, m_s_cavity_id_e, m_s_num_edges[0]);
    clear_bitmask_if_in_cavity(
        m_s_active_mask_f, m_s_in_cavity_f, m_s_cavity_id_f, m_s_num_faces[0]);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::clear_bitmask_if_in_cavity(
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


template <uint32_t blockThreads, CavityOp cop>
template <uint32_t itemPerThread>
__device__ __inline__ void
CavityManager<blockThreads, cop>::construct_cavities_edge_loop(
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
                local_offset[i] =
                    ::atomicAdd(m_s_cavity_size_prefix + face_cavity,
                                num_edges_on_boundary);
            }
        }
    }
    block.sync();

    // scan
    detail::cub_block_exclusive_sum<int, blockThreads>(m_s_cavity_size_prefix,
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
                    uint16_t offset = m_s_cavity_size_prefix[face_cavity] +
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


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::sort_cavities_edge_loop()
{

    // TODO need to increase the parallelism in this part. It should be at
    // least one warp processing one cavity
    for (uint16_t c = threadIdx.x; c < m_s_num_cavities[0]; c += blockThreads) {

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

        const uint16_t start = m_s_cavity_size_prefix[c];
        const uint16_t end   = m_s_cavity_size_prefix[c + 1];

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
                    uint16_t temp               = m_s_cavity_edge_loop[e + 1];
                    m_s_cavity_edge_loop[e + 1] = m_s_cavity_edge_loop[i];
                    m_s_cavity_edge_loop[i]     = temp;
                    break;
                }
            }
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
template <typename FillInT>
__device__ __inline__ void CavityManager<blockThreads, cop>::for_each_cavity(
    cooperative_groups::thread_block& block,
    FillInT                           FillInFunc)
{
    // TODO need to increase the parallelism in this part. It should be at
    // least one warp processing one cavity
    for (uint16_t c = threadIdx.x; c < m_s_num_cavities[0]; c += blockThreads) {
        const uint16_t size = get_cavity_size(c);
        if (size > 0) {
            FillInFunc(c, size);
        }
    }

    block.sync();
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ DEdgeHandle
CavityManager<blockThreads, cop>::get_cavity_edge(uint16_t c, uint16_t i) const
{
    assert(c < m_s_num_cavities[0]);
    assert(i < get_cavity_size(c));
    return DEdgeHandle(m_patch_info.patch_id,
                       m_s_cavity_edge_loop[m_s_cavity_size_prefix[c] + i]);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ VertexHandle
CavityManager<blockThreads, cop>::get_cavity_vertex(uint16_t c,
                                                    uint16_t i) const
{
    assert(c < m_s_num_cavities[0]);
    assert(i < get_cavity_size(c));

    uint16_t edge;
    flag_t   dir;
    Context::unpack_edge_dir(
        m_s_cavity_edge_loop[m_s_cavity_size_prefix[c] + i], edge, dir);

    const uint16_t v0 = m_s_ev[2 * edge];
    const uint16_t v1 = m_s_ev[2 * edge + 1];

    assert(m_s_active_mask_v(v0));
    assert(m_s_active_mask_v(v1));
    return VertexHandle(m_patch_info.patch_id, ((dir == 0) ? v0 : v1));
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ VertexHandle
CavityManager<blockThreads, cop>::add_vertex()
{
    // First try to reuse a vertex in the cavity or a deleted vertex
    uint16_t v_id = add_element(m_s_active_mask_v, m_s_num_vertices[0]);

    if (v_id == INVALID16) {
        // if this fails, then add a new vertex to the mesh
        v_id = atomicAdd(m_s_num_vertices, 1);
        assert(v_id < m_patch_info.vertices_capacity[0]);
    }

    assert(!m_s_active_mask_v(v_id));
    m_s_owned_mask_v.set(v_id, true);
    return {m_patch_info.patch_id, v_id};
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ DEdgeHandle CavityManager<blockThreads, cop>::add_edge(
    const VertexHandle src,
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
    m_s_owned_mask_e.set(e_id, true);
    return {m_patch_info.patch_id, e_id, 0};
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ FaceHandle CavityManager<blockThreads, cop>::add_face(
    const DEdgeHandle e0,
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

    m_s_owned_mask_f.set(f_id, true);

    return {m_patch_info.patch_id, f_id};
}


__device__ __inline__ uint16_t add_element(Bitmask        active_bitmask,
                                           const uint16_t num_elements)
{

    for (uint16_t i = 0; i < num_elements; ++i) {
        if (active_bitmask.try_set(i)) {
            return i;
        }
    }

    return INVALID16;
}
}  // namespace rxmesh