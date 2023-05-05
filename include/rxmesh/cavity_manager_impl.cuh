namespace rxmesh {

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ CavityManager<blockThreads, cop>::CavityManager(
    cooperative_groups::thread_block& block,
    Context&                          context,
    ShmemAllocator&                   shrd_alloc)
    : m_write_to_gmem(true), m_context(context)
{
    __shared__ uint32_t s_patch_id;
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
        s_patch_id = m_context.m_patch_scheduler.pop();

        // try to lock the patch
        if (s_patch_id != INVALID32) {
            bool locked =
                m_context.m_patches_info[s_patch_id].lock.acquire_lock(
                    blockIdx.x);

            if (!locked) {
                // if we can not, we add it again to the queue
                push();

                // and signal other threads to also exit
                s_patch_id = INVALID32;
            }
        }

        if (s_patch_id != INVALID32) {
            m_s_num_vertices[0] =
                m_context.m_patches_info[s_patch_id].num_vertices[0];
            m_s_num_edges[0] =
                m_context.m_patches_info[s_patch_id].num_edges[0];
            m_s_num_faces[0] =
                m_context.m_patches_info[s_patch_id].num_faces[0];
        }
    }
    block.sync();

    if (s_patch_id == INVALID32) {
        return;
    }

    m_patch_info = m_context.m_patches_info[s_patch_id];

    m_vert_cap = m_patch_info.vertices_capacity[0];
    m_edge_cap = m_patch_info.edges_capacity[0];
    m_face_cap = m_patch_info.faces_capacity[0];

    const uint32_t vert_cap_bytes = sizeof(uint16_t) * m_vert_cap;
    const uint32_t edge_cap_bytes = sizeof(uint16_t) * m_edge_cap;
    const uint32_t face_cap_bytes = sizeof(uint16_t) * m_face_cap;

    m_s_cavity_id_v = reinterpret_cast<uint16_t*>(shrd_alloc.alloc(
        std::max(vert_cap_bytes, m_patch_info.lp_v.num_bytes())));
    m_s_cavity_id_e = reinterpret_cast<uint16_t*>(shrd_alloc.alloc(
        std::max(edge_cap_bytes, m_patch_info.lp_e.num_bytes())));
    m_s_cavity_id_f = reinterpret_cast<uint16_t*>(shrd_alloc.alloc(
        std::max(face_cap_bytes, m_patch_info.lp_f.num_bytes())));

    m_s_cavity_boundary_edges = shrd_alloc.alloc<uint16_t>(m_s_num_edges[0]);

    auto alloc_masks = [&](uint16_t        num_elements,
                           Bitmask&        owned,
                           Bitmask&        active,
                           Bitmask&        ownership,
                           Bitmask&        in_cavity,
                           const uint32_t* g_owned,
                           const uint32_t* g_active) {
        owned     = Bitmask(num_elements, shrd_alloc);
        active    = Bitmask(num_elements, shrd_alloc);
        ownership = Bitmask(num_elements, shrd_alloc);
        in_cavity = Bitmask(num_elements, shrd_alloc);

        owned.reset(block);
        active.reset(block);
        ownership.reset(block);
        in_cavity.reset(block);

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


    // vertices masks
    alloc_masks(m_vert_cap,
                m_s_owned_mask_v,
                m_s_active_mask_v,
                m_s_ownership_change_mask_v,
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
template <typename HandleT>
__device__ __inline__ bool CavityManager<blockThreads, cop>::is_successful(
    HandleT seed)
{
    if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                  cop == CavityOp::VE || cop == CavityOp::VF) {
        static_assert(std::is_same_v<HandleT, VertexHandle>,
                      "CavityManager::is_successful() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::V/CavityOp::VV/CavityOp::VE/CavityOp::VF, "
                      "create() should take VertexHandle as an input");
    }

    if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                  cop == CavityOp::EE || cop == CavityOp::EF) {
        static_assert(std::is_same_v<HandleT, EdgeHandle>,
                      "CavityManager::is_successful() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::E/CavityOp::EV/CavityOp::EE/CavityOp::EF, "
                      "create() should take EdgeHandle as an input");
    }

    if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                  cop == CavityOp::FE || cop == CavityOp::FF) {
        static_assert(std::is_same_v<HandleT, FaceHandle>,
                      "CavityManager::is_successful() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::F/CavityOp::FV/CavityOp::FE/CavityOp::FF, "
                      "create() should take FaceHandle as an input");
    }

    if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                  cop == CavityOp::VE || cop == CavityOp::VF) {
        return m_s_in_cavity_v(seed.local_id());
    }

    if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                  cop == CavityOp::EE || cop == CavityOp::EF) {
        return m_s_in_cavity_e(seed.local_id());
    }


    if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                  cop == CavityOp::FE || cop == CavityOp::FF) {
        return m_s_in_cavity_f(seed.local_id());
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


    // load hashtables
    load_hashtable(block);
    block.sync();

    // change patch layout to accommodate all cavities created in the patch
    if (!migrate(block)) {
        m_write_to_gmem = false;
        unlock_locked_patches();
        return false;
    }
    block.sync();

    change_ownership(block);

    if (threadIdx.x == 0) {
        m_patch_info.num_vertices[0] = m_s_num_vertices[0];
        m_patch_info.num_edges[0]    = m_s_num_edges[0];
        m_patch_info.num_faces[0]    = m_s_num_faces[0];
    }

    block.sync();

    return true;
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
                    m_s_cavity_boundary_edges[offset] = e;
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
            uint32_t edge = m_s_cavity_boundary_edges[e];

            assert(m_s_active_mask_e((edge >> 1)));
            if (get_cavity_vertex(c, e - start).local_id() ==
                cavity_edge_src_vertex) {
                uint16_t temp = m_s_cavity_boundary_edges[start];
                m_s_cavity_boundary_edges[start] = edge;
                m_s_cavity_boundary_edges[e]     = temp;
                break;
            }
        }


        for (uint16_t e = start; e < end; ++e) {
            uint16_t edge;
            uint8_t  dir;
            Context::unpack_edge_dir(m_s_cavity_boundary_edges[e], edge, dir);
            uint16_t end_vertex = m_s_ev[2 * edge + 1];
            if (dir) {
                end_vertex = m_s_ev[2 * edge];
            }

            for (uint16_t i = e + 1; i < end; ++i) {
                uint32_t ee = m_s_cavity_boundary_edges[i] >> 1;
                uint32_t v0 = m_s_ev[2 * ee + 0];
                uint32_t v1 = m_s_ev[2 * ee + 1];

                assert(m_s_active_mask_v(v0));
                assert(m_s_active_mask_v(v1));
                if (v0 == end_vertex || v1 == end_vertex) {
                    uint16_t temp = m_s_cavity_boundary_edges[e + 1];
                    m_s_cavity_boundary_edges[e + 1] =
                        m_s_cavity_boundary_edges[i];
                    m_s_cavity_boundary_edges[i] = temp;
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
    return DEdgeHandle(
        m_patch_info.patch_id,
        m_s_cavity_boundary_edges[m_s_cavity_size_prefix[c] + i]);
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
        m_s_cavity_boundary_edges[m_s_cavity_size_prefix[c] + i], edge, dir);

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


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ uint16_t CavityManager<blockThreads, cop>::add_element(
    Bitmask        active_bitmask,
    const uint16_t num_elements)
{

    for (uint16_t i = 0; i < num_elements; ++i) {
        if (active_bitmask.try_set(i)) {
            return i;
        }
    }

    return INVALID16;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::load_hashtable(
    cooperative_groups::thread_block& block)
{

    m_s_table_v = reinterpret_cast<LPPair*>(m_s_cavity_id_v);
    m_s_table_e = reinterpret_cast<LPPair*>(m_s_cavity_id_e);
    m_s_table_f = reinterpret_cast<LPPair*>(m_s_cavity_id_f);

    m_patch_info.lp_v.load_in_shared_memory(m_s_table_v, false);
    m_patch_info.lp_e.load_in_shared_memory(m_s_table_e, false);
    m_patch_info.lp_f.load_in_shared_memory(m_s_table_f, true);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::push()
{
    if (threadIdx.x == 0) {
        m_context.m_patch_scheduler.push(m_patch_info.patch_id);
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __forceinline__ bool CavityManager<blockThreads, cop>::lock(
    cooperative_groups::thread_block& block,
    const uint8_t                     stash_id,
    const uint32_t                    q)
{
    __shared__ bool s_success;
    if (threadIdx.x == 0) {
        bool okay = m_s_locked_patches_mask(stash_id);
        if (!okay) {
            okay = m_context.m_patches_info[q].lock.acquire_lock(blockIdx.x);
            if (okay) {
                m_s_locked_patches_mask.set(stash_id);
            }
        }
        s_success = okay;
    }
    block.sync();
    return s_success;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __forceinline__ void CavityManager<blockThreads, cop>::unlock()
{
    if (threadIdx.x == 0) {
        m_patch_info.lock.release_lock();
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::unlock_locked_patches()
{
    if (threadIdx.x == 0) {
        for (uint8_t st = 0; st < PatchStash::stash_size; ++st) {
            if (m_s_locked_patches_mask(st)) {
                uint32_t q = m_patch_info.patch_stash.get_patch(st);
                assert(q != INVALID32);
                m_context.m_patches_info[q].lock.release_lock();
                m_s_locked_patches_mask.reset(st);
            }
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __forceinline__ void CavityManager<blockThreads, cop>::unlock(
    const uint8_t  stash_id,
    const uint32_t q)
{
    if (threadIdx.x == 0) {
        assert(m_s_locked_patches_mask(stash_id));
        m_context.m_patches_info[q].lock.release_lock();
        m_s_locked_patches_mask.reset(stash_id);
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool CavityManager<blockThreads, cop>::migrate(
    cooperative_groups::thread_block& block)
{
    // Some vertices on the boundary of the cavity are owned and other are
    // not. For owned vertices, edges and faces connected to them exists in
    // the patch (by definition) and they could be owned or not. For that,
    // we need to first make sure that these edges and faces are marked in
    // m_s_ownership_change_mask_e/f.
    // For not-owned vertices on the cavity boundary, we process them by
    // first marking them in m_s_migrate_mask_v and then look for their
    // owned version in the neighbor patches in migrate_from_patch


    m_s_ribbonize_v.reset(block);
    m_s_owned_cavity_bdry_v.reset(block);
    m_s_migrate_mask_v.reset(block);
    m_s_patches_to_lock_mask.reset(block);
    m_s_locked_patches_mask.reset(block);
    block.sync();


    // Mark vertices on the boundary of all active cavities in this patch
    // Owned vertices are marked in m_s_owned_cavity_bdry_v and not-owned
    // vertices are marked in m_s_migrate_mask_v (since we need to migrate
    // them) as well as in m_s_ownership_change_mask_v (since a vertex on the
    // boundary of the cavity has to be owned by the patch)
    // TODO this could be fused in construct_cavities_edge_loop()
    for_each_cavity(block, [&](uint16_t c, uint16_t size) {
        for (uint16_t i = 0; i < size; ++i) {
            uint16_t vertex = get_cavity_vertex(c, i).local_id();
            assert(m_s_active_mask_v(vertex));
            if (m_s_owned_mask_v(vertex)) {
                m_s_owned_cavity_bdry_v.set(vertex, true);
            } else {
                m_s_migrate_mask_v.set(vertex, true);
                m_s_ownership_change_mask_v.set(vertex, true);
            }
        }
    });
    block.sync();


    // Mark a face in the ownership change (m_s_ownership_change_mask_f) if
    // one of its edges is connected to a vertex that is marked in
    // m_s_owned_cavity_bdry_v. Then mark that face's three edges in the
    // ownership change (m_s_ownership_change_mask_e)
    for (uint16_t f = threadIdx.x; f < m_s_num_faces[0]; f += blockThreads) {
        if (!m_s_owned_mask_f(f) &&
            (m_s_active_mask_f(f) || m_s_in_cavity_f(f))) {
            bool change = false;

            const uint16_t edges[3] = {m_s_fe[3 * f + 0] >> 1,
                                       m_s_fe[3 * f + 1] >> 1,
                                       m_s_fe[3 * f + 2] >> 1};
            for (int i = 0; i < 3; ++i) {
                const uint16_t e = edges[i];

                assert(m_s_active_mask_e(e) || m_s_in_cavity_e(e));

                const uint16_t v0 = m_s_ev[2 * e + 0];
                const uint16_t v1 = m_s_ev[2 * e + 1];

                assert(m_s_active_mask_v(v0) || m_s_in_cavity_v(v0));
                assert(m_s_active_mask_v(v1) || m_s_in_cavity_v(v1));

                if (m_s_owned_cavity_bdry_v(v0) ||
                    m_s_owned_cavity_bdry_v(v1) || m_s_migrate_mask_v(v0) ||
                    m_s_migrate_mask_v(v1)) {
                    change = true;
                    m_s_ownership_change_mask_f.set(f, true);
                    break;
                }
            }

            if (change) {
                for (int i = 0; i < 3; ++i) {
                    const uint16_t e = edges[i];
                    if (!m_s_owned_mask_e(e)) {
                        assert(m_s_active_mask_e(e) || m_s_in_cavity_e(e));
                        m_s_ownership_change_mask_e.set(e, true);
                    }
                }
            }
        }
    }
    block.sync();


    // construct protection zone
    for (uint32_t st = 0; st < PatchStash::stash_size; ++st) {
        const uint32_t q = m_patch_info.patch_stash.get_patch(st);
        if (q != INVALID32) {
            if (!migrate_from_patch(block, st, q, m_s_migrate_mask_v, true)) {
                return false;
            }
        }
    }
    block.sync();


    // ribbonize protection zone
    for (uint16_t e = threadIdx.x; e < m_s_num_edges[0]; e += blockThreads) {
        if (m_s_active_mask_e(e) || m_s_in_cavity_e(e)) {

            // We want to ribbonize vertices connected to a vertex on
            // the boundary of a cavity boundaries. If the two vertices are
            // on the cavity boundaries (b0=true and b1=true), then this is
            // an edge on the cavity and we don't to ribbonize any of these
            // two vertices. Only when one of the vertices are on the cavity
            // boundaries and the other is not, we then want to ribbonize
            // the other one

            const uint16_t v0 = m_s_ev[2 * e + 0];
            const uint16_t v1 = m_s_ev[2 * e + 1];

            assert(m_s_active_mask_v(v0));
            assert(m_s_active_mask_v(v1));

            const bool b0 =
                m_s_migrate_mask_v(v0) || m_s_owned_cavity_bdry_v(v0);

            const bool b1 =
                m_s_migrate_mask_v(v1) || m_s_owned_cavity_bdry_v(v1);

            if (b0 && !b1 && !m_s_owned_mask_v(v1)) {
                // The vertex we want to ribbonize should not be inside the
                // cavity. we assert this here and if this fails then maybe we
                // should look into this case to undertsand if this makes sense
                assert(!m_s_in_cavity_v(v1));
                m_s_ribbonize_v.set(v1, true);
            }

            if (b1 && !b0 && !m_s_owned_mask_v(v0)) {
                assert(!m_s_in_cavity_v(v0));
                m_s_ribbonize_v.set(v0, true);
            }
        }
    }

    block.sync();

    for (uint32_t st = 0; st < PatchStash::stash_size; ++st) {
        const uint32_t q = m_patch_info.patch_stash.get_patch(st);
        if (q != INVALID32) {
            if (!migrate_from_patch(block, st, q, m_s_ribbonize_v, false)) {
                return false;
            }
        }
    }

    return true;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool CavityManager<blockThreads, cop>::migrate_from_patch(
    cooperative_groups::thread_block& block,
    const uint8_t                     q_stash_id,
    const uint32_t                    q,
    const Bitmask&                    migrate_mask_v,
    const bool                        change_ownership)
{
    // migrate_mask_v uses the index space of p
    // m_s_src_mask_v and m_s_src_connect_mask_v use the index space of q

    // 1. mark vertices in m_s_src_mask_v that corresponds to vertices
    // marked in migrate_mask_v
    // 2. mark vertices in m_s_src_connect_mask_v that are connected to
    // vertices in m_s_src_mask_v
    // 3. move vertices marked in m_s_src_connect_mask_v to p
    // 4. move any edges formed by a vertex in m_s_src_mask_v from q to p
    // and mark these edges in m_s_src_mask_e
    // 5. move edges needed to represent any face that has a vertex marked
    // in m_s_src_mask_v
    // 6. move the faces that touch at least one vertex marked in
    // m_s_src_mask_v


    __shared__ int s_ok_q;
    if (threadIdx.x == 0) {
        s_ok_q = 0;
    }


    // loop over m_s_patches_to_lock and make sure that any patch set to true is
    // locked (which is indicated in m_s_locked_patches)
    auto lock_patches_to_lock = [&]() {
        for (uint8_t st = 0; st < PatchStash::stash_size; ++st) {
            if (m_s_patches_to_lock_mask(st) && !m_s_locked_patches_mask(st)) {
                const uint32_t patch = m_patch_info.patch_stash.get_patch(st);
                if (!lock(block, st, patch)) {
                    return false;
                }
            }
        }
        return true;
    };

    // try to lock q before reading from it
    if (!lock(block, q_stash_id, q)) {
        return false;
    }

    // init src_v bitmask
    m_s_src_mask_v.reset(block);
    block.sync();


    // 1. mark vertices in q that will be migrated into p
    // this requires query p's hashtable, so we could not insert in
    // it now. If no vertices found, then we skip this patch
    for (uint32_t v = threadIdx.x; v < m_s_num_vertices[0]; v += blockThreads) {
        if (migrate_mask_v(v)) {
            // get the owner patch of v

            // we don't check if this vertex is active in global memory
            // since, it could have been activated/added only in shared
            // memory (through a previous call to mirgate_from_patch)
            assert(m_s_active_mask_v(v));
            assert(!m_s_owned_mask_v(v));

            // const VertexHandle v_owner =
            //    m_context.get_owner_handle<VertexHandle>(
            //        {m_patch_info.patch_id, {v}},
            //        nullptr,
            //        m_s_table_v,
            //        false,
            //        false);

            const VertexHandle v_owner =
                m_patch_info.find<VertexHandle>(v, m_s_table_v);

            assert(v_owner.is_valid());
            assert(v_owner.patch_id() != INVALID32);
            assert(v_owner.local_id() != INVALID16);

            if (v_owner.patch_id() == q) {

                // make sure that q is the actual owner of of v
                assert(m_context.m_patches_info[q].is_owned(
                    LocalVertexT(v_owner.local_id())));

                ::atomicAdd(&s_ok_q, 1);
                m_s_src_mask_v.set(v_owner.local_id(), true);
            }
        }
    }
    block.sync();


    if (s_ok_q == 0) {
        // we should not keep q locked
        unlock(q_stash_id, q);
    } else {

        // In every call to migrate_vertex/edge/face, threads make sure that
        // they mark patches they read from in m_s_patches_to_lock_mask.
        // At the end of every round, one thread make sure make sure that all
        // patches marked in m_s_patches_to_lock_mask are actually locked.

        PatchInfo q_patch_info = m_context.m_patches_info[q];

        const uint16_t q_num_vertices = q_patch_info.num_vertices[0];
        const uint16_t q_num_edges    = q_patch_info.num_edges[0];
        const uint16_t q_num_faces    = q_patch_info.num_faces[0];

        // initialize connect_mask and src_e bitmask
        m_s_src_connect_mask_v.reset(block);
        m_s_src_connect_mask_e.reset(block);
        m_s_src_mask_e.reset(block);
        block.sync();

        // 2. in m_s_src_connect_mask_v, mark the vertices connected to
        // vertices in m_s_src_mask_v
        for (uint16_t e = threadIdx.x; e < q_num_edges; e += blockThreads) {
            if (!q_patch_info.is_deleted(LocalEdgeT(e))) {
                const uint16_t v0q = q_patch_info.ev[2 * e + 0].id;
                const uint16_t v1q = q_patch_info.ev[2 * e + 1].id;

                if (m_s_src_mask_v(v0q)) {
                    m_s_src_connect_mask_v.set(v1q, true);
                    assert(!q_patch_info.is_deleted(LocalVertexT(v1q)));
                }

                if (m_s_src_mask_v(v1q)) {
                    m_s_src_connect_mask_v.set(v0q, true);
                    assert(!q_patch_info.is_deleted(LocalVertexT(v0q)));
                }
            }
        }
        block.sync();

        // 3. make sure there is a copy in p for any vertex in
        // m_s_src_connect_mask_v
        const uint16_t q_num_vertices_up =
            ROUND_UP_TO_NEXT_MULTIPLE(q_num_vertices, blockThreads);

        // we need to make sure that no other thread is query the
        // vertex hashtable before adding items to it. So, we need
        // to sync the whole block before adding a new vertex but
        // some threads may not be participant in this for-loop.
        // So, we round up the end of the loop to be multiple of the
        // blockthreads and check inside the loop so we don't access
        // non-existing vertices
        for (uint16_t v = threadIdx.x; v < q_num_vertices_up;
             v += blockThreads) {

            LPPair lp =
                migrate_vertex(q,
                               q_num_vertices,
                               v,
                               change_ownership,
                               q_patch_info,
                               [&](const uint16_t vertex) {
                                   return m_s_src_connect_mask_v(vertex);
                               });
            // we need to make sure that no other
            // thread is querying the hashtable while we
            // insert in it
            block.sync();

            if (!lp.is_sentinel()) {
                bool inserted = m_patch_info.lp_v.insert(lp, m_s_table_v);
                assert(inserted);
            }
            block.sync();
        }


        if (!lock_patches_to_lock()) {
            return false;
        }


        // same story as with the loop that adds vertices
        const uint16_t q_num_edges_up =
            ROUND_UP_TO_NEXT_MULTIPLE(q_num_edges, blockThreads);

        // 4. move edges since we now have a copy of the vertices in p
        for (uint16_t e = threadIdx.x; e < q_num_edges_up; e += blockThreads) {
            LPPair lp = migrate_edge(
                q,
                q_num_edges,
                e,
                change_ownership,
                q_patch_info,
                [&](const uint16_t edge,
                    const uint16_t v0q,
                    const uint16_t v1q) {
                    // If any of these two vertices are participant in
                    // the src bitmask
                    if (m_s_src_mask_v(v0q) || m_s_src_mask_v(v1q)) {
                        assert(!q_patch_info.is_deleted(LocalEdgeT(edge)));
                        // set the bit for this edge in src_e mask so we
                        // can use it for migrating faces
                        m_s_src_mask_e.set(edge, true);
                        return true;
                    }
                    return false;
                });

            block.sync();
            if (!lp.is_sentinel()) {
                bool inserted = m_patch_info.lp_e.insert(lp, m_s_table_e);
                assert(inserted);
            }
            block.sync();
        }


        if (!lock_patches_to_lock()) {
            return false;
        }

        // 5. in m_s_src_connect_mask_e, mark the edges connected to
        // faces that has an edge that is marked in m_s_src_mask_e
        // Since edges in m_s_src_mask_e are marked because they
        // have one vertex in m_s_src_mask_v, then any face touches
        // these edges also touches a vertex in m_s_src_mask_v. Since
        // we migrate all faces touches a vertex in m_s_src_mask_v,
        // we need first to represent the edges that touch these
        // faces in q before migrating the faces
        for (uint16_t f = threadIdx.x; f < q_num_faces; f += blockThreads) {
            if (!q_patch_info.is_deleted(LocalFaceT(f))) {

                const uint16_t e0 = q_patch_info.fe[3 * f + 0].id >> 1;
                const uint16_t e1 = q_patch_info.fe[3 * f + 1].id >> 1;
                const uint16_t e2 = q_patch_info.fe[3 * f + 2].id >> 1;

                assert(!q_patch_info.is_deleted(LocalEdgeT(e0)));
                assert(!q_patch_info.is_deleted(LocalEdgeT(e1)));
                assert(!q_patch_info.is_deleted(LocalEdgeT(e2)));

                bool b0 = m_s_src_mask_e(e0);
                bool b1 = m_s_src_mask_e(e1);
                bool b2 = m_s_src_mask_e(e2);

                if (b0 || b1 || b2) {
                    if (!b0) {
                        m_s_src_connect_mask_e.set(e0, true);
                    }
                    if (!b1) {
                        m_s_src_connect_mask_e.set(e1, true);
                    }
                    if (!b2) {
                        m_s_src_connect_mask_e.set(e2, true);
                    }
                }
            }
        }
        block.sync();

        // make sure that there is a copy of edge in
        // m_s_src_connect_mask_e in q
        for (uint16_t e = threadIdx.x; e < q_num_edges_up; e += blockThreads) {

            LPPair lp = migrate_edge(q,
                                     q_num_edges,
                                     e,
                                     change_ownership,
                                     q_patch_info,
                                     [&](const uint16_t edge,
                                         const uint16_t v0q,
                                         const uint16_t v1q) {
                                         return m_s_src_connect_mask_e(edge);
                                     });
            block.sync();

            if (!lp.is_sentinel()) {
                bool inserted = m_patch_info.lp_e.insert(lp, m_s_table_e);
                assert(inserted);
            }
            block.sync();
        }

        if (!lock_patches_to_lock()) {
            return false;
        }


        // same story as with the loop that adds vertices
        const uint16_t q_num_faces_up =
            ROUND_UP_TO_NEXT_MULTIPLE(q_num_faces, blockThreads);

        // 6.  move face since we now have a copy of the edges in p
        for (uint16_t f = threadIdx.x; f < q_num_faces_up; f += blockThreads) {
            LPPair lp = migrate_face(q,
                                     q_num_faces,
                                     f,
                                     change_ownership,
                                     q_patch_info,
                                     [&](const uint16_t face,
                                         const uint16_t e0q,
                                         const uint16_t e1q,
                                         const uint16_t e2q) {
                                         return m_s_src_mask_e(e0q) ||
                                                m_s_src_mask_e(e1q) ||
                                                m_s_src_mask_e(e2q);
                                     });
            block.sync();

            if (!lp.is_sentinel()) {
                bool inserted = m_patch_info.lp_f.insert(lp, m_s_table_f);
                assert(inserted);
            }
            block.sync();
        }

        if (!lock_patches_to_lock()) {
            return false;
        }
    }

    return true;
}


template <uint32_t blockThreads, CavityOp cop>
template <typename FuncT>
__device__ __inline__ LPPair CavityManager<blockThreads, cop>::migrate_vertex(
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

            assert(!m_context.m_patches_info[o].is_deleted(LocalVertexT(vq)));
            assert(m_context.m_patches_info[o].is_owned(LocalVertexT(vq)));

            if (vp == INVALID16) {

                vp = atomicAdd(m_s_num_vertices, 1u);

                assert(vp < m_patch_info.vertices_capacity[0]);

                // activate the vertex in the bit mask
                m_s_active_mask_v.set(vp, true);

                // since it is owned by some other patch
                m_s_owned_mask_v.reset(vp, true);


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
            }
        }
    }
    return ret;
}


template <uint32_t blockThreads, CavityOp cop>
template <typename FuncT>
__device__ __inline__ LPPair CavityManager<blockThreads, cop>::migrate_edge(
    const uint32_t q,
    const uint16_t q_num_edges,
    const uint16_t q_edge,
    const bool     require_ownership_change,
    PatchInfo&     q_patch_info,
    FuncT          should_migrate)
{
    LPPair ret;

    if (q_edge < q_num_edges && !q_patch_info.is_deleted(LocalEdgeT(q_edge))) {

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
                assert(
                    m_context.m_patches_info[o0].is_owned(LocalVertexT(v0q)));

                assert(!m_context.m_patches_info[o1].is_deleted(
                    LocalVertexT(v1q)));
                assert(
                    m_context.m_patches_info[o1].is_owned(LocalVertexT(v1q)));


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
            }
        }
    }


    return ret;
}


template <uint32_t blockThreads, CavityOp cop>
template <typename FuncT>
__device__ __inline__ LPPair CavityManager<blockThreads, cop>::migrate_face(
    const uint32_t q,
    const uint16_t q_num_faces,
    const uint16_t q_face,
    const bool     require_ownership_change,
    PatchInfo&     q_patch_info,
    FuncT          should_migrate)
{
    LPPair ret;

    if (q_face < q_num_faces && !q_patch_info.is_deleted(LocalFaceT(q_face))) {

        uint16_t e0q, e1q, e2q;
        flag_t   d0, d1, d2;
        Context::unpack_edge_dir(q_patch_info.fe[3 * q_face + 0].id, e0q, d0);
        Context::unpack_edge_dir(q_patch_info.fe[3 * q_face + 1].id, e1q, d1);
        Context::unpack_edge_dir(q_patch_info.fe[3 * q_face + 2].id, e2q, d2);

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

                assert(
                    !m_context.m_patches_info[o0].is_deleted(LocalEdgeT(e0q)));
                assert(m_context.m_patches_info[o0].is_owned(LocalEdgeT(e0q)));

                assert(
                    !m_context.m_patches_info[o1].is_deleted(LocalEdgeT(e1q)));
                assert(m_context.m_patches_info[o1].is_owned(LocalEdgeT(e1q)));

                assert(
                    !m_context.m_patches_info[o2].is_deleted(LocalEdgeT(e2q)));
                assert(m_context.m_patches_info[o2].is_owned(LocalEdgeT(e2q)));

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
            }
        }
    }

    return ret;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ uint16_t
CavityManager<blockThreads, cop>::find_copy_vertex(uint16_t& local_id,
                                                   uint32_t& patch)
{
    return find_copy<VertexHandle>(local_id,
                                   patch,
                                   m_s_num_vertices[0],
                                   m_s_owned_mask_v,
                                   m_s_active_mask_v,
                                   m_s_in_cavity_v,
                                   m_s_table_v);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ uint16_t CavityManager<blockThreads, cop>::find_copy_edge(
    uint16_t& local_id,
    uint32_t& patch)
{
    return find_copy<EdgeHandle>(local_id,
                                 patch,
                                 m_s_num_edges[0],
                                 m_s_owned_mask_e,
                                 m_s_active_mask_e,
                                 m_s_in_cavity_e,
                                 m_s_table_e);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ uint16_t CavityManager<blockThreads, cop>::find_copy_face(
    uint16_t& local_id,
    uint32_t& patch)
{
    return find_copy<FaceHandle>(local_id,
                                 patch,
                                 m_s_num_faces[0],
                                 m_s_owned_mask_f,
                                 m_s_active_mask_f,
                                 m_s_in_cavity_f,
                                 m_s_table_f);
}


template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ uint16_t CavityManager<blockThreads, cop>::find_copy(
    uint16_t&      lid,
    uint32_t&      src_patch,
    const uint16_t dest_patch_num_elements,
    const Bitmask& dest_patch_owned_mask,
    const Bitmask& dest_patch_active_mask,
    const Bitmask& dest_in_cavity,
    const LPPair*  s_table)
{

    assert(
        !m_context.m_patches_info[src_patch].is_deleted(HandleT::LocalT(lid)));

    // First check if lid is owned by src_patch. If not, then map it to its
    // owner patch and local index in it

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
        if (!dest_patch_owned_mask(i) &&
            (dest_patch_active_mask(i) || dest_in_cavity(i))) {
            // we disable check0 since the element might have been just
            // added in the patch in shared memory and not visible to global
            // memory yet
            // auto handle = m_context.get_owner_handle<HandleT>(
            //    {m_patch_info.patch_id, {i}}, nullptr, s_table, false, true);

            const HandleT handle = m_patch_info.find<HandleT>(i, s_table);

            assert(handle.is_valid());
            assert(handle.patch_id() != INVALID32);
            assert(handle.local_id() != INVALID16);

            if (handle.patch_id() == src_patch && handle.local_id() == lid) {
                return i;
            }
        }
    }
    return INVALID16;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::change_ownership(
    cooperative_groups::thread_block& block)
{
    change_ownership<VertexHandle>(block,
                                   m_s_num_vertices[0],
                                   m_s_ownership_change_mask_v,
                                   m_s_table_v,
                                   m_s_owned_mask_v);

    change_ownership<EdgeHandle>(block,
                                 m_s_num_edges[0],
                                 m_s_ownership_change_mask_e,
                                 m_s_table_e,
                                 m_s_owned_mask_e);

    change_ownership<FaceHandle>(block,
                                 m_s_num_faces[0],
                                 m_s_ownership_change_mask_f,
                                 m_s_table_f,
                                 m_s_owned_mask_f);
}


template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ void CavityManager<blockThreads, cop>::change_ownership(
    cooperative_groups::thread_block& block,
    const uint16_t                    num_elements,
    const Bitmask&                    s_ownership_change,
    const LPPair*                     s_table,
    Bitmask&                          s_owned_bitmask)
{
    for (uint16_t vp = threadIdx.x; vp < num_elements; vp += blockThreads) {

        if (s_ownership_change(vp)) {

            assert(!m_patch_info.is_owned(HandleT::LocalT(vp)));

            const HandleT h = m_patch_info.find<HandleT>(vp, s_table);

            assert(h.patch_id() != INVALID32);
            assert(h.local_id() != INVALID16);

            const uint32_t q  = h.patch_id();
            const uint16_t vq = h.local_id();

            // set the bitmask of this element in shared memory
            s_owned_bitmask.set(vp, true);

            // m_patch_info.get_lp<HandleT>().remove(vp);

            // make sure that q is locked
            assert(m_s_locked_patches_mask(
                m_patch_info.patch_stash.find_patch_index(q)));


            assert(q != m_patch_info.patch_id);

            assert(
                !m_context.m_patches_info[q].is_deleted(HandleT::LocalT(vq)));

            // TODO if q is no longer the owner, that means some other patch has
            // changed the ownership of vq can be explained as cavities overlap
            assert(m_context.m_patches_info[q].is_owned(HandleT::LocalT(vq)));

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


template <uint32_t blockThreads, CavityOp cop>
template <typename AttributeT>
__device__ __inline__ void CavityManager<blockThreads, cop>::update_attributes(
    cooperative_groups::thread_block& block,
    AttributeT&                       attribute)
{
    using HandleT = typename AttributeT::HandleType;
    using Type    = typename AttributeT::Type;

    const uint32_t p = m_patch_info.patch_id;

    auto copy_from_owner = [&](const uint16_t vp, const LPPair* s_table) {
        const HandleT h = m_patch_info.find<HandleT>(vp, s_table);

        assert(h.patch_id() != p);
        assert(h.patch_id() != INVALID32);
        assert(h.local_id() != INVALID16);

        const uint32_t num_attr = attribute.get_num_attributes();
        for (uint32_t attr = 0; attr < num_attr; ++attr) {
            attribute(p, vp, attr) = attribute(h, attr);
        }
    };

    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        for (uint16_t vp = threadIdx.x; vp < m_s_num_vertices[0];
             vp += blockThreads) {
            if (m_s_ownership_change_mask_v(vp)) {

                assert(m_s_owned_mask_v(vp));
                assert(m_s_active_mask_v(vp) || m_s_in_cavity_v(vp));

                copy_from_owner(vp, m_s_table_v);
            }
        }
    }

    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        for (uint16_t ep = threadIdx.x; ep < m_s_num_edges[0];
             ep += blockThreads) {
            if (m_s_ownership_change_mask_e(ep)) {
                assert(m_s_owned_mask_e(ep));
                assert(m_s_active_mask_e(ep) || m_s_in_cavity_e(ep));

                copy_from_owner(ep, m_s_table_e);
            }
        }
    }

    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        for (uint16_t fp = threadIdx.x; fp < m_s_num_faces[0];
             fp += blockThreads) {
            if (m_s_ownership_change_mask_f(fp)) {
                assert(m_s_owned_mask_f(fp));
                assert(m_s_active_mask_f(fp) || m_s_in_cavity_f(fp));

                copy_from_owner(fp, m_s_table_f);
            }
        }
    }

    block.sync();
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::epilogue(
    cooperative_groups::thread_block& block)
{
    // make sure all writes are done
    block.sync();
    if (m_write_to_gmem) {
        // unlock any neighbor patch we have locked
        unlock_locked_patches();

        // update number of elements again since add_vertex/edge/face could have
        // changed it
        if (threadIdx.x == 0) {
            m_patch_info.num_vertices[0] = m_s_num_vertices[0];
            m_patch_info.num_edges[0]    = m_s_num_edges[0];
            m_patch_info.num_faces[0]    = m_s_num_faces[0];
        }

        // cleanup the hashtable by removing the vertices/edges/faces that has
        // changed their ownership to be in this patch (p) and thus should not
        // be in the hashtable
        for (uint32_t vp = threadIdx.x; vp < m_s_num_vertices[0];
             vp += blockThreads) {
            if (m_s_ownership_change_mask_v(vp)) {
                m_s_readd_to_queue[0] = true;
                m_patch_info.lp_v.remove(vp, m_s_table_v);
            }
        }

        for (uint32_t ep = threadIdx.x; ep < m_s_num_edges[0];
             ep += blockThreads) {
            if (m_s_ownership_change_mask_e(ep)) {
                m_s_readd_to_queue[0] = true;
                m_patch_info.lp_e.remove(ep, m_s_table_e);
            }
        }

        for (uint32_t fp = threadIdx.x; fp < m_s_num_faces[0];
             fp += blockThreads) {
            if (m_s_ownership_change_mask_f(fp)) {
                m_s_readd_to_queue[0] = true;
                m_patch_info.lp_f.remove(fp, m_s_table_f);
            }
        }

        ::atomicMax(m_context.m_max_num_vertices, m_s_num_vertices[0]);
        ::atomicMax(m_context.m_max_num_edges, m_s_num_edges[0]);
        ::atomicMax(m_context.m_max_num_faces, m_s_num_faces[0]);

        // store connectivity
        detail::store<blockThreads>(
            m_s_ev,
            2 * m_s_num_edges[0],
            reinterpret_cast<uint16_t*>(m_patch_info.ev));

        detail::store<blockThreads>(
            m_s_fe,
            3 * m_s_num_faces[0],
            reinterpret_cast<uint16_t*>(m_patch_info.fe));

        // store bitmask
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

        // store hashtable
        m_patch_info.lp_v.write_to_global_memory<blockThreads>(m_s_table_v);
        m_patch_info.lp_e.write_to_global_memory<blockThreads>(m_s_table_e);
        m_patch_info.lp_f.write_to_global_memory<blockThreads>(m_s_table_f);
    }

    // re-add the patch to the queue if there is ownership change
    // or we could not lock all neighbor patches (and thus could not write to
    // global memory)
    if (m_s_readd_to_queue[0] || !m_write_to_gmem) {
        push();
    }


    // unlock this patch
    unlock();
}
}  // namespace rxmesh