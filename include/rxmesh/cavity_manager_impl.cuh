namespace rxmesh {

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ CavityManager<blockThreads, cop>::CavityManager(
    cooperative_groups::thread_block& block,
    Context&                          context,
    ShmemAllocator&                   shrd_alloc,
    uint32_t                          current_p)
    : m_write_to_gmem(false), m_context(context)
{
    __shared__ uint32_t s_patch_id;


    __shared__ uint32_t counts[3];
    m_s_num_vertices = counts + 0;
    m_s_num_edges    = counts + 1;
    m_s_num_faces    = counts + 2;

    __shared__ bool readd[1];
    m_s_readd_to_queue = readd;

    __shared__ bool slice[1];
    m_s_should_slice = slice;

    __shared__ bool fill[1];
    m_s_remove_fill_in = fill;

    __shared__ int num_cavities[1];
    m_s_num_cavities = num_cavities;


    if (threadIdx.x == 0) {
        m_s_readd_to_queue[0] = false;
        m_s_should_slice[0]   = false;
        m_s_remove_fill_in[0] = false;
        m_s_num_cavities[0]   = 0;

        // get a patch
        s_patch_id = m_context.m_patch_scheduler.pop();
#ifdef PROCESS_SINGLE_PATCH
        if (s_patch_id != current_p) {
            s_patch_id = INVALID32;
        }
#endif


        if (s_patch_id != INVALID32) {
            if (m_context.m_patches_info[s_patch_id].patch_id == INVALID32) {
                s_patch_id = INVALID32;
            }
        }

        // try to lock the patch
        if (s_patch_id != INVALID32) {
            bool locked =
                m_context.m_patches_info[s_patch_id].lock.acquire_lock(
                    blockIdx.x);

            if (!locked) {
                // if we can not, we add it again to the queue
                push(s_patch_id);

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

    const uint32_t vert_cap = m_patch_info.vertices_capacity[0];
    const uint32_t edge_cap = m_patch_info.edges_capacity[0];
    const uint32_t face_cap = m_patch_info.faces_capacity[0];

    const uint32_t vert_cap_bytes = sizeof(uint16_t) * vert_cap;
    const uint32_t edge_cap_bytes = sizeof(uint16_t) * edge_cap;
    const uint32_t face_cap_bytes = sizeof(uint16_t) * face_cap;

    m_s_cavity_id_v = reinterpret_cast<uint16_t*>(shrd_alloc.alloc(
        std::max(vert_cap_bytes, m_patch_info.lp_v.num_bytes())));
    m_s_cavity_id_e = reinterpret_cast<uint16_t*>(shrd_alloc.alloc(
        std::max(edge_cap_bytes, m_patch_info.lp_e.num_bytes())));
    m_s_cavity_id_f = reinterpret_cast<uint16_t*>(shrd_alloc.alloc(
        std::max(face_cap_bytes, m_patch_info.lp_f.num_bytes())));

    const uint16_t assumed_num_cavities = m_context.m_max_num_faces[0] / 2;
    m_s_cavity_creator = shrd_alloc.alloc<uint16_t>(assumed_num_cavities);
    fill_n<blockThreads>(
        m_s_cavity_creator, assumed_num_cavities, uint16_t(INVALID16));

    fill_n<blockThreads>(m_s_cavity_id_v, vert_cap, uint16_t(INVALID16));
    fill_n<blockThreads>(m_s_cavity_id_e, edge_cap, uint16_t(INVALID16));
    fill_n<blockThreads>(m_s_cavity_id_f, face_cap, uint16_t(INVALID16));

    block.sync();
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::alloc_shared_memory(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc)
{
    m_s_patch_stash_mutex.alloc();

    const uint16_t vert_cap = m_patch_info.vertices_capacity[0];
    const uint16_t edge_cap = m_patch_info.edges_capacity[0];
    const uint16_t face_cap = m_patch_info.faces_capacity[0];

    const uint16_t max_vertex_cap = static_cast<uint16_t>(
        m_context.m_capacity_factor *
        static_cast<float>(m_context.m_max_num_vertices[0]));
    const uint16_t max_edge_cap =
        static_cast<uint16_t>(m_context.m_capacity_factor *
                              static_cast<float>(m_context.m_max_num_edges[0]));
    const uint16_t max_face_cap =
        static_cast<uint16_t>(m_context.m_capacity_factor *
                              static_cast<float>(m_context.m_max_num_faces[0]));

    // load EV and FE
    m_s_ev = shrd_alloc.alloc<uint16_t>(2 * edge_cap);
    m_s_fe = shrd_alloc.alloc<uint16_t>(3 * face_cap);
    detail::load_async(block,
                       reinterpret_cast<uint16_t*>(m_patch_info.ev),
                       2 * m_s_num_edges[0],
                       m_s_ev,
                       false);

    detail::load_async(block,
                       reinterpret_cast<uint16_t*>(m_patch_info.fe),
                       3 * m_s_num_faces[0],
                       m_s_fe,
                       false);

    auto alloc_masks = [&](uint16_t        num_elements,
                           Bitmask&        owned,
                           Bitmask&        active,
                           Bitmask&        ownership,
                           Bitmask&        fill_in,
                           Bitmask&        in_cavity,
                           const uint32_t* g_owned,
                           const uint32_t* g_active) {
        owned     = Bitmask(num_elements, shrd_alloc);
        active    = Bitmask(num_elements, shrd_alloc);
        ownership = Bitmask(num_elements, shrd_alloc);
        in_cavity = Bitmask(num_elements, shrd_alloc);
        fill_in   = Bitmask(num_elements, ownership.m_bitmask);

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
    alloc_masks(vert_cap,
                m_s_owned_mask_v,
                m_s_active_mask_v,
                m_s_ownership_change_mask_v,
                m_s_fill_in_v,
                m_s_in_cavity_v,
                m_patch_info.owned_mask_v,
                m_patch_info.active_mask_v);
    m_s_not_owned_cavity_bdry_v = Bitmask(vert_cap, shrd_alloc);
    m_s_owned_cavity_bdry_v     = Bitmask(vert_cap, shrd_alloc);
    m_s_connect_cavity_bdry_v   = Bitmask(vert_cap, shrd_alloc);
    m_s_src_mask_v = Bitmask(m_context.m_max_num_vertices[0], shrd_alloc);
    m_s_src_connect_mask_v =
        Bitmask(m_context.m_max_num_vertices[0], shrd_alloc);


    // edges masks
    alloc_masks(edge_cap,
                m_s_owned_mask_e,
                m_s_active_mask_e,
                m_s_ownership_change_mask_e,
                m_s_fill_in_e,
                m_s_in_cavity_e,
                m_patch_info.owned_mask_e,
                m_patch_info.active_mask_e);
    m_s_src_mask_e = Bitmask(std::max(max_edge_cap, edge_cap), shrd_alloc);
    m_s_src_connect_mask_e = Bitmask(m_context.m_max_num_edges[0], shrd_alloc);

    // faces masks
    alloc_masks(face_cap,
                m_s_owned_mask_f,
                m_s_active_mask_f,
                m_s_ownership_change_mask_f,
                m_s_fill_in_f,
                m_s_in_cavity_f,
                m_patch_info.owned_mask_f,
                m_patch_info.active_mask_f);

    // correspondence
    m_correspondence_size_e = max_edge_cap;
    m_s_q_correspondence_e =
        shrd_alloc.alloc<uint16_t>(m_correspondence_size_e);
    m_correspondence_size_vf = std::max(max_face_cap, max_vertex_cap);
    m_s_q_correspondence_vf =
        shrd_alloc.alloc<uint16_t>(m_correspondence_size_vf);

    // patch to lock
    __shared__ uint32_t p_to_lock[PatchStash::stash_size];
    m_s_patches_to_lock_mask = Bitmask(PatchStash::stash_size, p_to_lock);
    m_s_patches_to_lock_mask.reset(block);

    // locked patches
    __shared__ uint32_t p_locked[PatchStash::stash_size];
    m_s_locked_patches_mask = Bitmask(PatchStash::stash_size, p_locked);
    m_s_locked_patches_mask.reset(block);

    // cavity boundary edges
    m_s_cavity_boundary_edges = shrd_alloc.alloc<uint16_t>(m_s_num_edges[0]);

    // q hash table
    // m_s_table_q_size = std::max(
    //    std::max(m_context.m_max_lp_capacity_v,
    //    m_context.m_max_lp_capacity_e), m_context.m_max_lp_capacity_f);
    // m_s_table_q = shrd_alloc.alloc<LPPair>(m_s_table_q_size);

    //__shared__ LPPair st_q[LPHashTable::stash_size];
    // m_s_table_stash_q = st_q;
    // fill_n<blockThreads>(
    //    m_s_table_stash_q, uint16_t(LPHashTable::stash_size), LPPair());

    // lp stash
    __shared__ LPPair st_v[LPHashTable::stash_size];
    m_s_table_stash_v = st_v;

    __shared__ LPPair st_e[LPHashTable::stash_size];
    m_s_table_stash_e = st_e;

    __shared__ LPPair st_f[LPHashTable::stash_size];
    m_s_table_stash_f = st_f;

    fill_n<blockThreads>(
        m_s_table_stash_v, uint16_t(LPHashTable::stash_size), LPPair());
    fill_n<blockThreads>(
        m_s_table_stash_e, uint16_t(LPHashTable::stash_size), LPPair());
    fill_n<blockThreads>(
        m_s_table_stash_f, uint16_t(LPHashTable::stash_size), LPPair());

    // patch stash
    __shared__ uint32_t p_st[PatchStash::stash_size];
    m_s_patch_stash.m_stash = p_st;
    for (uint32_t i = threadIdx.x; i < PatchStash::stash_size;
         i += blockThreads) {
        m_s_patch_stash.m_stash[i] = m_patch_info.patch_stash.m_stash[i];
    }

    // cavity prefix sum
    // this assertion is because when we allocated dynamic shared memory
    // during kernel launch we assumed the number of cavities is at most
    // half the number of faces in the patch
    assert(m_s_num_cavities[0] <= face_cap / 2);
    m_s_cavity_size_prefix = shrd_alloc.alloc<int>(m_s_num_cavities[0] + 1);
    fill_n<blockThreads>(m_s_cavity_size_prefix, m_s_num_cavities[0] + 1, 0);

    // active cavity bitmask
    m_s_active_cavity_bitmask = Bitmask(m_s_num_cavities[0], shrd_alloc);
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

    assert(seed.patch_id() == patch_id());

    int id = ::atomicAdd(m_s_num_cavities, 1);

    assert(id < (m_context.m_max_num_faces[0] / 2));

    // there is no race condition in here since each thread is assigned to
    // one element
    if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                  cop == CavityOp::VE || cop == CavityOp::VF) {
        assert(!m_context.m_patches_info[patch_id()].is_deleted(
            LocalVertexT(seed.local_id())));
        assert(m_context.m_patches_info[patch_id()].is_owned(
            LocalVertexT(seed.local_id())));
        m_s_cavity_id_v[seed.local_id()] = id;
        m_s_cavity_creator[id]           = seed.local_id();
    }

    if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                  cop == CavityOp::EE || cop == CavityOp::EF) {
        assert(!m_context.m_patches_info[patch_id()].is_deleted(
            LocalEdgeT(seed.local_id())));
        assert(m_context.m_patches_info[patch_id()].is_owned(
            LocalEdgeT(seed.local_id())));
        m_s_cavity_id_e[seed.local_id()] = id;
        m_s_cavity_creator[id]           = seed.local_id();
    }


    if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                  cop == CavityOp::FE || cop == CavityOp::FF) {
        assert(!m_context.m_patches_info[patch_id()].is_deleted(
            LocalFaceT(seed.local_id())));
        assert(m_context.m_patches_info[patch_id()].is_owned(
            LocalFaceT(seed.local_id())));
        m_s_cavity_id_f[seed.local_id()] = id;
        m_s_cavity_creator[id]           = seed.local_id();
    }
}

template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ HandleT CavityManager<blockThreads, cop>::get_creator(
    const uint16_t cavity_id)
{
    if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                  cop == CavityOp::VE || cop == CavityOp::VF) {
        static_assert(std::is_same_v<HandleT, VertexHandle>,
                      "CavityManager::get_creator() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::V/CavityOp::VV/CavityOp::VE/CavityOp::VF, "
                      "get_creator() should return a VertexHandle");
    }

    if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                  cop == CavityOp::EE || cop == CavityOp::EF) {
        static_assert(std::is_same_v<HandleT, EdgeHandle>,
                      "CavityManager::get_creator() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::E/CavityOp::EV/CavityOp::EE/CavityOp::EF, "
                      "get_creator() should return an EdgeHandle");
    }

    if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                  cop == CavityOp::FE || cop == CavityOp::FF) {
        static_assert(std::is_same_v<HandleT, FaceHandle>,
                      "CavityManager::get_creator() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::F/CavityOp::FV/CavityOp::FE/CavityOp::FF, "
                      "get_creator() should return a FaceHandle");
    }

    assert(cavity_id < m_s_num_cavities[0]);

    return HandleT(m_patch_info.patch_id, m_s_cavity_creator[cavity_id]);
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
        assert(seed.local_id() < m_s_in_cavity_v.size());
        return m_s_in_cavity_v(seed.local_id());
    }

    if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                  cop == CavityOp::EE || cop == CavityOp::EF) {
        assert(seed.local_id() < m_s_in_cavity_e.size());
        return m_s_in_cavity_e(seed.local_id());
    }


    if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                  cop == CavityOp::FE || cop == CavityOp::FF) {
        assert(seed.local_id() < m_s_in_cavity_f.size());
        return m_s_in_cavity_f(seed.local_id());
    }
}


template <uint32_t blockThreads, CavityOp cop>
template <typename... AttributesT>
__device__ __inline__ bool CavityManager<blockThreads, cop>::prologue(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
    AttributesT&&... attributes)
{
    if (get_num_cavities() == 0) {
        return false;
    }

    // allocate shared memory
    alloc_shared_memory(block, shrd_alloc);


    // propagate the cavity ID
    propagate(block);
    block.sync();

    // Repair for conflicting cavities
    deactivate_conflicting_cavities();
    block.sync();

    // Clear bitmask for elements in the (active) cavity to indicate that
    // they are deleted (but only in shared memory)
    clear_bitmask_if_in_cavity();
    block.sync();

    // construct cavity boundary loop
    construct_cavities_edge_loop(block);
    block.sync();

    // sort each cavity edge loop
    sort_cavities_edge_loop();
    block.sync();

    // deactivate a cavity it may leave an imprint on a neighbor patch
    // deactivate_boundary_cavities(block);
    // block.sync();

    // load hashtables
    load_hashtable(block);
    block.sync();

    // change patch layout to accommodate all cavities created in the patch
    if (!migrate(block)) {
        block.sync();
        m_write_to_gmem = false;
        return false;
    }

    // mark this patch and locked patches as dirty
    m_patch_info.set_dirty();
    set_dirty_for_locked_patches();

    m_write_to_gmem = true;

    // do ownership change
    change_ownership(block);
    block.sync();

    // update attributes
    update_attributes(block, attributes...);
    block.sync();


    // reset the fill-in bitmask so we can use it during the cavity fill-in
    m_s_fill_in_v.reset(block);
    m_s_fill_in_e.reset(block);
    m_s_fill_in_f.reset(block);
    block.sync();

    return true;
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

    if constexpr (cop == CavityOp::EV) {
        mark_vertices_through_edges();
        block.sync();
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
CavityManager<blockThreads, cop>::mark_vertices_through_edges()
{
    for (uint16_t e = threadIdx.x; e < m_s_num_edges[0]; e += blockThreads) {
        assert(e < m_s_active_mask_e.size());
        if (m_s_active_mask_e(e)) {

            const uint16_t e_cavity = m_s_cavity_id_e[e];


            const uint16_t v0 = m_s_ev[2 * e + 0];
            const uint16_t v1 = m_s_ev[2 * e + 1];

            mark_element_scatter(m_s_cavity_id_v, v0, e_cavity);
            mark_element_scatter(m_s_cavity_id_v, v1, e_cavity);
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::mark_edges_through_faces()
{
    for (uint16_t f = threadIdx.x; f < m_s_num_faces[0]; f += blockThreads) {
        assert(f < m_s_active_mask_f.size());
        if (m_s_active_mask_f(f)) {

            const uint16_t f_cavity = m_s_cavity_id_f[f];

            const uint16_t e0 = m_s_fe[3 * f + 0] >> 1;
            const uint16_t e1 = m_s_fe[3 * f + 1] >> 1;
            const uint16_t e2 = m_s_fe[3 * f + 2] >> 1;

            mark_element_scatter(m_s_cavity_id_e, e0, f_cavity);
            mark_element_scatter(m_s_cavity_id_e, e1, f_cavity);
            mark_element_scatter(m_s_cavity_id_e, e2, f_cavity);
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::mark_edges_through_vertices()
{
    for (uint16_t e = threadIdx.x; e < m_s_num_edges[0]; e += blockThreads) {
        assert(e < m_s_active_mask_e.size());
        if (m_s_active_mask_e(e)) {

            // vertices tag
            const uint16_t v0 = m_s_ev[2 * e + 0];
            const uint16_t v1 = m_s_ev[2 * e + 1];

            const uint16_t c0 = m_s_cavity_id_v[v0];
            const uint16_t c1 = m_s_cavity_id_v[v1];

            mark_element_gather(m_s_cavity_id_e, e, c0);
            mark_element_gather(m_s_cavity_id_e, e, c1);
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::mark_faces_through_edges()
{
    for (uint16_t f = threadIdx.x; f < m_s_num_faces[0]; f += blockThreads) {
        assert(f < m_s_active_mask_f.size());
        if (m_s_active_mask_f(f)) {

            // edges tag
            const uint16_t e0 = m_s_fe[3 * f + 0] >> 1;
            const uint16_t e1 = m_s_fe[3 * f + 1] >> 1;
            const uint16_t e2 = m_s_fe[3 * f + 2] >> 1;

            const uint16_t c0 = m_s_cavity_id_e[e0];
            const uint16_t c1 = m_s_cavity_id_e[e1];
            const uint16_t c2 = m_s_cavity_id_e[e2];

            mark_element_gather(m_s_cavity_id_f, f, c0);
            mark_element_gather(m_s_cavity_id_f, f, c1);
            mark_element_gather(m_s_cavity_id_f, f, c2);
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::mark_element_scatter(
    uint16_t*      element_cavity_id,
    const uint16_t element_id,
    const uint16_t cavity_id)
{
    if (cavity_id != INVALID16) {
        uint16_t prv_cavity =
            atomicMin(&element_cavity_id[element_id], cavity_id);


        if (prv_cavity == cavity_id) {
            return;
        }

        if (prv_cavity < cavity_id) {
            // the vertex was marked with a cavity with lower id
            // than we deactivate this edge cavity
            deactivate_cavity(cavity_id);
        } else if (prv_cavity != INVALID16) {
            // otherwise, we deactivate the vertex cavity
            deactivate_cavity(prv_cavity);
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::mark_element_gather(
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
            deactivate_cavity(prv_element_cavity_id);
            element_cavity_id[element_id] = cavity_id;
        }

        if (prv_element_cavity_id < cavity_id) {
            // deactivate cavity ID
            deactivate_cavity(cavity_id);
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::deactivate_cavity(
    uint16_t c)
{
    assert(c < m_s_num_cavities[0]);
    m_s_active_cavity_bitmask.reset(c, true);
    m_s_cavity_creator[c] = INVALID16;
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::deactivate_conflicting_cavities()
{
    deactivate_conflicting_cavities(
        m_s_num_vertices[0], m_s_cavity_id_v, m_s_active_mask_v);

    deactivate_conflicting_cavities(
        m_s_num_edges[0], m_s_cavity_id_e, m_s_active_mask_e);

    deactivate_conflicting_cavities(
        m_s_num_faces[0], m_s_cavity_id_f, m_s_active_mask_f);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::deactivate_conflicting_cavities(
    const uint16_t num_elements,
    uint16_t*      element_cavity_id,
    const Bitmask& active_bitmask)
{
    for (uint16_t i = threadIdx.x; i < num_elements; i += blockThreads) {
        const uint32_t c = element_cavity_id[i];
        if (c != INVALID16) {
            assert(i < active_bitmask.size());
            assert(active_bitmask(i));
            assert(c < m_s_active_cavity_bitmask.size());
            if (!m_s_active_cavity_bitmask(c)) {
                element_cavity_id[i] = INVALID16;
            }
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::reactivate_elements()
{
    reactivate_elements(m_s_active_mask_v,
                        m_s_in_cavity_v,
                        m_s_cavity_id_v,
                        m_s_num_vertices[0]);
    reactivate_elements(
        m_s_active_mask_e, m_s_in_cavity_e, m_s_cavity_id_e, m_s_num_edges[0]);
    reactivate_elements(
        m_s_active_mask_f, m_s_in_cavity_f, m_s_cavity_id_f, m_s_num_faces[0]);
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::reactivate_elements(
    Bitmask&       active_bitmask,
    Bitmask&       in_cavity,
    uint16_t*      element_cavity_id,
    const uint16_t num_elements)
{
    for (uint16_t b = threadIdx.x; b < num_elements; b += blockThreads) {
        const uint16_t c = element_cavity_id[b];
        if (c != INVALID16) {
            assert(c < m_s_active_cavity_bitmask.size());
            if (!m_s_active_cavity_bitmask(c)) {
                assert(b < active_bitmask.size());
                assert(b < in_cavity.size());
                active_bitmask.set(b, true);
                in_cavity.reset(b, true);
                element_cavity_id[b] = INVALID16;
            }
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::deactivate_boundary_cavities(
    cooperative_groups::thread_block& block)
{
    m_s_owned_cavity_bdry_v.reset(block);
    block.sync();

    for (uint16_t f = threadIdx.x; f < m_s_num_faces[0]; f += blockThreads) {
        assert(f < m_s_active_mask_f.size());
        assert(f < m_s_in_cavity_f.size());
        if (m_s_active_mask_f(f) || m_s_in_cavity_f(f)) {

            const uint16_t edges[3] = {m_s_fe[3 * f + 0] >> 1,
                                       m_s_fe[3 * f + 1] >> 1,
                                       m_s_fe[3 * f + 2] >> 1};

            for (int i = 0; i < 3; ++i) {
                const uint16_t e = edges[i];
                assert(e < m_s_active_mask_e.size());
                assert(e < m_s_in_cavity_e.size());
                assert(m_s_active_mask_e(e) || m_s_in_cavity_e(e));

                const uint16_t v0 = m_s_ev[2 * e + 0];
                const uint16_t v1 = m_s_ev[2 * e + 1];

                assert(v0 < m_s_active_mask_v.size());
                assert(v1 < m_s_active_mask_v.size());
                assert(v0 < m_s_in_cavity_v.size());
                assert(v1 < m_s_in_cavity_v.size());
                assert(m_s_active_mask_v(v0) || m_s_in_cavity_v(v0));
                assert(m_s_active_mask_v(v1) || m_s_in_cavity_v(v1));

                assert(e < m_s_owned_mask_e.size());
                assert(f < m_s_owned_mask_f.size());
                if (!m_s_owned_mask_f(f) || !m_s_owned_mask_e(e)) {
                    assert(v0 < m_s_owned_cavity_bdry_v.size());
                    assert(v1 < m_s_owned_cavity_bdry_v.size());
                    m_s_owned_cavity_bdry_v.set(v0, true);
                    m_s_owned_cavity_bdry_v.set(v1, true);
                }
            }
        }
    }
    block.sync();

    for (uint16_t e = threadIdx.x; e < m_s_num_edges[0]; e += blockThreads) {
        assert(e < m_s_active_mask_e.size());
        assert(e < m_s_in_cavity_e.size());
        if (m_s_active_mask_e(e) || m_s_in_cavity_e(e)) {

            const uint16_t v0 = m_s_ev[2 * e + 0];
            const uint16_t v1 = m_s_ev[2 * e + 1];

            assert(v0 < m_s_active_mask_v.size());
            assert(v1 < m_s_active_mask_v.size());
            assert(v0 < m_s_in_cavity_v.size());
            assert(v1 < m_s_in_cavity_v.size());
            assert(m_s_active_mask_v(v0) || m_s_in_cavity_v(v0));
            assert(m_s_active_mask_v(v1) || m_s_in_cavity_v(v1));

            assert(v0 < m_s_owned_mask_v.size());
            assert(v1 < m_s_owned_mask_v.size());
            assert(e < m_s_owned_mask_e.size());
            if (!m_s_owned_mask_v(v0) || !m_s_owned_mask_v(v1) ||
                !m_s_owned_mask_e(e)) {
                assert(v0 < m_s_owned_cavity_bdry_v.size());
                assert(v1 < m_s_owned_cavity_bdry_v.size());
                m_s_owned_cavity_bdry_v.set(v0, true);
                m_s_owned_cavity_bdry_v.set(v1, true);
            }
        }
    }
    block.sync();

    for_each_cavity(block, [&](uint16_t c, uint16_t size) {
        for (uint16_t i = 0; i < size; ++i) {
            uint16_t vertex = get_cavity_vertex(c, i).local_id();
            assert(vertex < m_s_owned_mask_v.size());
            assert(vertex < m_s_owned_cavity_bdry_v.size());
            if (m_s_owned_cavity_bdry_v(vertex) || !m_s_owned_mask_v(vertex)) {
                assert(c < m_s_active_cavity_bitmask.size());
                deactivate_cavity(c);
                break;
            }
        }
    });
    block.sync();

    reactivate_elements();
    block.sync();
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
            assert(b < active_bitmask.size());
            assert(b < in_cavity.size());
            active_bitmask.reset(b, true);
            // we don't reset owned bitmask since we use it in find_copy
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


    // we have allocated m_s_cavity_boundary_edges with size equal to number of
    // edges in the patch. However, total number of edges that represent all
    // cavities boundary could be larger than the number of edges in the patch
    // since one edge could be on the boundary of two cavities.
    // Thus we prune out cavities that could have overflown
    // m_s_cavity_boundary_edges

    // deactivate the cavities
    for (uint16_t c = threadIdx.x; c < m_s_num_cavities[0]; c += blockThreads) {
        if (m_s_cavity_size_prefix[c + 1] >= m_s_num_edges[0]) {
            assert(c < m_s_active_cavity_bitmask.size());
            deactivate_cavity(c);
        }
    }
    block.sync();

    // reactivate elements that now fall in a deactivated cavity
    reactivate_elements();
    block.sync();

    for (uint16_t i = 0; i < itemPerThread; ++i) {
        if (local_offset[i] != INVALID16) {

            uint16_t f = index(i);

            const uint16_t face_cavity = m_s_cavity_id_f[f];

            if (face_cavity != INVALID16) {

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
                        assert(offset < m_s_num_edges[0]);
                        m_s_cavity_boundary_edges[offset] = e;
                        num_added++;
                    }
                };

                check_and_add(c0, e0);
                check_and_add(c1, e1);
                check_and_add(c2, e2);
            }
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::sort_cavities_edge_loop()
{

    // TODO need to increase the parallelism in this part. It should be at
    // least one warp processing one cavity
    for (uint16_t c = threadIdx.x; c < m_s_num_cavities[0]; c += blockThreads) {
        assert(c < m_s_active_cavity_bitmask.size());
        if (m_s_active_cavity_bitmask(c)) {
            // Specify the starting edge of the cavity before sorting everything
            // TODO this may be tuned for different CavityOp's

            const uint16_t start = m_s_cavity_size_prefix[c];
            const uint16_t end   = m_s_cavity_size_prefix[c + 1];

            assert(end >= start);

            if constexpr (cop == CavityOp::E) {
                // we pick one end vertex of the edge to be the starting point
                // of the cavity boundary loop
                uint16_t cavity_edge_src_vertex;
                for (uint16_t e = 0; e < m_s_num_edges[0]; ++e) {
                    if (m_s_cavity_id_e[e] == c) {
                        cavity_edge_src_vertex = m_s_ev[2 * e];
                        break;
                    }
                }

                for (uint16_t e = start; e < end; ++e) {
                    uint32_t edge = m_s_cavity_boundary_edges[e];
                    assert((edge >> 1) < m_s_active_mask_e.size());
                    assert(m_s_active_mask_e((edge >> 1)));
                    if (get_cavity_vertex(c, e - start).local_id() ==
                        cavity_edge_src_vertex) {
                        uint16_t temp = m_s_cavity_boundary_edges[start];
                        m_s_cavity_boundary_edges[start] = edge;
                        m_s_cavity_boundary_edges[e]     = temp;
                        break;
                    }
                }
            }


            for (uint16_t e = start; e < end; ++e) {
                uint16_t edge;
                uint8_t  dir;
                Context::unpack_edge_dir(
                    m_s_cavity_boundary_edges[e], edge, dir);
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
        assert(c < m_s_active_cavity_bitmask.size());
        if (m_s_active_cavity_bitmask(c)) {
            const uint16_t size = get_cavity_size(c);
            if (size > 0) {
                FillInFunc(c, size);
            }
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
    assert(c < m_s_active_cavity_bitmask.size());
    assert(m_s_active_cavity_bitmask(c));
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
    assert(c < m_s_active_cavity_bitmask.size());
    assert(m_s_active_cavity_bitmask(c));

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

    uint16_t v_id = add_element(m_s_active_mask_v,
                                m_s_num_vertices,
                                m_patch_info.vertices_capacity[0],
                                m_s_in_cavity_v,
                                m_s_owned_mask_v,
                                true,
                                false);
    if (v_id == INVALID16) {
        m_s_should_slice[0]   = true;
        m_s_remove_fill_in[0] = true;
        m_s_readd_to_queue[0] = true;
        return VertexHandle();
    }
    assert(v_id < m_patch_info.vertices_capacity[0]);
    assert(m_s_active_mask_v(v_id));
    assert(v_id < m_s_owned_mask_v.size());

    assert(v_id < m_s_fill_in_v.size());

    m_s_fill_in_v.set(v_id, true);

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

    uint16_t e_id = add_element(m_s_active_mask_e,
                                m_s_num_edges,
                                m_patch_info.edges_capacity[0],
                                m_s_in_cavity_e,
                                m_s_owned_mask_e,
                                true,
                                false);

    if (e_id == INVALID16) {
        m_s_should_slice[0]   = true;
        m_s_remove_fill_in[0] = true;
        m_s_readd_to_queue[0] = true;
        return DEdgeHandle();
    }
    assert(e_id < m_patch_info.edges_capacity[0]);
    assert(e_id < m_s_active_mask_e.size());
    assert(m_s_active_mask_e(e_id));
    assert(m_s_active_mask_v(src.local_id()));
    assert(m_s_active_mask_v(dest.local_id()));

    assert(e_id < m_s_fill_in_e.size());

    m_s_fill_in_e.set(e_id, true);

    m_s_ev[2 * e_id + 0] = src.local_id();
    m_s_ev[2 * e_id + 1] = dest.local_id();
    assert(e_id < m_s_owned_mask_e.size());
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

    uint16_t f_id = add_element(m_s_active_mask_f,
                                m_s_num_faces,
                                m_patch_info.faces_capacity[0],
                                m_s_in_cavity_f,
                                m_s_owned_mask_f,
                                true,
                                false);
    if (f_id == INVALID16) {
        m_s_should_slice[0]   = true;
        m_s_remove_fill_in[0] = true;
        m_s_readd_to_queue[0] = true;
        return FaceHandle();
    }
    assert(f_id < m_patch_info.faces_capacity[0]);
    assert(f_id < m_s_active_mask_f.size());
    assert(m_s_active_mask_f(f_id));
    assert(f_id < m_s_fill_in_f.size());

    m_s_fill_in_f.set(f_id, true);

    m_s_fe[3 * f_id + 0] = e0.unpack().second;
    m_s_fe[3 * f_id + 1] = e1.unpack().second;
    m_s_fe[3 * f_id + 2] = e2.unpack().second;


    assert(e0.get_edge_handle().local_id() < m_s_active_mask_e.size());
    assert(e1.get_edge_handle().local_id() < m_s_active_mask_e.size());
    assert(e2.get_edge_handle().local_id() < m_s_active_mask_e.size());
    assert(m_s_active_mask_e(e0.get_edge_handle().local_id()));
    assert(m_s_active_mask_e(e1.get_edge_handle().local_id()));
    assert(m_s_active_mask_e(e2.get_edge_handle().local_id()));
    assert(f_id < m_s_owned_mask_f.size());

    m_s_owned_mask_f.set(f_id, true);

    return {m_patch_info.patch_id, f_id};
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ uint16_t CavityManager<blockThreads, cop>::add_element(
    Bitmask&       active_bitmask,
    uint32_t*      num_elements,
    const uint16_t capacity,
    const Bitmask& in_cavity,
    const Bitmask& owned,
    bool           avoid_in_cavity,
    bool           avoid_not_owned_in_cavity)
{
    assert(capacity == in_cavity.size());
    assert(capacity == active_bitmask.size());
    assert(capacity == owned.size());

    uint16_t found = INVALID16;

    // number of 32-bit unsigned int used in the bit mask
    const uint32_t num32 = DIVIDE_UP(capacity, 32);

    for (uint32_t i = 0; i < num32 && found == INVALID16; ++i) {
        // flip the bits so that we are not looking for an element whose bit is
        // set
        uint32_t mask = ~active_bitmask.m_bitmask[i];
        // if there is at least one element that is not active in this 32
        // elements i.e., its bit is set
        if (mask != 0) {
            if (avoid_not_owned_in_cavity) {
                mask &= (~in_cavity.m_bitmask[i] | owned.m_bitmask[i]);
            }

            if (avoid_in_cavity) {
                mask &= ~in_cavity.m_bitmask[i];
            }
            while (mask != 0) {
                // find the first set bit
                // ffs finds the position of the least significant bit set to 1
                uint32_t first = __ffs(mask) - 1;

                // now this is the element that meet all the requirements
                uint32_t pos = 32 * i + first;

                if (pos >= capacity) {
                    break;
                }
                // try to set its bit
                assert(pos < active_bitmask.size());
                if (active_bitmask.try_set(pos)) {
                    found = pos;
                    break;
                }
                // if not successful, then we mask out this elements and try the
                // next one in this `mask` until we turn it all to zero
                mask &= ~(1 << first);
            }
        }
    }


    if (found != INVALID16) {
        assert(found < active_bitmask.size());
        ::atomicMax(num_elements, found + 1);
    }

    return found;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::get_vertices(
    const EdgeHandle eh,
    VertexHandle&    v0,
    VertexHandle&    v1)
{
    assert(eh.patch_id() == m_patch_info.patch_id);
    assert(eh.local_id() < m_s_num_edges[0]);
    // assert(m_s_active_mask_e(eh.local_id()));
    assert(m_s_owned_mask_e(eh.local_id()));

    v0 = VertexHandle(m_patch_info.patch_id, m_s_ev[2 * eh.local_id() + 0]);
    v1 = VertexHandle(m_patch_info.patch_id, m_s_ev[2 * eh.local_id() + 1]);

    // assert(m_s_active_mask_v(v0.local_id()));
    // assert(m_s_active_mask_v(v1.local_id()));

    assert(m_s_owned_mask_v(v0.local_id()));
    assert(m_s_owned_mask_v(v1.local_id()));
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::get_edges(
    const FaceHandle fh,
    EdgeHandle&      e0,
    EdgeHandle&      e1,
    EdgeHandle&      e2)
{
    assert(fh.patch_id() == m_patch_info.patch_id);
    assert(fh.local_id() < m_s_num_faces[0]);
    // assert(m_s_active_mask_e(fh.local_id()));
    assert(m_s_owned_mask_e(fh.local_id()));

    e0 = EdgeHandle(m_patch_info.patch_id, m_s_fe[3 * fh.local_id() + 0]);
    e1 = EdgeHandle(m_patch_info.patch_id, m_s_fe[3 * fh.local_id() + 1]);
    e2 = EdgeHandle(m_patch_info.patch_id, m_s_fe[3 * fh.local_id() + 2]);

    // assert(m_s_active_mask_e(e0.local_id()));
    // assert(m_s_active_mask_e(e1.local_id()));
    // assert(m_s_active_mask_e(e2.local_id()));

    assert(m_s_owned_mask_e(e0.local_id()));
    assert(m_s_owned_mask_e(e1.local_id()));
    assert(m_s_owned_mask_e(e2.local_id()));
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::load_hashtable(
    cooperative_groups::thread_block& block)
{

    m_s_table_v = reinterpret_cast<LPPair*>(m_s_cavity_id_v);
    m_s_table_e = reinterpret_cast<LPPair*>(m_s_cavity_id_e);
    m_s_table_f = reinterpret_cast<LPPair*>(m_s_cavity_id_f);

    m_patch_info.lp_v.load_in_shared_memory(
        m_s_table_v, false, m_s_table_stash_v);
    m_patch_info.lp_e.load_in_shared_memory(
        m_s_table_e, false, m_s_table_stash_e);
    m_patch_info.lp_f.load_in_shared_memory(
        m_s_table_f, true, m_s_table_stash_f);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::push()
{
    if (threadIdx.x == 0) {
        bool ret = m_context.m_patch_scheduler.push(m_patch_info.patch_id);
        assert(ret);
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::push(
    const uint32_t pid)
{
    if (threadIdx.x == 0) {
        bool ret = m_context.m_patch_scheduler.push(pid);
        assert(ret);
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
        assert(stash_id < m_s_locked_patches_mask.size());
        bool okay = m_s_locked_patches_mask(stash_id);
        if (!okay) {
            okay = m_context.m_patches_info[q].lock.acquire_lock(blockIdx.x);
            if (okay) {
                assert(stash_id < m_s_locked_patches_mask.size());
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
CavityManager<blockThreads, cop>::set_dirty_for_locked_patches()
{
    if (threadIdx.x == 0) {
        for (uint8_t st = 0; st < PatchStash::stash_size; ++st) {
            assert(st < m_s_locked_patches_mask.size());
            if (m_s_locked_patches_mask(st)) {
                uint32_t q = m_s_patch_stash.get_patch(st);
                assert(q != INVALID32);
                m_context.m_patches_info[q].set_dirty();
            }
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::unlock_locked_patches()
{
    if (threadIdx.x == 0) {
        for (uint8_t st = 0; st < m_s_locked_patches_mask.size(); ++st) {
            if (m_s_locked_patches_mask(st)) {
                uint32_t q = m_s_patch_stash.get_patch(st);
                assert(q != INVALID32);
                unlock(st, q);
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
        assert(stash_id < m_s_locked_patches_mask.size());
        assert(m_s_locked_patches_mask(stash_id));
        m_context.m_patches_info[q].lock.release_lock();
        m_s_locked_patches_mask.reset(stash_id);
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::pre_migrate(
    cooperative_groups::thread_block& block)
{
    // Some vertices on the boundary of the cavity are owned and other are
    // not. For owned vertices, edges and faces connected to them exists in
    // the patch (by definition) and they could be owned or not. For that,
    // we need to first make sure that these edges and faces are marked in
    // m_s_ownership_change_mask_e/f.
    // For not-owned vertices on the cavity boundary, we process them by
    // first marking them in m_s_not_owned_cavity_bdry_v and then look for their
    // owned version in the neighbor patches in migrate_from_patch

    m_s_connect_cavity_bdry_v.reset(block);
    m_s_owned_cavity_bdry_v.reset(block);
    m_s_not_owned_cavity_bdry_v.reset(block);
    m_s_patches_to_lock_mask.reset(block);
    m_s_locked_patches_mask.reset(block);
    m_s_ownership_change_mask_v.reset(block);
    block.sync();

    // Mark vertices on the boundary of all active cavities in this patch
    // Owned vertices are marked in m_s_owned_cavity_bdry_v and not-owned
    // vertices are marked in m_s_not_owned_cavity_bdry_v (since we need to
    // migrate them) as well as in m_s_ownership_change_mask_v (since a vertex
    // on the boundary of the cavity has to be owned by the patch)
    // TODO this could be fused in construct_cavities_edge_loop()
    for_each_cavity(block, [&](uint16_t c, uint16_t size) {
        for (uint16_t i = 0; i < size; ++i) {
            uint16_t vertex = get_cavity_vertex(c, i).local_id();
            assert(m_s_active_mask_v(vertex));
            assert(vertex < m_s_owned_mask_v.size());
            if (m_s_owned_mask_v(vertex)) {
                assert(vertex < m_s_owned_cavity_bdry_v.size());
                m_s_owned_cavity_bdry_v.set(vertex, true);
            } else {
                assert(vertex < m_s_not_owned_cavity_bdry_v.size());
                m_s_not_owned_cavity_bdry_v.set(vertex, true);
                assert(vertex < m_s_ownership_change_mask_v.size());
                m_s_ownership_change_mask_v.set(vertex, true);
            }
        }
    });
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::pre_ribbonize(
    cooperative_groups::thread_block& block)
{
    for (uint16_t e = threadIdx.x; e < m_s_num_edges[0]; e += blockThreads) {
        assert(e < m_s_active_mask_e.size());
        assert(e < m_s_in_cavity_e.size());
        if (m_s_active_mask_e(e) || m_s_in_cavity_e(e)) {

            // We want to ribbonize vertices connected to a vertex on
            // the boundary of a cavity boundaries. If the two vertices are
            // on the cavity boundaries (b0=true and b1=true), then this is
            // an edge on the cavity and we don't to ribbonize any of these
            // two vertices. Only when one of the vertices are on the cavity
            // boundaries and the other is not, we then want to ribbonize
            // the other one. Additionaly, if the other vertex is inside the
            // cavity, we don't want to ribbonize it (it will be migrated)

            const uint16_t v0 = m_s_ev[2 * e + 0];
            const uint16_t v1 = m_s_ev[2 * e + 1];

            assert(v0 < m_s_num_vertices[0]);
            assert(v1 < m_s_num_vertices[0]);
            assert(m_s_active_mask_v(v0) || m_s_in_cavity_v(v0));
            assert(m_s_active_mask_v(v1) || m_s_in_cavity_v(v1));

            assert(v0 < m_s_owned_cavity_bdry_v.size());
            assert(v0 < m_s_not_owned_cavity_bdry_v.size());
            const bool b0 =
                m_s_not_owned_cavity_bdry_v(v0) || m_s_owned_cavity_bdry_v(v0);

            assert(v1 < m_s_owned_cavity_bdry_v.size());
            assert(v1 < m_s_not_owned_cavity_bdry_v.size());
            const bool b1 =
                m_s_not_owned_cavity_bdry_v(v1) || m_s_owned_cavity_bdry_v(v1);

            assert(v1 < m_s_owned_mask_v.size());
            if (b0 && !b1 && !m_s_owned_mask_v(v1) && !m_s_in_cavity_v(v1)) {
                assert(v1 < m_s_connect_cavity_bdry_v.size());
                assert(v1 < m_s_in_cavity_v.size());
                m_s_connect_cavity_bdry_v.set(v1, true);
            }

            assert(v0 < m_s_owned_mask_v.size());
            if (b1 && !b0 && !m_s_owned_mask_v(v0) && !m_s_in_cavity_v(v0)) {
                assert(v0 < m_s_connect_cavity_bdry_v.size());
                assert(v0 < m_s_in_cavity_v.size());
                m_s_connect_cavity_bdry_v.set(v0, true);
            }
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::set_ownership_change_bitmask(
    cooperative_groups::thread_block& block)
{

    m_s_ownership_change_mask_e.reset(block);
    m_s_ownership_change_mask_f.reset(block);
    block.sync();

    for (uint16_t v = threadIdx.x; v < m_s_num_vertices[0]; v += blockThreads) {
        if (!m_s_owned_mask_v(v) && m_s_in_cavity_v(v)) {
            m_s_ownership_change_mask_v.set(v, true);
        }
    }

    for (uint16_t f = threadIdx.x; f < m_s_num_faces[0]; f += blockThreads) {
        assert(f < m_s_owned_mask_f.size());
        assert(f < m_s_active_mask_f.size());
        assert(f < m_s_in_cavity_f.size());
        if (!m_s_owned_mask_f(f) && m_s_in_cavity_f(f)) {
            m_s_ownership_change_mask_f.set(f, true);
        }
        if (!m_s_owned_mask_f(f) && m_s_active_mask_f(f)) {

            const uint16_t edges[3] = {m_s_fe[3 * f + 0] >> 1,
                                       m_s_fe[3 * f + 1] >> 1,
                                       m_s_fe[3 * f + 2] >> 1};
            for (int i = 0; i < 3; ++i) {
                const uint16_t e = edges[i];
                assert(e < m_s_active_mask_e.size());
                assert(e < m_s_in_cavity_e.size());
                assert(m_s_active_mask_e(e) || m_s_in_cavity_e(e));

                const uint16_t v0 = m_s_ev[2 * e + 0];
                const uint16_t v1 = m_s_ev[2 * e + 1];

                assert(v0 < m_s_in_cavity_v.size());
                assert(v1 < m_s_in_cavity_v.size());
                assert(m_s_active_mask_v(v0) || m_s_in_cavity_v(v0));
                assert(m_s_active_mask_v(v1) || m_s_in_cavity_v(v1));

                assert(v0 < m_s_owned_cavity_bdry_v.size());
                assert(v1 < m_s_owned_cavity_bdry_v.size());
                assert(v0 < m_s_not_owned_cavity_bdry_v.size());
                assert(v1 < m_s_not_owned_cavity_bdry_v.size());
                if (m_s_owned_cavity_bdry_v(v0) ||
                    m_s_owned_cavity_bdry_v(v1) ||
                    m_s_not_owned_cavity_bdry_v(v0) ||
                    m_s_not_owned_cavity_bdry_v(v1)) {
                    assert(f < m_s_ownership_change_mask_f.size());
                    m_s_ownership_change_mask_f.set(f, true);
                    break;
                }
            }
        }
    }


    for (uint16_t e = threadIdx.x; e < m_s_num_edges[0]; e += blockThreads) {
        assert(e < m_s_owned_mask_e.size());
        assert(e < m_s_active_mask_e.size());
        assert(e < m_s_in_cavity_e.size());
        if (!m_s_owned_mask_e(e) && m_s_in_cavity_e(e)) {
            m_s_ownership_change_mask_e.set(e, true);
        }
        if (!m_s_owned_mask_e(e) && m_s_active_mask_e(e)) {

            for (int i = 0; i < 2; ++i) {
                const uint16_t v = m_s_ev[2 * e + i];
                assert(v < m_s_in_cavity_v.size());
                assert(m_s_active_mask_v(v) || m_s_in_cavity_v(v));
                assert(v < m_s_owned_cavity_bdry_v.size());
                assert(v < m_s_not_owned_cavity_bdry_v.size());
                if (m_s_owned_cavity_bdry_v(v) ||
                    m_s_not_owned_cavity_bdry_v(v)) {
                    assert(e < m_s_ownership_change_mask_e.size());
                    m_s_ownership_change_mask_e.set(e, true);
                    break;
                }
            }
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool CavityManager<blockThreads, cop>::migrate(
    cooperative_groups::thread_block& block)
{
    pre_migrate(block);
    block.sync();
    pre_ribbonize(block);
    block.sync();

    // soft migrate
    for (uint32_t st = 0; st < PatchStash::stash_size; ++st) {
        const uint32_t q = m_s_patch_stash.get_patch(st);
        if (q != INVALID32) {
            if (!soft_migrate_from_patch(block, st, q)) {
                return false;
            }
        }
    }
    block.sync();

    // early quit if we have not locked the owner patch (but a proxy)
    if (!ensure_ownership<VertexHandle>(block,
                                        m_s_num_vertices[0],
                                        m_s_ownership_change_mask_v,
                                        m_s_table_v,
                                        m_s_table_stash_v)) {
        return false;
    }
    block.sync();


    // full migrate
    for (uint32_t st = 0; st < PatchStash::stash_size; ++st) {
        const uint32_t q = m_s_patch_stash.get_patch(st);
        if (q != INVALID32) {
            if (!migrate_from_patch(block, st, q)) {
                return false;
            }
        }
    }
    block.sync();


    // make sure non of the q patches are dirty
    for (uint8_t st = 0; st < PatchStash::stash_size; ++st) {
        assert(st < m_s_locked_patches_mask.size());
        if (m_s_locked_patches_mask(st)) {
            const uint32_t q = m_s_patch_stash.get_patch(st);
            if (m_context.m_patches_info[q].is_dirty()) {
                return false;
            }
        }
    }


    set_ownership_change_bitmask(block);
    block.sync();

    // make sure that we locked the owner (not a proxy for the owner)
    if (!ensure_ownership<VertexHandle>(block,
                                        m_s_num_vertices[0],
                                        m_s_ownership_change_mask_v,
                                        m_s_table_v,
                                        m_s_table_stash_v) ||
        !ensure_ownership<EdgeHandle>(block,
                                      m_s_num_edges[0],
                                      m_s_ownership_change_mask_e,
                                      m_s_table_e,
                                      m_s_table_stash_e) ||
        !ensure_ownership<FaceHandle>(block,
                                      m_s_num_faces[0],
                                      m_s_ownership_change_mask_f,
                                      m_s_table_f,
                                      m_s_table_stash_f)) {
        return false;
    }

    return true;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool
CavityManager<blockThreads, cop>::lock_patches_to_lock(
    cooperative_groups::thread_block& block)
{
    block.sync();
    for (uint8_t st = 0; st < PatchStash::stash_size; ++st) {
        assert(st < m_s_patches_to_lock_mask.size());
        if (m_s_patches_to_lock_mask(st)) {
            const uint32_t patch = m_s_patch_stash.get_patch(st);
            if (!lock(block, st, patch)) {
                return false;
            } else {
                assert(st < m_s_locked_patches_mask.size());
                assert(m_s_locked_patches_mask(st));
            }
        }
    }
    return true;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool
CavityManager<blockThreads, cop>::soft_migrate_from_patch(
    cooperative_groups::thread_block& block,
    const uint8_t                     q_stash_id,
    const uint32_t                    q)
{
    // Here,  we want to make sure that the 1-ring of cavity boundary vertices
    // are represented in p. This is not a full migration where we also move the
    // edges and faces incident to these vertices, but rather making sure that
    // these vertices do exist in p (owned/not-owned)
    // This also involves locking the patches we will read from during
    // migration so 1. we don't have to lock them during migrate, 2. we can fail
    // fast since if we can not lock a patch now, we could just quit

    __shared__ int s_ok_q;
    if (threadIdx.x == 0) {
        s_ok_q = 0;
    }

    // first check if the patch (q) is locked,
    // if locked, then it is safe to read from it
    // if not, then lock it and remember that it was not locked since if we
    // don't need to read from this patch (q) then we should unlock it
    assert(q_stash_id < m_s_locked_patches_mask.size());
    bool was_locked = m_s_locked_patches_mask(q_stash_id);
    block.sync();
    if (!was_locked) {
        if (!lock(block, q_stash_id, q)) {
            return false;
        } else {
            if (threadIdx.x == 0) {
                assert(q_stash_id < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(q_stash_id, true);
            }
            assert(q_stash_id < m_s_locked_patches_mask.size());
            assert(m_s_locked_patches_mask(q_stash_id));
        }
    }

    // init src_v bitmask
    m_s_src_mask_v.reset(block);
    block.sync();


    for (uint32_t v = threadIdx.x; v < m_s_num_vertices[0]; v += blockThreads) {
        assert(v < m_s_not_owned_cavity_bdry_v.size());
        if (m_s_not_owned_cavity_bdry_v(v)) {
            // get the owner patch of v

            // we don't check if this vertex is active in global memory
            // since, it could have been activated/added only in shared
            // memory (through a previous call to mirgate_from_patch)
            assert(m_s_active_mask_v(v));
            assert(v < m_s_owned_mask_v.size());
            assert(!m_s_owned_mask_v(v));

            const VertexHandle v_owner = m_patch_info.find<VertexHandle>(
                v, m_s_table_v, m_s_table_stash_v, m_s_patch_stash);

            assert(v_owner.is_valid());
            assert(v_owner.patch_id() != INVALID32);
            assert(v_owner.patch_id() != patch_id());
            assert(v_owner.local_id() != INVALID16);

            if (v_owner.patch_id() == q) {
                ::atomicAdd(&s_ok_q, 1);
                assert(v_owner.local_id() < m_s_src_mask_v.size());
                m_s_src_mask_v.set(v_owner.local_id(), true);
            }
        }
    }
    block.sync();


    if (s_ok_q != 0) {

        PatchInfo q_patch_info = m_context.m_patches_info[q];

        const uint16_t q_num_vertices = q_patch_info.num_vertices[0];
        const uint16_t q_num_edges    = q_patch_info.num_edges[0];

        // initialize connect_mask and src_e bitmask
        m_s_src_connect_mask_v.reset(block);

        block.sync();

        // in m_s_src_connect_mask_v, mark the vertices connected to
        // vertices in m_s_src_mask_v
        for (uint16_t e = threadIdx.x; e < q_num_edges; e += blockThreads) {
            if (!q_patch_info.is_deleted(LocalEdgeT(e))) {
                const uint16_t v0q = q_patch_info.ev[2 * e + 0].id;
                const uint16_t v1q = q_patch_info.ev[2 * e + 1].id;

                assert(v0q < m_s_src_mask_v.size());

                if (m_s_src_mask_v(v0q)) {
                    assert(v1q < m_s_src_connect_mask_v.size());
                    m_s_src_connect_mask_v.set(v1q, true);
                    assert(!q_patch_info.is_deleted(LocalVertexT(v1q)));
                }

                assert(v1q < m_s_src_mask_v.size());
                if (m_s_src_mask_v(v1q)) {
                    assert(v0q < m_s_src_connect_mask_v.size());
                    m_s_src_connect_mask_v.set(v0q, true);
                    assert(!q_patch_info.is_deleted(LocalVertexT(v0q)));
                }
            }
        }

        populate_correspondence<VertexHandle>(block,
                                              q_stash_id,
                                              m_s_q_correspondence_vf,
                                              m_correspondence_size_vf,
                                              m_s_table_v,
                                              m_s_table_stash_v);

        // assert(m_s_table_q_size >=
        //        m_context.m_patches_info[q].lp_v.get_capacity());
        // m_context.m_patches_info[q].lp_v.load_in_shared_memory(
        //     m_s_table_q, true, m_s_table_stash_q);

        block.sync();

        // make sure there is a copy in p for any vertex in
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

            LPPair lp = migrate_vertex(
                q,
                q_stash_id,
                q_num_vertices,
                v,
                q_patch_info,
                [&](const uint16_t vertex) {
                    assert(vertex < m_s_src_connect_mask_v.size());
                    return m_s_src_connect_mask_v(vertex);
                },
                true);
            // we need to make sure that no other
            // thread is querying the hashtable while we
            // insert in it
            block.sync();
            if (m_s_should_slice[0]) {
                return false;
            }
            if (!lp.is_sentinel()) {
                bool inserted = m_patch_info.lp_v.insert(
                    lp, m_s_table_v, m_s_table_stash_v);
                assert(inserted);
            }
            block.sync();
        }


        if (!lock_patches_to_lock(block)) {
            return false;
        }
    }

    return true;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool CavityManager<blockThreads, cop>::migrate_from_patch(
    cooperative_groups::thread_block& block,
    const uint8_t                     q_stash_id,
    const uint32_t                    q)
{
    assert(q_stash_id < m_s_locked_patches_mask.size());
    assert(m_s_locked_patches_mask(q_stash_id));
    assert(q_stash_id < m_s_patches_to_lock_mask.size());
    assert(m_s_patches_to_lock_mask(q_stash_id));

    __shared__ int s_ok_q;
    if (threadIdx.x == 0) {
        s_ok_q = 0;
    }

    // init src_v bitmask
    m_s_src_mask_v.reset(block);
    block.sync();


    for (uint32_t v = threadIdx.x; v < m_s_num_vertices[0]; v += blockThreads) {
        // migrate a vertex if it is not owned and either 1) on the cavity
        // boundary, 2) connected to a cavity boundary vertex, or 3) inside the
        // cavity
        assert(v < m_s_owned_mask_v.size());
        assert(v < m_s_not_owned_cavity_bdry_v.size());
        assert(v < m_s_connect_cavity_bdry_v.size());
        if ((!m_s_owned_mask_v(v) && m_s_connect_cavity_bdry_v(v)) ||
            m_s_not_owned_cavity_bdry_v(v) ||
            (!m_s_owned_mask_v(v) && m_s_in_cavity_v(v))) {
            // get the owner patch of v

            // we don't check if this vertex is active in global memory
            // since, it could have been activated/added only in shared
            // memory (through a previous call to mirgate_from_patch)
            assert(v < m_s_in_cavity_v.size());
            assert(m_s_active_mask_v(v) || m_s_in_cavity_v(v));

            const VertexHandle v_owner = m_patch_info.find<VertexHandle>(
                v, m_s_table_v, m_s_table_stash_v, m_s_patch_stash);

            assert(v_owner.is_valid());
            assert(v_owner.patch_id() != INVALID32);
            assert(v_owner.local_id() != INVALID16);

            if (v_owner.patch_id() == q) {

                // we no longer check if q is the actual owner
                // if it turned up that q is no longer the owner (after locking
                // q) we just quite. This check happens at the end of migrate
                // assert(m_context.m_patches_info[q].is_owned(
                //    LocalVertexT(v_owner.local_id())));

                ::atomicAdd(&s_ok_q, 1);
                assert(v_owner.local_id() < m_s_src_mask_v.size());
                m_s_src_mask_v.set(v_owner.local_id(), true);
            }
        }
    }
    block.sync();


    if (s_ok_q != 0) {
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

        // in m_s_src_connect_mask_v, mark the vertices connected to
        // vertices in m_s_src_mask_v
        for (uint16_t e = threadIdx.x; e < q_num_edges; e += blockThreads) {
            if (!q_patch_info.is_deleted(LocalEdgeT(e))) {
                const uint16_t v0q = q_patch_info.ev[2 * e + 0].id;
                const uint16_t v1q = q_patch_info.ev[2 * e + 1].id;

                assert(v0q < m_s_src_mask_v.size());
                if (m_s_src_mask_v(v0q)) {
                    assert(v1q < m_s_src_connect_mask_v.size());
                    m_s_src_connect_mask_v.set(v1q, true);
                    assert(!q_patch_info.is_deleted(LocalVertexT(v1q)));
                }

                assert(v1q < m_s_src_mask_v.size());
                if (m_s_src_mask_v(v1q)) {
                    assert(v0q < m_s_src_connect_mask_v.size());
                    m_s_src_connect_mask_v.set(v0q, true);
                    assert(!q_patch_info.is_deleted(LocalVertexT(v0q)));
                }
            }
        }

        populate_correspondence<VertexHandle>(block,
                                              q_stash_id,
                                              m_s_q_correspondence_vf,
                                              m_correspondence_size_vf,
                                              m_s_table_v,
                                              m_s_table_stash_v);

        // assert(m_s_table_q_size >=
        //        m_context.m_patches_info[q].lp_v.get_capacity());
        // m_context.m_patches_info[q].lp_v.load_in_shared_memory(
        //     m_s_table_q, true, m_s_table_stash_q);

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

            LPPair lp = migrate_vertex(
                q,
                q_stash_id,
                q_num_vertices,
                v,
                q_patch_info,
                [&](const uint16_t vertex) {
                    assert(vertex < m_s_src_connect_mask_v.size());
                    return m_s_src_connect_mask_v(vertex);
                });
            // we need to make sure that no other
            // thread is querying the hashtable while we
            // insert in it
            block.sync();
            if (m_s_should_slice[0]) {
                return false;
            }
            if (!lp.is_sentinel()) {
                bool inserted = m_patch_info.lp_v.insert(
                    lp, m_s_table_v, m_s_table_stash_v);
                assert(inserted);
            }
            block.sync();
        }


        if (!lock_patches_to_lock(block)) {
            return false;
        }

        populate_correspondence<EdgeHandle>(block,
                                            q_stash_id,
                                            m_s_q_correspondence_e,
                                            m_correspondence_size_e,
                                            m_s_table_e,
                                            m_s_table_stash_e);
        // assert(m_s_table_q_size >=
        //        m_context.m_patches_info[q].lp_e.get_capacity());
        // m_context.m_patches_info[q].lp_e.load_in_shared_memory(
        //     m_s_table_q, true, m_s_table_stash_q);

        block.sync();

        // same story as with the loop that adds vertices
        const uint16_t q_num_edges_up =
            ROUND_UP_TO_NEXT_MULTIPLE(q_num_edges, blockThreads);

        // 4. move edges since we now have a copy of the vertices in p
        for (uint16_t e = threadIdx.x; e < q_num_edges_up; e += blockThreads) {
            LPPair lp = migrate_edge(
                q,
                q_stash_id,
                q_num_edges,
                e,
                q_patch_info,
                [&](const uint16_t edge,
                    const uint16_t v0q,
                    const uint16_t v1q) {
                    // If any of these two vertices are participant in
                    // the src bitmask
                    assert(v0q < m_s_src_mask_v.size());
                    assert(v1q < m_s_src_mask_v.size());
                    if (m_s_src_mask_v(v0q) || m_s_src_mask_v(v1q)) {
                        assert(!q_patch_info.is_deleted(LocalEdgeT(edge)));
                        // set the bit for this edge in src_e mask so we
                        // can use it for migrating faces
                        assert(edge < m_s_src_mask_e.size());
                        m_s_src_mask_e.set(edge, true);
                        return true;
                    }
                    return false;
                });

            block.sync();
            if (m_s_should_slice[0]) {
                return false;
            }
            if (!lp.is_sentinel()) {
                bool inserted = m_patch_info.lp_e.insert(
                    lp, m_s_table_e, m_s_table_stash_e);
                assert(inserted);
            }
            block.sync();
        }


        if (!lock_patches_to_lock(block)) {
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

                assert(e0 < m_s_src_mask_e.size());
                assert(e1 < m_s_src_mask_e.size());
                assert(e2 < m_s_src_mask_e.size());

                bool b0 = m_s_src_mask_e(e0);
                bool b1 = m_s_src_mask_e(e1);
                bool b2 = m_s_src_mask_e(e2);

                if (b0 || b1 || b2) {
                    if (!b0) {
                        assert(e0 < m_s_src_connect_mask_e.size());
                        m_s_src_connect_mask_e.set(e0, true);
                    }
                    if (!b1) {
                        assert(e1 < m_s_src_connect_mask_e.size());
                        m_s_src_connect_mask_e.set(e1, true);
                    }
                    if (!b2) {
                        assert(e2 < m_s_src_connect_mask_e.size());
                        m_s_src_connect_mask_e.set(e2, true);
                    }
                }
            }
        }
        block.sync();

        // make sure that there is a copy of edge in
        // m_s_src_connect_mask_e in q
        for (uint16_t e = threadIdx.x; e < q_num_edges_up; e += blockThreads) {

            LPPair lp =
                migrate_edge(q,
                             q_stash_id,
                             q_num_edges,
                             e,
                             q_patch_info,
                             [&](const uint16_t edge,
                                 const uint16_t v0q,
                                 const uint16_t v1q) {
                                 assert(edge < m_s_src_connect_mask_e.size());
                                 return m_s_src_connect_mask_e(edge);
                             });
            block.sync();
            if (m_s_should_slice[0]) {
                return false;
            }
            if (!lp.is_sentinel()) {
                bool inserted = m_patch_info.lp_e.insert(
                    lp, m_s_table_e, m_s_table_stash_e);
                assert(inserted);
            }
            block.sync();
        }

        if (!lock_patches_to_lock(block)) {
            return false;
        }

        populate_correspondence<FaceHandle>(block,
                                            q_stash_id,
                                            m_s_q_correspondence_vf,
                                            m_correspondence_size_vf,
                                            m_s_table_f,
                                            m_s_table_stash_f);

        // assert(m_s_table_q_size >=
        //        m_context.m_patches_info[q].lp_f.get_capacity());
        // m_context.m_patches_info[q].lp_f.load_in_shared_memory(
        //     m_s_table_q, true, m_s_table_stash_q);

        block.sync();

        // same story as with the loop that adds vertices
        const uint16_t q_num_faces_up =
            ROUND_UP_TO_NEXT_MULTIPLE(q_num_faces, blockThreads);

        // 6.  move face since we now have a copy of the edges in p
        for (uint16_t f = threadIdx.x; f < q_num_faces_up; f += blockThreads) {
            LPPair lp = migrate_face(q,
                                     q_stash_id,
                                     q_num_faces,
                                     f,
                                     q_patch_info,
                                     [&](const uint16_t face,
                                         const uint16_t e0q,
                                         const uint16_t e1q,
                                         const uint16_t e2q) {
                                         assert(e0q < m_s_src_mask_e.size());
                                         assert(e1q < m_s_src_mask_e.size());
                                         assert(e2q < m_s_src_mask_e.size());

                                         return m_s_src_mask_e(e0q) ||
                                                m_s_src_mask_e(e1q) ||
                                                m_s_src_mask_e(e2q);
                                     });
            block.sync();
            if (m_s_should_slice[0]) {
                return false;
            }
            if (!lp.is_sentinel()) {
                bool inserted = m_patch_info.lp_f.insert(
                    lp, m_s_table_f, m_s_table_stash_f);
                assert(inserted);
            }
            block.sync();
        }

        if (!lock_patches_to_lock(block)) {
            return false;
        }
    }

    return true;
}

template <uint32_t blockThreads, CavityOp cop>
template <typename FuncT>
__device__ __inline__ LPPair CavityManager<blockThreads, cop>::migrate_vertex(
    const uint32_t q,
    const uint8_t  q_stash_id,
    const uint16_t q_num_vertices,
    const uint16_t q_vertex,
    PatchInfo&     q_patch_info,
    FuncT          should_migrate,
    bool           add_to_connect_cavity_bdry_v)
{
    LPPair ret;
    if (q_vertex < q_num_vertices &&
        !q_patch_info.is_deleted(LocalVertexT(q_vertex))) {

        if (should_migrate(q_vertex)) {
            uint16_t vq      = q_vertex;
            uint32_t o       = q;
            uint8_t  o_stash = q_stash_id;

            uint16_t vp = find_copy_vertex(vq, o, o_stash);

            // assert(!m_context.m_patches_info[o].is_deleted(LocalVertexT(vq)));
            // assert(m_context.m_patches_info[o].is_owned(LocalVertexT(vq)));

            if (vp == INVALID16) {
                vp = add_element(m_s_active_mask_v,
                                 m_s_num_vertices,
                                 m_patch_info.vertices_capacity[0],
                                 m_s_in_cavity_v,
                                 m_s_owned_mask_v,
                                 true,
                                 false);
                if (vp == INVALID16) {
                    m_s_should_slice[0] = true;
                    return ret;
                }
                assert(vp < m_patch_info.vertices_capacity[0]);

                // active bitmask is set in add_element

                // since it is owned by some other patch
                assert(vp < m_patch_info.vertices_capacity[0]);
                assert(vp < m_s_owned_mask_v.size());
                m_s_owned_mask_v.reset(vp, true);


                // insert the patch in the patch stash and return its
                // id in the stash
                const uint8_t owner_stash_id =
                    m_s_patch_stash.insert_patch(o, m_s_patch_stash_mutex);

                assert(owner_stash_id != INVALID8);
                ret = LPPair(vp, vq, owner_stash_id);

                assert(q_vertex < m_correspondence_size_vf);
                m_s_q_correspondence_vf[q_vertex] =
                    LPPair::make_value(vp, owner_stash_id);

                assert(owner_stash_id < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(owner_stash_id, true);
            } else if (o != q && o != m_patch_info.patch_id &&
                       o_stash != INVALID4) {
                assert(o_stash != INVALID8);
                assert(o_stash < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(o_stash, true);
            }
            if (add_to_connect_cavity_bdry_v) {
                assert(vp < m_s_connect_cavity_bdry_v.size());
                m_s_connect_cavity_bdry_v.set(vp, true);
            }
        }
    }
    return ret;
}


template <uint32_t blockThreads, CavityOp cop>
template <typename FuncT>
__device__ __inline__ LPPair CavityManager<blockThreads, cop>::migrate_edge(
    const uint32_t q,
    const uint8_t  q_stash_id,
    const uint16_t q_num_edges,
    const uint16_t q_edge,
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
            uint16_t eq      = q_edge;
            uint32_t o       = q;
            uint8_t  o_stash = q_stash_id;
            uint16_t ep      = find_copy_edge(eq, o, o_stash);

            // assert(!m_context.m_patches_info[o].is_deleted(LocalEdgeT(eq)));
            // assert(m_context.m_patches_info[o].is_owned(LocalEdgeT(eq)));

            if (ep == INVALID16) {
                ep = add_element(m_s_active_mask_e,
                                 m_s_num_edges,
                                 m_patch_info.edges_capacity[0],
                                 m_s_in_cavity_e,
                                 m_s_owned_mask_e,
                                 true,
                                 false);
                if (ep == INVALID16) {
                    m_s_should_slice[0] = true;
                    return ret;
                }
                assert(ep < m_patch_info.edges_capacity[0]);

                // We assume that the owner patch is q and will
                // fix this later
                uint32_t o0(q), o1(q);

                // vq -> mapped to its local index in owner
                // patch o-> mapped to the owner patch vp->
                // mapped to the corresponding local index in p
                uint8_t  o0_stash(q_stash_id), o1_stash(q_stash_id);
                uint16_t v0p = find_copy_vertex(v0q, o0, o0_stash);
                uint16_t v1p = find_copy_vertex(v1q, o1, o1_stash);

                // since any vertex in m_s_src_mask_v has been
                // added already to p, then we should find the
                // copy otherwise there is something wrong
                assert(v0p != INVALID16);
                assert(v1p != INVALID16);


                m_s_ev[2 * ep + 0] = v0p;
                m_s_ev[2 * ep + 1] = v1p;

                // active bitmask is set in add_element

                // since it is owned by some other patch
                m_s_owned_mask_e.reset(ep, true);

                const uint8_t owner_stash_id =
                    m_s_patch_stash.insert_patch(o, m_s_patch_stash_mutex);

                assert(q_edge < m_correspondence_size_e);
                m_s_q_correspondence_e[q_edge] =
                    LPPair::make_value(ep, owner_stash_id);

                assert(owner_stash_id != INVALID8);
                ret = LPPair(ep, eq, owner_stash_id);

                assert(owner_stash_id < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(owner_stash_id, true);
            } else if (o != q && o != m_patch_info.patch_id &&
                       o_stash != INVALID4) {
                assert(o_stash != INVALID8);
                assert(o_stash < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(o_stash, true);
            }
        }
    }


    return ret;
}


template <uint32_t blockThreads, CavityOp cop>
template <typename FuncT>
__device__ __inline__ LPPair CavityManager<blockThreads, cop>::migrate_face(
    const uint32_t q,
    const uint8_t  q_stash_id,
    const uint16_t q_num_faces,
    const uint16_t q_face,
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
            uint16_t fq      = q_face;
            uint32_t o       = q;
            uint8_t  o_stash = q_stash_id;
            uint16_t fp      = find_copy_face(fq, o, o_stash);


            if (fp == INVALID16) {
                fp = add_element(m_s_active_mask_f,
                                 m_s_num_faces,
                                 m_patch_info.faces_capacity[0],
                                 m_s_in_cavity_f,
                                 m_s_owned_mask_f,
                                 true,
                                 false);

                if (fp == INVALID16) {
                    m_s_should_slice[0] = true;
                    return ret;
                }
                assert(fp < m_patch_info.faces_capacity[0]);

                uint32_t o0(q), o1(q), o2(q);
                uint8_t  o0_stash(q_stash_id), o1_stash(q_stash_id),
                    o2_stash(q_stash_id);

                // eq -> mapped it to its local index in owner
                // patch o-> mapped to the owner patch ep->
                // mapped to the corresponding local index in p
                const uint16_t e0p = find_copy_edge(e0q, o0, o0_stash);
                const uint16_t e1p = find_copy_edge(e1q, o1, o1_stash);
                const uint16_t e2p = find_copy_edge(e2q, o2, o2_stash);


                // since any edge in m_s_src_mask_e has been
                // added already to p, then we should find the
                // copy otherwise there is something wrong
                assert(e0p != INVALID16);
                assert(e1p != INVALID16);
                assert(e2p != INVALID16);

                m_s_fe[3 * fp + 0] = (e0p << 1) | d0;
                m_s_fe[3 * fp + 1] = (e1p << 1) | d1;
                m_s_fe[3 * fp + 2] = (e2p << 1) | d2;

                // active bitmask is set in add_element

                // since it is owned by some other patch
                assert(fp < m_s_owned_mask_f.size());
                m_s_owned_mask_f.reset(fp, true);

                const uint8_t owner_stash_id =
                    m_s_patch_stash.insert_patch(o, m_s_patch_stash_mutex);
                assert(owner_stash_id != INVALID8);

                assert(q_face < m_correspondence_size_vf);
                m_s_q_correspondence_vf[q_face] =
                    LPPair::make_value(fp, owner_stash_id);

                ret = LPPair(fp, fq, owner_stash_id);

                assert(owner_stash_id < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(owner_stash_id, true);
            } else if (o != q && o != m_patch_info.patch_id &&
                       o_stash != INVALID4) {
                assert(o_stash != INVALID8);
                assert(o_stash < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(o_stash, true);
            }
        }
    }

    return ret;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ uint16_t
CavityManager<blockThreads, cop>::find_copy_vertex(uint16_t& local_id,
                                                   uint32_t& patch,
                                                   uint8_t&  patch_stash_id)
{
    return find_copy<VertexHandle>(local_id,
                                   patch,
                                   patch_stash_id,
                                   m_s_q_correspondence_vf,
                                   m_s_num_vertices[0],
                                   m_s_owned_mask_v,
                                   m_s_active_mask_v,
                                   m_s_in_cavity_v,
                                   m_s_table_v,
                                   m_s_table_stash_v);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ uint16_t CavityManager<blockThreads, cop>::find_copy_edge(
    uint16_t& local_id,
    uint32_t& patch,
    uint8_t&  patch_stash_id)
{
    return find_copy<EdgeHandle>(local_id,
                                 patch,
                                 patch_stash_id,
                                 m_s_q_correspondence_e,
                                 m_s_num_edges[0],
                                 m_s_owned_mask_e,
                                 m_s_active_mask_e,
                                 m_s_in_cavity_e,
                                 m_s_table_e,
                                 m_s_table_stash_e);
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ uint16_t CavityManager<blockThreads, cop>::find_copy_face(
    uint16_t& local_id,
    uint32_t& patch,
    uint8_t&  patch_stash_id)
{
    return find_copy<FaceHandle>(local_id,
                                 patch,
                                 patch_stash_id,
                                 m_s_q_correspondence_vf,
                                 m_s_num_faces[0],
                                 m_s_owned_mask_f,
                                 m_s_active_mask_f,
                                 m_s_in_cavity_f,
                                 m_s_table_f,
                                 m_s_table_stash_f);
}


template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ uint16_t CavityManager<blockThreads, cop>::find_copy(
    uint16_t&      lid,
    uint32_t&      src_patch,
    uint8_t&       src_patch_stash_id,
    uint16_t*      q_correspondence,
    const uint16_t dest_patch_num_elements,
    const Bitmask& dest_patch_owned_mask,
    const Bitmask& dest_patch_active_mask,
    const Bitmask& dest_in_cavity,
    const LPPair*  s_table,
    const LPPair*  s_stash)
{

    assert(
        !m_context.m_patches_info[src_patch].is_deleted(HandleT::LocalT(lid)));

    uint16_t corres = q_correspondence[lid];

    // we have cached this lid before
    if (corres != INVALID16) {
        src_patch_stash_id =
            detail::extract_high_bits<LPPair::PatchStashNumBits>(corres);
        src_patch = m_s_patch_stash.get_patch(src_patch_stash_id);
        uint16_t ret =
            detail::extract_low_bits<LPPair::LIDOwnerNumBits>(corres);
        return ret;
    }

    const uint16_t lid_in(lid);
    HandleT        owner;
    if (!m_context.m_patches_info[src_patch].is_owned(HandleT::LocalT(lid))) {
        owner = m_context.m_patches_info[src_patch].find<HandleT>(
            {lid} /*, m_s_table_q, m_s_table_stash_q */);

        assert(owner.is_valid());

        // if the owner src_patch is the same as the patch associated with this
        // cavity, the lid is the local index we are looking for
        src_patch = owner.patch_id();
        lid       = owner.local_id();
        if (src_patch == m_patch_info.patch_id) {
            q_correspondence[lid_in] =
                LPPair::make_value(owner.local_id(), INVALID4);
            return lid;
        }
    } else {
        // if lid is owned by q then there is no need to check the lp table
        // (because if it existed in p, then it would have shown up in the
        // correspondence array)
        return INVALID16;
    }

    // otherwise, we do a search over the not-owned elements in the dest
    // patch. For every not-owned element, we map it to its owner patch and
    // check against lid-src_patch pair
    for (uint16_t i = 0; i < dest_patch_num_elements; ++i) {
        assert(i < dest_patch_owned_mask.size());
        assert(i < dest_patch_active_mask.size());
        assert(i < dest_in_cavity.size());
        if (!dest_patch_owned_mask(i) &&
            (dest_patch_active_mask(i) || dest_in_cavity(i))) {

            const LPPair lp =
                m_patch_info.get_lp<HandleT>().find(i, s_table, s_stash);

            if (m_s_patch_stash.get_patch(lp) == src_patch &&
                lp.local_id_in_owner_patch() == lid) {
                q_correspondence[lid_in] =
                    LPPair::make_value(i, lp.patch_stash_id());
                src_patch_stash_id = lp.patch_stash_id();
                src_patch = m_s_patch_stash.get_patch(src_patch_stash_id);
                return i;
            }
        }
    }
    return INVALID16;
}

template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ void
CavityManager<blockThreads, cop>::populate_correspondence(
    cooperative_groups::thread_block& block,
    const uint8_t                     q_stash,
    uint16_t*                         s_correspondence,
    const uint16_t                    s_correspondence_size,
    const LPPair*                     s_table,
    const LPPair*                     s_stash)
{

    fill_n<blockThreads>(
        s_correspondence, s_correspondence_size, uint16_t(INVALID16));
    block.sync();

    LPHashTable lp = m_patch_info.get_lp<HandleT>();

    for (uint16_t b = threadIdx.x; b < lp.m_capacity; b += blockThreads) {
        const auto pair = s_table[b];
        if (pair.patch_stash_id() == q_stash) {
            assert(pair.local_id_in_owner_patch() < s_correspondence_size);
            s_correspondence[pair.local_id_in_owner_patch()] =
                LPPair::make_value(pair.key(), q_stash);
        }
    }


    for (uint16_t b = threadIdx.x; b < LPHashTable::stash_size;
         b += blockThreads) {
        auto pair = s_stash[b];
        if (pair.patch_stash_id() == q_stash) {
            assert(pair.local_id_in_owner_patch() < s_correspondence_size);
            s_correspondence[pair.local_id_in_owner_patch()] =
                LPPair::make_value(pair.key(), q_stash);
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ bool CavityManager<blockThreads, cop>::ensure_ownership(
    cooperative_groups::thread_block& block,
    const uint16_t                    num_elements,
    const Bitmask&                    s_ownership_change,
    const LPPair*                     s_table,
    const LPPair*                     s_stash)
{
    __shared__ bool s_all_good;
    if (threadIdx.x == 0) {
        s_all_good = true;
    }
    block.sync();

    for (uint16_t vp = threadIdx.x; vp < num_elements; vp += blockThreads) {
        assert(vp < s_ownership_change.size());
        if (s_ownership_change(vp)) {
            const HandleT h = m_patch_info.find<HandleT>(
                vp, s_table, s_stash, m_s_patch_stash);
            assert(h.patch_id() != INVALID32);
            assert(h.local_id() != INVALID16);

            const uint32_t q  = h.patch_id();
            const uint16_t vq = h.local_id();

            if (!m_context.m_patches_info[q].is_owned(HandleT::LocalT(vq))) {
                s_all_good = false;
            }
        }
    }
    block.sync();
    return s_all_good;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::change_ownership(
    cooperative_groups::thread_block& block)
{
    change_ownership<VertexHandle>(block,
                                   m_s_num_vertices[0],
                                   m_s_ownership_change_mask_v,
                                   m_s_table_v,
                                   m_s_table_stash_v,
                                   m_s_owned_mask_v);

    change_ownership<EdgeHandle>(block,
                                 m_s_num_edges[0],
                                 m_s_ownership_change_mask_e,
                                 m_s_table_e,
                                 m_s_table_stash_e,
                                 m_s_owned_mask_e);

    change_ownership<FaceHandle>(block,
                                 m_s_num_faces[0],
                                 m_s_ownership_change_mask_f,
                                 m_s_table_f,
                                 m_s_table_stash_f,
                                 m_s_owned_mask_f);
}


template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ void CavityManager<blockThreads, cop>::change_ownership(
    cooperative_groups::thread_block& block,
    const uint16_t                    num_elements,
    const Bitmask&                    s_ownership_change,
    const LPPair*                     s_table,
    const LPPair*                     s_stash,
    Bitmask&                          s_owned_bitmask)
{
    for (uint16_t vp = threadIdx.x; vp < num_elements; vp += blockThreads) {
        assert(vp < s_ownership_change.size());
        if (s_ownership_change(vp)) {
            m_s_readd_to_queue[0] = true;
            assert(vp < s_owned_bitmask.size());
            assert(!s_owned_bitmask(vp));

            const HandleT h = m_patch_info.find<HandleT>(
                vp, s_table, s_stash, m_s_patch_stash);

            assert(h.patch_id() != INVALID32);
            assert(h.local_id() != INVALID16);

            const uint32_t q  = h.patch_id();
            const uint16_t vq = h.local_id();

            // set the bitmask of this element in shared memory
            s_owned_bitmask.set(vp, true);

            // ensure patch inclusion
            assert(m_s_patch_stash.find_patch_index(q) != INVALID8);

            // make sure that q is locked
            assert(m_s_patch_stash.find_patch_index(q) <
                   m_s_locked_patches_mask.size());
            assert(
                m_s_locked_patches_mask(m_s_patch_stash.find_patch_index(q)));


            assert(q != m_patch_info.patch_id);

            assert(
                !m_context.m_patches_info[q].is_deleted(HandleT::LocalT(vq)));

            // TODO if q is no longer the owner, that means some other patch has
            // changed the ownership of vq can be explained as cavities overlap
            assert(m_context.m_patches_info[q].is_owned(HandleT::LocalT(vq)));

            // add this patch (p) to the owner's patch stash
            const uint8_t stash_id =
                m_context.m_patches_info[q].patch_stash.insert_patch(
                    m_patch_info.patch_id, m_s_patch_stash_mutex);

            assert(stash_id != INVALID8);

            // clear the bitmask of the owner's patch
            detail::bitmask_clear_bit(
                vq,
                m_context.m_patches_info[q].get_owned_mask<HandleT>(),
                true);

            // add an LP entry in the owner's patch
            LPPair lp(vq, vp, stash_id);
            if (!m_context.m_patches_info[q].get_lp<HandleT>().insert(
                    lp, nullptr, nullptr)) {
                assert(false);
            }
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
template <typename AttributeT>
__device__ __inline__ void CavityManager<blockThreads, cop>::update_attribute(
    AttributeT& attribute)
{
    using HandleT = typename AttributeT::HandleType;
    using Type    = typename AttributeT::Type;

    const uint32_t p        = m_patch_info.patch_id;
    const uint32_t num_attr = attribute.get_num_attributes();

    auto copy_from_owner =
        [&](const uint16_t vp, const LPPair* s_table, const LPPair* s_stash) {
            const HandleT h = m_patch_info.find<HandleT>(
                vp, s_table, s_stash, m_s_patch_stash);

            assert(h.patch_id() != p);
            assert(h.patch_id() != INVALID32);
            assert(h.local_id() != INVALID16);
            assert(h.patch_id() < m_context.m_max_num_patches);

            for (uint32_t attr = 0; attr < num_attr; ++attr) {
                attribute(p, vp, attr) = attribute(h, attr);
            }
        };

    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        for (uint16_t vp = threadIdx.x; vp < m_s_num_vertices[0];
             vp += blockThreads) {
            assert(vp < m_s_ownership_change_mask_v.size());
            if (m_s_ownership_change_mask_v(vp)) {

                assert(vp < m_s_owned_mask_v.size());
                assert(m_s_owned_mask_v(vp));
                assert(vp < m_s_in_cavity_v.size());
                assert(m_s_active_mask_v(vp) || m_s_in_cavity_v(vp));

                copy_from_owner(vp, m_s_table_v, m_s_table_stash_v);
            }
        }
    }

    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        for (uint16_t ep = threadIdx.x; ep < m_s_num_edges[0];
             ep += blockThreads) {
            assert(ep < m_s_ownership_change_mask_e.size());
            if (m_s_ownership_change_mask_e(ep)) {
                assert(ep < m_s_owned_mask_e.size());
                assert(m_s_owned_mask_e(ep));
                assert(ep < m_s_active_mask_e.size());
                assert(ep < m_s_in_cavity_e.size());
                assert(m_s_active_mask_e(ep) || m_s_in_cavity_e(ep));

                copy_from_owner(ep, m_s_table_e, m_s_table_stash_e);
            }
        }
    }

    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        for (uint16_t fp = threadIdx.x; fp < m_s_num_faces[0];
             fp += blockThreads) {
            assert(fp < m_s_ownership_change_mask_f.size());
            if (m_s_ownership_change_mask_f(fp)) {
                assert(fp < m_s_owned_mask_f.size());
                assert(m_s_owned_mask_f(fp));
                assert(fp < m_s_active_mask_f.size());
                assert(fp < m_s_in_cavity_f.size());
                assert(m_s_active_mask_f(fp) || m_s_in_cavity_f(fp));

                copy_from_owner(fp, m_s_table_f, m_s_table_stash_f);
            }
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::epilogue(
    cooperative_groups::thread_block& block)
{
    // make sure all writes are done
    block.sync();
    if (m_write_to_gmem) {

        // update number of elements again since add_vertex/edge/face could have
        // changed it
        if (threadIdx.x == 0) {
            m_patch_info.num_vertices[0] = m_s_num_vertices[0];
            m_patch_info.num_edges[0]    = m_s_num_edges[0];
            m_patch_info.num_faces[0]    = m_s_num_faces[0];
        }

        block.sync();

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

#ifndef NDEBUG
        for (uint16_t v = threadIdx.x; v < m_s_active_mask_v.size();
             v += blockThreads) {
            if (m_s_in_cavity_v(v)) {
                assert(!m_s_active_mask_v(v));
            }
            if (m_s_fill_in_v(v)) {
                assert(m_s_active_mask_v(v));
            }
        }
        for (uint16_t e = threadIdx.x; e < m_s_active_mask_e.size();
             e += blockThreads) {
            if (m_s_in_cavity_e(e)) {
                assert(!m_s_active_mask_e(e));
            }
            if (m_s_fill_in_e(e)) {
                assert(m_s_active_mask_e(e));
            }
        }
        for (uint16_t f = threadIdx.x; f < m_s_active_mask_f.size();
             f += blockThreads) {
            if (m_s_in_cavity_f(f)) {
                assert(!m_s_active_mask_f(f));
            }
            if (m_s_fill_in_f(f)) {
                assert(m_s_active_mask_f(f));
            }
        }
#endif

        // store bitmask
        if (m_s_remove_fill_in[0]) {
            // TODO optimize this by working on whole 32-bit mask
            //
            //  removing fill-in elements since we were not successful in adding
            //  all of them. Thus, we need to preserve the original mesh by
            //  removing these elements and re-activating the in-cavity ones
            for (uint16_t v = threadIdx.x; v < m_s_active_mask_v.size();
                 v += blockThreads) {
                if (m_s_in_cavity_v(v)) {
                    m_s_active_mask_v.set(v, true);
                }
                if (m_s_fill_in_v(v)) {
                    m_s_active_mask_v.reset(v, true);
                }
            }

            for (uint16_t e = threadIdx.x; e < m_s_active_mask_e.size();
                 e += blockThreads) {
                if (m_s_in_cavity_e(e)) {
                    m_s_active_mask_e.set(e, true);
                }
                if (m_s_fill_in_e(e)) {
                    m_s_active_mask_e.reset(e, true);
                }
            }


            for (uint16_t f = threadIdx.x; f < m_s_active_mask_f.size();
                 f += blockThreads) {
                if (m_s_in_cavity_f(f)) {
                    m_s_active_mask_f.set(f, true);
                }
                if (m_s_fill_in_f(f)) {
                    m_s_active_mask_f.reset(f, true);
                }
            }
        }
        block.sync();
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
        m_patch_info.lp_v.write_to_global_memory<blockThreads>(
            m_s_table_v, m_s_table_stash_v);
        m_patch_info.lp_e.write_to_global_memory<blockThreads>(
            m_s_table_e, m_s_table_stash_e);
        m_patch_info.lp_f.write_to_global_memory<blockThreads>(
            m_s_table_f, m_s_table_stash_f);

        // patch stash
        for (uint32_t i = threadIdx.x; i < PatchStash::stash_size;
             i += blockThreads) {
            m_patch_info.patch_stash.m_stash[i] = m_s_patch_stash.m_stash[i];
        }
    }

    if (m_s_should_slice[0]) {
        if (threadIdx.x == 0) {
            m_context.m_patches_info[patch_id()].should_slice = true;
        }
    }

    // re-add the patch to the queue if there is ownership change
    // or we could not lock all neighbor patches (and thus could not write to
    // global memory)
    if ((m_s_readd_to_queue[0] || !m_write_to_gmem) && get_num_cavities() > 0) {
        push();
    }

    // unlock any neighbor patch we have locked
    unlock_locked_patches();

    // unlock this patch
    unlock();
}
}  // namespace rxmesh