namespace rxmesh {

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ CavityManager<blockThreads, cop>::CavityManager(
    cooperative_groups::thread_block& block,
    Context&                          context,
    ShmemAllocator&                   shrd_alloc,
    bool                              preserve_cavity,
    bool                              allow_touching_cavities,
    uint32_t                          current_p)
    : m_write_to_gmem(false),
      m_context(context),
      m_preserve_cavity(preserve_cavity),
      m_allow_touching_cavities(allow_touching_cavities)
{
    static_assert(cop == CavityOp::EV || cop == CavityOp::E);

    __shared__ uint32_t s_patch_id;


    __shared__ uint32_t s_uint32[3];
    m_s_num_vertices = s_uint32 + 0;
    m_s_num_edges    = s_uint32 + 1;
    m_s_num_faces    = s_uint32 + 2;

    __shared__ bool s_bool[4];
    m_s_should_slice    = s_bool + 0;
    m_s_remove_fill_in  = s_bool + 1;
    m_s_recover         = s_bool + 2;
    m_s_new_patch_added = s_bool + 3;

    __shared__ int s_int[1];
    m_s_num_cavities = s_int;

    if (threadIdx.x == 0) {
        m_s_should_slice[0]    = false;
        m_s_remove_fill_in[0]  = false;
        m_s_recover[0]         = false;
        m_s_new_patch_added[0] = false;
        m_s_num_cavities[0]    = 0;


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

        // filter based on color
        // uint32_t color = INVALID32;
        // if (s_patch_id != INVALID32) {
        //    color = m_context.m_patches_info[s_patch_id].color;
        //    if (color != iteration) {
        //        push(s_patch_id);
        //        s_patch_id = INVALID32;
        //    }
        //}


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

            if (locked) {
                // if we lock the patch but it is dirty, we should unlock and
                // not work on it
                if (m_context.m_patches_info[s_patch_id].is_dirty()) {
                    push(s_patch_id);
                    m_context.m_patches_info[s_patch_id].lock.release_lock();
                    s_patch_id = INVALID32;
                }
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
    assert(m_s_cavity_id_v);

    m_s_cavity_id_e = reinterpret_cast<uint16_t*>(shrd_alloc.alloc(
        std::max(edge_cap_bytes, m_patch_info.lp_e.num_bytes())));
    assert(m_s_cavity_id_e);

    m_s_cavity_id_f = reinterpret_cast<uint16_t*>(shrd_alloc.alloc(
        std::max(face_cap_bytes, m_patch_info.lp_f.num_bytes())));
    assert(m_s_cavity_id_f);

    const uint16_t assumed_num_cavities = m_context.m_max_num_faces[0] / 2;
    m_s_cavity_creator = shrd_alloc.alloc<uint16_t>(assumed_num_cavities);
    assert(m_s_cavity_creator);
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

    const uint16_t max_vertex_cap =
        static_cast<uint16_t>(m_context.m_max_num_vertices[0]);
    const uint16_t max_edge_cap =
        static_cast<uint16_t>(m_context.m_max_num_edges[0]);
    const uint16_t max_face_cap =
        static_cast<uint16_t>(m_context.m_max_num_faces[0]);

    // load EV and FE
    m_s_ev = shrd_alloc.alloc<uint16_t>(2 * edge_cap);
    assert(m_s_ev);
    m_s_fe = shrd_alloc.alloc<uint16_t>(3 * face_cap);
    assert(m_s_fe);
    detail::load_async(block,
                       reinterpret_cast<uint16_t*>(m_patch_info.ev),
                       2 * m_s_num_edges[0],
                       m_s_ev,
                       true);

    detail::load_async(block,
                       reinterpret_cast<uint16_t*>(m_patch_info.fe),
                       3 * m_s_num_faces[0],
                       m_s_fe,
                       true);

    auto alloc_masks = [&](uint16_t        num_elements,
                           Bitmask&        owned,
                           Bitmask&        active,
                           Bitmask&        ownership,
                           Bitmask&        recover,
                           Bitmask&        fill_in,
                           Bitmask&        in_cavity,
                           const uint32_t* g_owned,
                           const uint32_t* g_active) {
        owned     = Bitmask(num_elements, shrd_alloc);
        active    = Bitmask(num_elements, shrd_alloc);
        ownership = Bitmask(num_elements, shrd_alloc);
        in_cavity = Bitmask(num_elements, shrd_alloc);
        fill_in   = Bitmask(num_elements, ownership.m_bitmask);
        recover   = Bitmask(num_elements, shrd_alloc);

        assert(owned.m_bitmask);
        assert(active.m_bitmask);
        assert(ownership.m_bitmask);
        assert(in_cavity.m_bitmask);
        assert(fill_in.m_bitmask);
        assert(recover.m_bitmask);

        owned.reset(block);
        active.reset(block);
        ownership.reset(block);
        in_cavity.reset(block);
        recover.reset(block);

        // to remove the racecheck hazard report due to WAW on owned and active
        block.sync();

        detail::load_async(block,
                           reinterpret_cast<const char*>(g_owned),
                           owned.num_bytes(),
                           reinterpret_cast<char*>(owned.m_bitmask),
                           true);
        detail::load_async(block,
                           reinterpret_cast<const char*>(g_active),
                           active.num_bytes(),
                           reinterpret_cast<char*>(active.m_bitmask),
                           true);
    };


    // vertices masks
    alloc_masks(vert_cap,
                m_s_owned_mask_v,
                m_s_active_mask_v,
                m_s_ownership_change_mask_v,
                m_s_recover_v,
                m_s_fill_in_v,
                m_s_in_cavity_v,
                m_patch_info.owned_mask_v,
                m_patch_info.active_mask_v);
    m_s_not_owned_cavity_bdry_v = Bitmask(vert_cap, shrd_alloc);
    assert(m_s_not_owned_cavity_bdry_v.m_bitmask);
    m_s_owned_cavity_bdry_v = Bitmask(vert_cap, shrd_alloc);
    assert(m_s_owned_cavity_bdry_v.m_bitmask);
    m_s_connect_cavity_bdry_v = Bitmask(vert_cap, shrd_alloc);
    assert(m_s_connect_cavity_bdry_v.m_bitmask);
    m_s_src_mask_v = Bitmask(m_context.m_max_num_vertices[0], shrd_alloc);
    assert(m_s_src_mask_v.m_bitmask);
    m_s_src_connect_mask_v =
        Bitmask(m_context.m_max_num_vertices[0], shrd_alloc);
    assert(m_s_src_connect_mask_v.m_bitmask);


    // edges masks
    alloc_masks(edge_cap,
                m_s_owned_mask_e,
                m_s_active_mask_e,
                m_s_ownership_change_mask_e,
                m_s_recover_e,
                m_s_fill_in_e,
                m_s_in_cavity_e,
                m_patch_info.owned_mask_e,
                m_patch_info.active_mask_e);
    m_s_src_mask_e = Bitmask(std::max(max_edge_cap, edge_cap), shrd_alloc);
    assert(m_s_src_mask_e.m_bitmask);
    m_s_src_connect_mask_e = Bitmask(m_context.m_max_num_edges[0], shrd_alloc);
    assert(m_s_src_connect_mask_e.m_bitmask);

    // faces masks
    alloc_masks(face_cap,
                m_s_owned_mask_f,
                m_s_active_mask_f,
                m_s_ownership_change_mask_f,
                m_s_recover_f,
                m_s_fill_in_f,
                m_s_in_cavity_f,
                m_patch_info.owned_mask_f,
                m_patch_info.active_mask_f);

    // correspondence
    assert(max_edge_cap >= m_s_num_edges[0]);
    m_correspondence_size_e = max_edge_cap;
    m_s_q_correspondence_e =
        shrd_alloc.alloc<uint16_t>(m_correspondence_size_e);
    assert(m_s_q_correspondence_e);
    m_s_q_correspondence_stash_e =
        shrd_alloc.alloc<uint8_t>(m_correspondence_size_e);
    assert(m_s_q_correspondence_stash_e);

    m_s_boudary_edges_cavity_id = m_s_q_correspondence_e;
    m_correspondence_size_vf    = std::max(max_face_cap, max_vertex_cap);
    m_s_q_correspondence_vf =
        shrd_alloc.alloc<uint16_t>(m_correspondence_size_vf);
    assert(m_s_q_correspondence_vf);
    m_s_q_correspondence_stash_vf =
        shrd_alloc.alloc<uint8_t>(m_correspondence_size_vf);
    assert(m_s_q_correspondence_stash_vf);

    // cavity graph
    m_s_cavity_graph = m_s_q_correspondence_e;

    assert(MAX_OVERLAP_CAVITIES * get_num_cavities() <=
           m_correspondence_size_vf + m_correspondence_size_e);

    // bitmask used for maximal independent set calculation
    assert(get_num_cavities() <= m_s_in_cavity_f.size());
    m_s_active_cavity_mis =
        Bitmask(get_num_cavities(), m_s_in_cavity_f.m_bitmask);

    m_s_cavity_mis =
        Bitmask(get_num_cavities(), m_s_ownership_change_mask_f.m_bitmask);

    m_s_candidate_cavity_mis =
        Bitmask(get_num_cavities(), m_s_recover_f.m_bitmask);

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
    assert(m_s_cavity_boundary_edges);

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
    __shared__ uint32_t p_new_st[PatchStash::stash_size];
    m_s_patch_stash.m_stash     = p_st;
    m_s_new_patch_stash.m_stash = p_new_st;

    for (int i = threadIdx.x; i < int(PatchStash::stash_size);
         i += blockThreads) {
        const uint32_t pp              = m_patch_info.patch_stash.m_stash[i];
        m_s_patch_stash.m_stash[i]     = pp;
        m_s_new_patch_stash.m_stash[i] = pp;
    }

    __shared__ uint8_t p_st2[PatchStash::stash_size];
    m_s_patch_stash_mapper = p_st2;

    // cavity prefix sum
    // this assertion is because when we allocated dynamic shared memory
    // during kernel launch we assumed the number of cavities is at most
    // half the number of faces in the patch
    assert(m_s_num_cavities[0] <= m_s_num_faces[0] / 2);
    m_s_cavity_size_prefix = shrd_alloc.alloc<int>(m_s_num_cavities[0] + 1);
    assert(m_s_cavity_size_prefix);


    m_s_cavity_graph_mutex =
        ShmemMutexArray(block, m_s_num_cavities[0], m_s_cavity_size_prefix);

    // active cavity bitmask
    m_s_active_cavity_bitmask = Bitmask(m_s_num_cavities[0], shrd_alloc);
    assert(m_s_active_cavity_bitmask.m_bitmask);
    m_s_active_cavity_bitmask.set(block);

    cooperative_groups::wait(block);
    block.sync();
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::verify_reading_from_global_memory(
    cooperative_groups::thread_block& block) const
{
    assert(m_s_num_vertices[0] == m_patch_info.num_vertices[0]);
    assert(m_s_num_edges[0] == m_patch_info.num_edges[0]);
    assert(m_s_num_faces[0] == m_patch_info.num_faces[0]);


    // active and owned vertices
    for (int v = threadIdx.x; v < int(m_s_num_vertices[0]); v += blockThreads) {
        assert(v < m_s_active_mask_v.size());
        assert(m_s_active_mask_v(v) ==
               !m_patch_info.is_deleted(LocalVertexT(v)));

        assert(m_s_owned_mask_v(v) == m_patch_info.is_owned(LocalVertexT(v)));
    }

    // active and owned edges
    for (int e = threadIdx.x; e < int(m_s_num_edges[0]); e += blockThreads) {
        assert(e < m_s_active_mask_e.size());
        assert(m_s_active_mask_e(e) == !m_patch_info.is_deleted(LocalEdgeT(e)));
        assert(m_s_owned_mask_e(e) == m_patch_info.is_owned(LocalEdgeT(e)));
    }

    // active and owned faces
    for (int f = threadIdx.x; f < int(m_s_num_faces[0]); f += blockThreads) {
        assert(f < m_s_active_mask_f.size());
        assert(m_s_active_mask_f(f) == !m_patch_info.is_deleted(LocalFaceT(f)));
        assert(m_s_owned_mask_f(f) == m_patch_info.is_owned(LocalFaceT(f)));
    }

    // EV
    for (int e = threadIdx.x; e < int(m_s_num_edges[0]); e += blockThreads) {
        if (m_s_active_mask_e(e)) {
            assert(m_s_ev[2 * e + 0] == m_patch_info.ev[2 * e + 0].id);
            assert(m_s_ev[2 * e + 1] == m_patch_info.ev[2 * e + 1].id);
        }
    }

    // FE
    for (int f = threadIdx.x; f < int(m_s_num_faces[0]); f += blockThreads) {
        if (m_s_active_mask_f(f)) {
            assert(m_s_fe[3 * f + 0] == m_patch_info.fe[3 * f + 0].id);
            assert(m_s_fe[3 * f + 1] == m_patch_info.fe[3 * f + 1].id);
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ uint32_t CavityManager<blockThreads, cop>::create(
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

    // assert(id < (m_s_num_faces[0] / 2));

    // we assume that the number of cavities is at max the number of faces/2
    // an more cavities is practically a conflicting cavity. so, the user gotta
    // attempt in next iteration or something
    // this is "practically" makes sense for all types of cavities unless the
    // cavity is created by deleting a face.
    if (id < (m_s_num_faces[0] / 2)) {
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
        return id;
    } else {
        ::atomicAdd(m_s_num_cavities, -1);
        return INVALID32;
    }
}


template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ void CavityManager<blockThreads, cop>::recover(
    HandleT seed)
{
    if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                  cop == CavityOp::VE || cop == CavityOp::VF) {
        static_assert(std::is_same_v<HandleT, VertexHandle>,
                      "CavityManager::recover() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::V/CavityOp::VV/CavityOp::VE/CavityOp::VF, "
                      "recover() should take VertexHandle as an input");
    }

    if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                  cop == CavityOp::EE || cop == CavityOp::EF) {
        static_assert(std::is_same_v<HandleT, EdgeHandle>,
                      "CavityManager::recover() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::E/CavityOp::EV/CavityOp::EE/CavityOp::EF, "
                      "recover() should take EdgeHandle as an input");
    }

    if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                  cop == CavityOp::FE || cop == CavityOp::FF) {
        static_assert(std::is_same_v<HandleT, FaceHandle>,
                      "CavityManager::recover() since CavityManager's "
                      "template parameter operation is "
                      "CavityOp::F/CavityOp::FV/CavityOp::FE/CavityOp::FF, "
                      "recover() should take FaceHandle as an input");
    }

    assert(seed.patch_id() == patch_id());

    // we can not recover if the cavity topology is not preserverd
    assert(m_preserve_cavity);


    if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                  cop == CavityOp::VE || cop == CavityOp::VF) {
        assert(!m_context.m_patches_info[patch_id()].is_deleted(
            LocalVertexT(seed.local_id())));
        assert(m_context.m_patches_info[patch_id()].is_owned(
            LocalVertexT(seed.local_id())));
        m_s_recover_v.set(seed.local_id(), true);
        m_s_recover[0] = true;
    }

    if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                  cop == CavityOp::EE || cop == CavityOp::EF) {
        assert(!m_context.m_patches_info[patch_id()].is_deleted(
            LocalEdgeT(seed.local_id())));
        assert(m_context.m_patches_info[patch_id()].is_owned(
            LocalEdgeT(seed.local_id())));
        m_s_recover_e.set(seed.local_id(), true);
        m_s_recover[0] = true;
    }


    if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                  cop == CavityOp::FE || cop == CavityOp::FF) {
        assert(!m_context.m_patches_info[patch_id()].is_deleted(
            LocalFaceT(seed.local_id())));
        assert(m_context.m_patches_info[patch_id()].is_owned(
            LocalFaceT(seed.local_id())));
        m_s_recover_f.set(seed.local_id(), true);
        m_s_recover[0] = true;
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
    // allocate shared memory
    alloc_shared_memory(block, shrd_alloc);

    if (get_num_cavities() <= 0) {
        return false;
    }

#ifndef NDEBUG
    verify_reading_from_global_memory(block);
    block.sync();
#endif

    // construct cavity graph
    construct_cavity_graph(block);
    block.sync();

    // calculate a maximal independent set of non-overlapping cavities
    calc_cavity_maximal_independent_set(block);
    block.sync();

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

    // reset the recover bitmake which maybe used during the fill-in
    m_s_recover_v.reset(block);
    m_s_recover_e.reset(block);
    m_s_recover_f.reset(block);

    block.sync();

    // store hashtable now so we could re-use the shared memory
    // overlap this memcpy with the user operations that mostly happens in
    // shared memory
    store_hashtable(block);

    return true;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::construct_cavity_graph(
    cooperative_groups::thread_block& block)
{

    fill_n<blockThreads>(m_s_cavity_graph,
                         MAX_OVERLAP_CAVITIES * get_num_cavities(),
                         uint16_t(INVALID16));
    block.sync();


    // try to add an edge between c_a and c_b
    auto add_edge_gather = [&](const uint16_t c_a, const uint16_t c_b) {
        if (c_a != INVALID16 && c_b != INVALID16 && c_a != c_b) {
            if (m_s_active_cavity_bitmask(c_a) &&
                m_s_active_cavity_bitmask(c_b)) {
                add_edge_to_cavity_graph(c_a, c_b);
            }
        }
    };

    auto add_edge_scatter = [&](uint16_t*      element_cavity_id,
                                const uint16_t element_id,
                                const uint16_t cavity_id) {
        if (cavity_id != INVALID16) {
            if (m_s_active_cavity_bitmask(cavity_id)) {
                uint16_t prv_cavity =
                    atomicMin(&element_cavity_id[element_id], cavity_id);

                if (prv_cavity == cavity_id) {
                    return;
                }
                if (prv_cavity != INVALID16) {
                    add_edge_to_cavity_graph(prv_cavity, cavity_id);
                }
            }
        }
    };

    auto is_active_cavity = [&](const uint16_t cavity_id) -> bool {
        if (cavity_id != INVALID16) {
            if (m_s_active_cavity_bitmask(cavity_id)) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    };

    auto add_graph_edge_by_faces_through_edges = [&]() {
        for (int f = threadIdx.x; f < int(m_s_num_faces[0]);
             f += blockThreads) {
            assert(f < m_s_active_mask_f.size());
            if (m_s_active_mask_f(f)) {

                assert(!m_patch_info.is_deleted(LocalFaceT(f)));

                // edges tag
                const uint16_t e0 =
                    static_cast<uint16_t>(m_s_fe[3 * f + 0] >> 1);
                const uint16_t e1 =
                    static_cast<uint16_t>(m_s_fe[3 * f + 1] >> 1);
                const uint16_t e2 =
                    static_cast<uint16_t>(m_s_fe[3 * f + 2] >> 1);

                const uint16_t c0 = m_s_cavity_id_e[e0];
                const uint16_t c1 = m_s_cavity_id_e[e1];
                const uint16_t c2 = m_s_cavity_id_e[e2];

                add_edge_gather(c0, c1);
                add_edge_gather(c1, c2);
                add_edge_gather(c2, c0);

                if (is_active_cavity(c0)) {
                    assert(m_s_active_mask_f(f));
                    m_s_cavity_id_f[f] = c0;
                } else if (is_active_cavity(c1)) {
                    assert(m_s_active_mask_f(f));
                    m_s_cavity_id_f[f] = c1;
                } else if (is_active_cavity(c2)) {
                    assert(m_s_active_mask_f(f));
                    m_s_cavity_id_f[f] = c2;
                }
            }
        }
    };


    auto add_graph_edge_by_edges_through_vertices = [&]() {
        for (int e = threadIdx.x; e < int(m_s_num_edges[0]);
             e += blockThreads) {
            assert(e < m_s_active_mask_e.size());
            if (m_s_active_mask_e(e)) {

                // vertices tag
                const uint16_t v0 = m_s_ev[2 * e + 0];
                const uint16_t v1 = m_s_ev[2 * e + 1];

                const uint16_t c0 = m_s_cavity_id_v[v0];
                const uint16_t c1 = m_s_cavity_id_v[v1];

                add_edge_gather(c0, c1);

                if (is_active_cavity(c0)) {
                    assert(m_s_active_mask_e(e));
                    m_s_cavity_id_e[e] = c0;
                } else if (is_active_cavity(c1)) {
                    assert(m_s_active_mask_e(e));
                    m_s_cavity_id_e[e] = c1;
                }
            }
        }
    };


    auto add_graph_edge_by_vertices_through_edges = [&]() {
        for (int e = threadIdx.x; e < int(m_s_num_edges[0]);
             e += blockThreads) {
            assert(e < m_s_active_mask_e.size());
            if (m_s_active_mask_e(e)) {

                const uint16_t e_cavity = m_s_cavity_id_e[e];


                const uint16_t v0 = m_s_ev[2 * e + 0];
                const uint16_t v1 = m_s_ev[2 * e + 1];

                add_edge_scatter(m_s_cavity_id_v, v0, e_cavity);
                add_edge_scatter(m_s_cavity_id_v, v1, e_cavity);
            }
        }
    };


    auto add_graph_edge_by_edges_through_faces = [&]() {
        for (int f = threadIdx.x; f < int(m_s_num_faces[0]);
             f += blockThreads) {
            assert(f < m_s_active_mask_f.size());
            if (m_s_active_mask_f(f)) {

                const uint16_t f_cavity = m_s_cavity_id_f[f];

                const uint16_t e0 =
                    static_cast<uint16_t>(m_s_fe[3 * f + 0] >> 1);
                const uint16_t e1 =
                    static_cast<uint16_t>(m_s_fe[3 * f + 1] >> 1);
                const uint16_t e2 =
                    static_cast<uint16_t>(m_s_fe[3 * f + 2] >> 1);

                add_edge_scatter(m_s_cavity_id_e, e0, f_cavity);
                add_edge_scatter(m_s_cavity_id_e, e1, f_cavity);
                add_edge_scatter(m_s_cavity_id_e, e2, f_cavity);
            }
        }
    };


    if constexpr (cop == CavityOp::EV) {
        add_graph_edge_by_vertices_through_edges();
        block.sync();
        add_graph_edge_by_edges_through_vertices();
        block.sync();
        add_graph_edge_by_faces_through_edges();
    }

    if constexpr (cop == CavityOp::E) {
        add_graph_edge_by_faces_through_edges();
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::calc_cavity_maximal_independent_set(
    cooperative_groups::thread_block& block)
{
    int num_cavities = get_num_cavities();

    __shared__ int s_num_active_cavitites_mis[1];
    if (threadIdx.x == 0) {
        s_num_active_cavitites_mis[0] = 0;
    }

    m_s_active_cavity_mis.copy(block, m_s_active_cavity_bitmask);

    m_s_cavity_mis.reset(block);

    block.sync();

    int iter = 0;

    while (true) {
        // calc number of current active cavities
        // active here means active to be considered for MIS calculation
        // for (int c = threadIdx.x; c < DIVIDE_UP(num_cavities, 32);
        //     c += blockThreads) {
        //    uint32_t mask = m_s_active_cavity_mis.m_bitmask[c];
        //    ::atomicAdd(s_num_active_cavitites_mis, __popc(mask));
        //}


        for (int c = threadIdx.x; c < num_cavities; c += blockThreads) {
            if (m_s_active_cavity_mis(c)) {
                ::atomicAdd(s_num_active_cavitites_mis, 1);
            }
        }

        // reset the candidate cavities
        m_s_candidate_cavity_mis.reset(block);
        block.sync();
        // if (threadIdx.x == 0) {
        //     printf("\n s_num_active_cavitites_mis= %u",
        //            s_num_active_cavitites_mis[0]);
        // }
        if (s_num_active_cavitites_mis[0] == 0 || iter++ > 100) {
            break;
        }


        for (int c = threadIdx.x; c < num_cavities; c += blockThreads) {
            // if the cavity is one of the active cavities for MIS calculation
            if (m_s_active_cavity_mis(c)) {
                bool add_c = true;
                for (int i = 0; i < MAX_OVERLAP_CAVITIES; ++i) {
                    const uint16_t neighbour_c =
                        m_s_cavity_graph[MAX_OVERLAP_CAVITIES * c + i];
                    if (neighbour_c != INVALID16) {

                        if (m_s_active_cavity_mis(neighbour_c) &&
                            neighbour_c > c) {
                            add_c = false;
                            break;
                        }
                    }
                }
                if (add_c) {
                    m_s_candidate_cavity_mis.set(c, true);
                }
            }
        }
        block.sync();


        // add the candidate to the MIS and remove them from the active set
        for (int c = threadIdx.x; c < num_cavities; c += blockThreads) {
            if (m_s_candidate_cavity_mis(c)) {
                m_s_active_cavity_mis.reset(c, true);
                m_s_cavity_mis.set(c, true);

                // remove the neighbour from the active set
                for (int i = 0; i < MAX_OVERLAP_CAVITIES; ++i) {
                    const uint16_t neighbour_c =
                        m_s_cavity_graph[MAX_OVERLAP_CAVITIES * c + i];
                    if (neighbour_c != INVALID16) {
                        // assert(!m_s_cavity_mis(neighbour_c));
                        m_s_active_cavity_mis.reset(neighbour_c, true);
                    }
                }
            }
        }

        if (threadIdx.x == 0) {
            s_num_active_cavitites_mis[0] = 0;
        }
        block.sync();
    }


    // deactivate cavities that are not in the MIS
    for (int c = threadIdx.x; c < num_cavities; c += blockThreads) {
        if (!m_s_cavity_mis(c)) {
            deactivate_cavity(c);
        }
    }
    block.sync();

    // clean up
    // deactivate_conflicting_cavities();

    // because it overlaps with m_s_active_cavity_mis
    m_s_in_cavity_f.reset(block);
    m_s_active_cavity_mis.reset(block);

    // because it overlaps with m_s_candidate_cavity_mis
    m_s_recover_f.reset(block);
    m_s_candidate_cavity_mis.reset(block);

    // because it overlaps with m_s_cavity_mis
    m_s_ownership_change_mask_f.reset(block);
    m_s_cavity_mis.reset(block);

    fill_n<blockThreads>(m_s_cavity_id_v,
                         m_patch_info.vertices_capacity[0],
                         uint16_t(INVALID16));
    fill_n<blockThreads>(
        m_s_cavity_id_e, m_patch_info.edges_capacity[0], uint16_t(INVALID16));
    fill_n<blockThreads>(
        m_s_cavity_id_f, m_patch_info.faces_capacity[0], uint16_t(INVALID16));

    block.sync();

    for (int c = threadIdx.x; c < num_cavities; c += blockThreads) {
        if (m_s_active_cavity_bitmask(c)) {
            const uint16_t creator = m_s_cavity_creator[c];
            assert(creator != INVALID16);
            if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                          cop == CavityOp::VE || cop == CavityOp::VF) {
                assert(m_s_active_mask_v(creator));
                m_s_cavity_id_v[creator] = c;
                assert(m_s_active_mask_v(creator));
            }

            if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                          cop == CavityOp::EE || cop == CavityOp::EF) {
                assert(m_s_active_mask_e(creator));
                m_s_cavity_id_e[creator] = c;
                assert(m_s_active_mask_e(creator));
            }

            if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                          cop == CavityOp::FE || cop == CavityOp::FF) {
                assert(m_s_active_mask_f(creator));
                m_s_cavity_id_f[creator] = c;
                assert(m_s_active_mask_f(creator));
            }
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::add_edge_to_cavity_graph(const uint16_t c0,
                                                           const uint16_t c1)
{
    auto add_edge = [&](const uint16_t from_c,
                        const uint16_t to_c) -> uint16_t {
        int i;
        m_s_cavity_graph_mutex.lock(from_c);

        for (i = 0; i < MAX_OVERLAP_CAVITIES; ++i) {
            int index = from_c * MAX_OVERLAP_CAVITIES + i;

            if (m_s_cavity_graph[index] == to_c) {
                break;
            }

            if (m_s_cavity_graph[index] == INVALID16) {
                m_s_cavity_graph[index] = to_c;
                break;
            }
        }

        m_s_cavity_graph_mutex.unlock(from_c);

        return i;
    };

    auto clear = [&](const uint16_t c, const uint16_t index) {
        m_s_cavity_graph_mutex.lock(c);
        m_s_active_cavity_bitmask.reset(c, true);
        m_s_cavity_graph[c * MAX_OVERLAP_CAVITIES + index] = INVALID16;
        m_s_cavity_graph_mutex.unlock(c);
    };

    // add c0 to c1
    uint16_t c0_id = add_edge(c0, c1);

    // add c1 to c0
    uint16_t c1_id = add_edge(c1, c0);

    if (c0_id < MAX_OVERLAP_CAVITIES && c1_id < MAX_OVERLAP_CAVITIES) {
        return;
    }

    // decide which cavity to deactivate
    // we choose the one with more overlaps
    // printf("\n deactiaving !!!!!!!!!!! ");
    if (c0_id > c1_id) {
        clear(c0, c0_id);
    } else {
        clear(c1, c1_id);
    }
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
    for (int e = threadIdx.x; e < int(m_s_num_edges[0]); e += blockThreads) {
        assert(e < m_s_active_mask_e.size());
        if (m_s_active_mask_e(e)) {

            const uint16_t e_cavity = m_s_cavity_id_e[e];


            const uint16_t v0 = m_s_ev[2 * e + 0];
            const uint16_t v1 = m_s_ev[2 * e + 1];

            assert(m_s_active_mask_v(v0));
            assert(m_s_active_mask_v(v1));

            mark_element_scatter(m_s_cavity_id_v, v0, e_cavity);
            mark_element_scatter(m_s_cavity_id_v, v1, e_cavity);
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::mark_edges_through_faces()
{
    for (int f = threadIdx.x; f < int(m_s_num_faces[0]); f += blockThreads) {
        assert(f < m_s_active_mask_f.size());
        if (m_s_active_mask_f(f)) {

            const uint16_t f_cavity = m_s_cavity_id_f[f];

            const uint16_t e0 = static_cast<uint16_t>(m_s_fe[3 * f + 0] >> 1);
            const uint16_t e1 = static_cast<uint16_t>(m_s_fe[3 * f + 1] >> 1);
            const uint16_t e2 = static_cast<uint16_t>(m_s_fe[3 * f + 2] >> 1);

            assert(m_s_active_mask_e(e0));
            assert(m_s_active_mask_e(e0));
            assert(m_s_active_mask_e(e0));

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
    for (int e = threadIdx.x; e < int(m_s_num_edges[0]); e += blockThreads) {
        assert(e < m_s_active_mask_e.size());
        if (m_s_active_mask_e(e)) {

            // vertices tag
            const uint16_t v0 = m_s_ev[2 * e + 0];
            const uint16_t v1 = m_s_ev[2 * e + 1];

            const uint16_t c0 = m_s_cavity_id_v[v0];
            const uint16_t c1 = m_s_cavity_id_v[v1];

            assert(m_s_active_mask_v(v0));
            assert(m_s_active_mask_v(v1));

            mark_element_gather(m_s_cavity_id_e, e, c0);
            mark_element_gather(m_s_cavity_id_e, e, c1);
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::mark_faces_through_edges()
{
    for (int f = threadIdx.x; f < int(m_s_num_faces[0]); f += blockThreads) {
        assert(f < m_s_active_mask_f.size());
        if (m_s_active_mask_f(f)) {

            // edges tag
            const uint16_t e0 = static_cast<uint16_t>(m_s_fe[3 * f + 0] >> 1);
            const uint16_t e1 = static_cast<uint16_t>(m_s_fe[3 * f + 1] >> 1);
            const uint16_t e2 = static_cast<uint16_t>(m_s_fe[3 * f + 2] >> 1);


            assert(!m_patch_info.is_deleted(LocalFaceT(f)));

            assert(!m_patch_info.is_deleted(LocalEdgeT(e0)));
            assert(!m_patch_info.is_deleted(LocalEdgeT(e1)));
            assert(!m_patch_info.is_deleted(LocalEdgeT(e2)));

            assert(m_s_active_mask_e(e0));
            assert(m_s_active_mask_e(e1));
            assert(m_s_active_mask_e(e2));

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
        if (m_s_active_cavity_bitmask(cavity_id)) {
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
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::mark_element_gather(
    uint16_t*      element_cavity_id,
    const uint16_t element_id,
    const uint16_t cavity_id)
{
    if (cavity_id != INVALID16) {
        if (m_s_active_cavity_bitmask(cavity_id)) {
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
                deactivate_cavity(prv_element_cavity_id);
                element_cavity_id[element_id] = cavity_id;
                // printf("\n A T= %u, F= %u - %u deactivated %u",
                //        threadIdx.x,
                //        element_id,
                //        cavity_id,
                //        prv_element_cavity_id);
            }

            if (prv_element_cavity_id < cavity_id) {
                // deactivate cavity ID
                deactivate_cavity(cavity_id);
                // printf("\n B T= %u, F= %u - %u deactivated %u",
                //        threadIdx.x,
                //        element_id,
                //        prv_element_cavity_id,
                //        cavity_id);
            }
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
    for (int i = threadIdx.x; i < int(num_elements); i += blockThreads) {
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
    for (int b = threadIdx.x; b < int(num_elements); b += blockThreads) {
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

    for (int f = threadIdx.x; f < int(m_s_num_faces[0]); f += blockThreads) {
        assert(f < m_s_active_mask_f.size());
        assert(f < m_s_in_cavity_f.size());
        if (m_s_active_mask_f(f) || m_s_in_cavity_f(f)) {

            const uint16_t edges[3] = {
                static_cast<uint16_t>(m_s_fe[3 * f + 0] >> 1),
                static_cast<uint16_t>(m_s_fe[3 * f + 1] >> 1),
                static_cast<uint16_t>(m_s_fe[3 * f + 2] >> 1)};

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
                if (!m_s_owned_mask_f(f) || !m_s_owned_mask_e(e) ||
                    !m_s_owned_mask_v(v0) || !m_s_owned_mask_v(v1)) {
                    assert(v0 < m_s_owned_cavity_bdry_v.size());
                    assert(v1 < m_s_owned_cavity_bdry_v.size());
                    m_s_owned_cavity_bdry_v.set(v0, true);
                    m_s_owned_cavity_bdry_v.set(v1, true);
                }
            }
        }
    }
    block.sync();

    for (int e = threadIdx.x; e < int(m_s_num_edges[0]); e += blockThreads) {
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
        for (int i = 0; i < int(size); ++i) {
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
    for (int b = threadIdx.x; b < int(num_elements); b += blockThreads) {
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
template <int itemPerThread>
__device__ __inline__ void
CavityManager<blockThreads, cop>::construct_cavities_edge_loop(
    cooperative_groups::thread_block& block)
{
    if (!m_allow_touching_cavities) {
        fill_n<blockThreads>(
            m_s_boudary_edges_cavity_id, m_s_num_edges[0], uint16_t(INVALID16));
    }

    fill_n<blockThreads>(m_s_cavity_size_prefix, m_s_num_cavities[0] + 1, 0);
    block.sync();

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

    for (int i = 0; i < itemPerThread; ++i) {
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
            const uint16_t e0 = static_cast<uint16_t>(m_s_fe[3 * f + 0] >> 1);
            const uint16_t e1 = static_cast<uint16_t>(m_s_fe[3 * f + 1] >> 1);
            const uint16_t e2 = static_cast<uint16_t>(m_s_fe[3 * f + 2] >> 1);

            const uint16_t c0 = m_s_cavity_id_e[e0];
            const uint16_t c1 = m_s_cavity_id_e[e1];
            const uint16_t c2 = m_s_cavity_id_e[e2];

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

                if (!m_allow_touching_cavities) {

                    // we use here atomicMin so at least one cavity will move on
                    // if there is too many competing for the same edge
                    if (c0 == INVALID16) {
                        atomicMin(m_s_boudary_edges_cavity_id + e0,
                                  face_cavity);
                        // m_s_boudary_edges_cavity_id[e0] = face_cavity;
                    }
                    if (c1 == INVALID16) {
                        atomicMin(m_s_boudary_edges_cavity_id + e1,
                                  face_cavity);
                        // m_s_boudary_edges_cavity_id[e1] = face_cavity;
                    }
                    if (c2 == INVALID16) {
                        atomicMin(m_s_boudary_edges_cavity_id + e2,
                                  face_cavity);
                        // m_s_boudary_edges_cavity_id[e2] = face_cavity;
                    }
                }
            }
        }
    }
    block.sync();

    // scan
    detail::cub_block_exclusive_sum<int, blockThreads>(m_s_cavity_size_prefix,
                                                       m_s_num_cavities[0]);
    block.sync();


    if (!m_allow_touching_cavities) {
        for (int f = threadIdx.x; f < int(m_s_num_faces[0]);
             f += blockThreads) {
            uint16_t face_cavity = m_s_cavity_id_f[f];
            if (face_cavity != INVALID16) {
                for (int i = 0; i < 3; ++i) {

                    const uint16_t e =
                        static_cast<uint16_t>(m_s_fe[3 * f + i] >> 1);

                    const uint16_t e_bd_cavity_id =
                        m_s_boudary_edges_cavity_id[e];

                    if (e_bd_cavity_id != INVALID16 &&
                        e_bd_cavity_id != face_cavity) {
                        deactivate_cavity(face_cavity);
                    }
                }
            }
        }
    }
    block.sync();

    // we have allocated m_s_cavity_boundary_edges with size equal to number of
    // edges in the patch. However, total number of edges that represent all
    // cavities boundary could be larger than the number of edges in the patch
    // since one edge could be on the boundary of two cavities.
    // Thus we prune out cavities that could have overflown
    // m_s_cavity_boundary_edges

    // deactivate the cavities
    for (int c = threadIdx.x; c < int(m_s_num_cavities[0]); c += blockThreads) {
        if (m_s_cavity_size_prefix[c + 1] >= m_s_num_edges[0]) {
            assert(c < m_s_active_cavity_bitmask.size());
            deactivate_cavity(c);
        }
    }
    block.sync();

    // reactivate elements that now fall in a deactivated cavity
    reactivate_elements();
    block.sync();

    for (int i = 0; i < itemPerThread; ++i) {
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
    for (int c = threadIdx.x; c < int(m_s_num_cavities[0]); c += blockThreads) {
        assert(c < m_s_active_cavity_bitmask.size());
        if (m_s_active_cavity_bitmask(c)) {
            // Specify the starting edge of the cavity before sorting everything
            // TODO this may be tuned for different CavityOp's

            const int start = int(m_s_cavity_size_prefix[c]);
            const int end   = int(m_s_cavity_size_prefix[c + 1]);

            assert(end >= start);

            if constexpr (cop == CavityOp::E) {
                // we pick one end vertex of the edge to be the starting point
                // of the cavity boundary loop
                uint16_t cavity_edge_src_vertex;
                for (int e = 0; e < m_s_num_edges[0]; ++e) {
                    if (m_s_cavity_id_e[e] == c) {
                        cavity_edge_src_vertex = m_s_ev[2 * e];
                        break;
                    }
                }

                for (int e = start; e < end; ++e) {
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


            for (int e = start; e < end; ++e) {
                uint16_t edge;
                uint8_t  dir;
                Context::unpack_edge_dir(
                    m_s_cavity_boundary_edges[e], edge, dir);
                uint16_t end_vertex = m_s_ev[2 * edge + 1];
                if (dir) {
                    end_vertex = m_s_ev[2 * edge];
                }

                for (int i = e + 1; i < end; ++i) {
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
    for (int c = threadIdx.x; c < int(m_s_num_cavities[0]); c += blockThreads) {
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
                                m_preserve_cavity,
                                !m_preserve_cavity);
    if (v_id == INVALID16) {
        m_s_should_slice[0]   = true;
        m_s_remove_fill_in[0] = true;
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
                                m_preserve_cavity,
                                !m_preserve_cavity);

    if (e_id == INVALID16) {
        m_s_should_slice[0]   = true;
        m_s_remove_fill_in[0] = true;
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
                                m_preserve_cavity,
                                !m_preserve_cavity);
    if (f_id == INVALID16) {
        m_s_should_slice[0]   = true;
        m_s_remove_fill_in[0] = true;
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
    const int num32 = DIVIDE_UP(capacity, 32);

    for (int i = 0; i < num32 && found == INVALID16; ++i) {
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
__device__ __inline__ void CavityManager<blockThreads, cop>::store_hashtable(
    cooperative_groups::thread_block& block)
{

    // store hashtable
    m_patch_info.lp_v.write_to_global_memory<blockThreads>(m_s_table_v,
                                                           m_s_table_stash_v);
    m_patch_info.lp_e.write_to_global_memory<blockThreads>(m_s_table_e,
                                                           m_s_table_stash_e);
    m_patch_info.lp_f.write_to_global_memory<blockThreads>(m_s_table_f,
                                                           m_s_table_stash_f);

    // patch stash
    for (int i = threadIdx.x; i < PatchStash::stash_size; i += blockThreads) {
        m_patch_info.patch_stash.m_stash[i] = m_s_patch_stash.m_stash[i];
    }
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
                m_s_patches_to_lock_mask.set(stash_id);
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
        for (int st = 0; st < PatchStash::stash_size; ++st) {
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
        for (int st = 0; st < m_s_locked_patches_mask.size(); ++st) {
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
        for (int i = 0; i < int(size); ++i) {
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
    for (int e = threadIdx.x; e < int(m_s_num_edges[0]); e += blockThreads) {
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

    for (int v = threadIdx.x; v < int(m_s_num_vertices[0]); v += blockThreads) {
        if (!m_s_owned_mask_v(v) && m_s_in_cavity_v(v)) {
            m_s_ownership_change_mask_v.set(v, true);
        }
    }

    for (int f = threadIdx.x; f < int(m_s_num_faces[0]); f += blockThreads) {
        assert(f < m_s_owned_mask_f.size());
        assert(f < m_s_active_mask_f.size());
        assert(f < m_s_in_cavity_f.size());

        const uint16_t edges[3] = {
            static_cast<uint16_t>(m_s_fe[3 * f + 0] >> 1),
            static_cast<uint16_t>(m_s_fe[3 * f + 1] >> 1),
            static_cast<uint16_t>(m_s_fe[3 * f + 2] >> 1)};

        if (!m_s_owned_mask_f(f) && m_s_in_cavity_f(f)) {
            m_s_ownership_change_mask_f.set(f, true);
        }
        if (!m_s_owned_mask_f(f) && m_s_active_mask_f(f)) {

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

        if (m_s_ownership_change_mask_f(f)) {
            for (int e = 0; e < 3; ++e) {
                if (!m_s_owned_mask_e(edges[e]) &&
                    m_s_active_mask_e(edges[e])) {
                    m_s_ownership_change_mask_e.set(edges[e], true);
                }
            }
        }
    }


    for (int e = threadIdx.x; e < int(m_s_num_edges[0]); e += blockThreads) {
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

    // lock all neighbors and make sure non is dirty
    if (!lock_neighbour_patches(block)) {
        return false;
    }

    // make sure non of the locked q patches are dirty
    if (!ensure_locked_patches_are_not_dirty()) {
        return false;
    }


    // soft migrate
    for (int st = 0; st < PatchStash::stash_size; ++st) {
        const uint32_t q = m_s_patch_stash.get_patch(st);
        if (q != INVALID32) {
            if (!soft_migrate_from_patch(block, st, q)) {
                return false;
            }
        }
    }
    block.sync();

    if (!ensure_ownership<VertexHandle>(block,
                                        m_s_num_vertices[0],
                                        m_s_ownership_change_mask_v,
                                        m_s_table_v,
                                        m_s_table_stash_v)) {
        return false;
    }

    // make sure non of the locked q patches are dirty
    if (!ensure_locked_patches_are_not_dirty()) {
        return false;
    }

    // full migrate
    for (int st = 0; st < PatchStash::stash_size; ++st) {
        const uint32_t q = m_s_patch_stash.get_patch(st);
        if (q != INVALID32) {
            if (!migrate_from_patch(block, st, q)) {
                return false;
            }
        }
    }
    block.sync();

    // since we may have locked new patches during full migration, make sure non
    // of the q patches are dirty
    if (!ensure_locked_patches_are_not_dirty()) {
        return false;
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
CavityManager<blockThreads, cop>::lock_neighbour_patches(
    cooperative_groups::thread_block& block)
{
    // TODO since the whole block locks the whole thing, then we can use
    // different warps to locks different patches rather than relies on a single
    // thread to do the job
    block.sync();
    for (int st = 0; st < PatchStash::stash_size; ++st) {
        assert(st < m_s_locked_patches_mask.size());
        const uint32_t patch = m_s_patch_stash.get_patch(st);
        if (patch != INVALID32) {
            if (!lock(block, st, patch)) {
                return false;
            } else {
                assert(m_s_locked_patches_mask(st));
            }
        }
    }
    return true;
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool
CavityManager<blockThreads, cop>::lock_new_added_patches(
    cooperative_groups::thread_block& block)
{
    block.sync();
    for (int st = 0; st < PatchStash::stash_size; ++st) {
        if (m_s_patch_stash.get_patch(st) !=
            m_s_new_patch_stash.get_patch(st)) {
            // it is a new patch
            uint32_t new_patch = m_s_patch_stash.get_patch(st);

            if (!lock(block, st, new_patch)) {
                return false;
            } else {
                assert(st < m_s_locked_patches_mask.size());
                assert(m_s_locked_patches_mask(st));
                if (threadIdx.x == 0) {
                    m_s_new_patch_stash.m_stash[st] = new_patch;
                }
            }
        }
    }
    return true;
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool
CavityManager<blockThreads, cop>::lock_patches_to_lock(
    cooperative_groups::thread_block& block)
{
    block.sync();
    for (int st = 0; st < PatchStash::stash_size; ++st) {
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

    // initialize connect_mask and src_e bitmask
    m_s_src_connect_mask_v.reset(block);
    block.sync();


    for (int v = threadIdx.x; v < int(m_s_num_vertices[0]); v += blockThreads) {
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


        // in m_s_src_connect_mask_v, mark the vertices connected to
        // vertices in m_s_src_mask_v
        for (int e = threadIdx.x; e < int(q_num_edges); e += blockThreads) {
            if (!q_patch_info.is_deleted(LocalEdgeT(e))) {

                auto [v0q, v1q] = q_patch_info.get_edge_vertices(e);

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
                                              q_patch_info,
                                              q_stash_id,
                                              m_s_q_correspondence_vf,
                                              m_s_q_correspondence_stash_vf,
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
        for (int v = threadIdx.x; v < int(q_num_vertices_up);
             v += blockThreads) {
            if (m_s_should_slice[0]) {
                return false;
            }
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
                if (!inserted) {
                    m_s_should_slice[0] = true;
                }
            }
            block.sync();
        }
        if (m_s_should_slice[0]) {
            return false;
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


    for (int v = threadIdx.x; v < int(m_s_num_vertices[0]); v += blockThreads) {
        // migrate a vertex if it is not owned and either 1) on the cavity
        // boundary, 2) connected to a cavity boundary vertex, or 3) inside
        // the cavity
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
                // if it turned up that q is no longer the owner (after
                // locking q) we just quite. This check happens at the end
                // of migrate assert(m_context.m_patches_info[q].is_owned(
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
        for (int e = threadIdx.x; e < int(q_num_edges); e += blockThreads) {
            if (!q_patch_info.is_deleted(LocalEdgeT(e))) {

                auto [v0q, v1q] = q_patch_info.get_edge_vertices(e);


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
                                              q_patch_info,
                                              q_stash_id,
                                              m_s_q_correspondence_vf,
                                              m_s_q_correspondence_stash_vf,
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
        for (int v = threadIdx.x; v < int(q_num_vertices_up);
             v += blockThreads) {
            if (m_s_should_slice[0]) {
                return false;
            }
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
                if (!inserted) {
                    m_s_should_slice[0] = true;
                }
                // if (!inserted) {
                //     printf("\n p= %u, load factor = %f, stash load factor =
                //     %f",
                //            patch_id(),
                //            m_patch_info.lp_v.compute_load_factor(m_s_table_v),
                //            m_patch_info.lp_v.compute_load_factor(
                //                m_s_table_stash_v));
                // }
                //  assert(inserted);
            }
            block.sync();
        }
        if (m_s_should_slice[0]) {
            return false;
        }

        if (!lock_patches_to_lock(block)) {
            return false;
        }

        populate_correspondence<EdgeHandle>(block,
                                            q_patch_info,
                                            q_stash_id,
                                            m_s_q_correspondence_e,
                                            m_s_q_correspondence_stash_e,
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
        for (int e = threadIdx.x; e < int(q_num_edges_up); e += blockThreads) {
            if (m_s_should_slice[0]) {
                return false;
            }
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
                if (!inserted) {
                    m_s_should_slice[0] = true;
                }
            }
            block.sync();
        }
        if (m_s_should_slice[0]) {
            return false;
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
        for (int f = threadIdx.x; f < int(q_num_faces); f += blockThreads) {
            if (!q_patch_info.is_deleted(LocalFaceT(f))) {

                const uint16_t e0 =
                    static_cast<uint16_t>(q_patch_info.fe[3 * f + 0].id >> 1);
                const uint16_t e1 =
                    static_cast<uint16_t>(q_patch_info.fe[3 * f + 1].id >> 1);
                const uint16_t e2 =
                    static_cast<uint16_t>(q_patch_info.fe[3 * f + 2].id >> 1);

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


                    // uint16_t v0q = q_patch_info.ev[2 * e0 + 0].id;
                    // uint16_t v1q = q_patch_info.ev[2 * e0 + 1].id;
                    //
                    // uint16_t v2q = (q_patch_info.ev[2 * e1 + 1].id == v0q ||
                    //                 q_patch_info.ev[2 * e1 + 1].id == v1q) ?
                    //                    q_patch_info.ev[2 * e1 + 0].id :
                    //                    q_patch_info.ev[2 * e1 + 1].id;
                    //
                    // if (!(v0q != v1q && v0q != v2q && v1q != v2q)) {
                    //     printf(
                    //         "\n ## e0v=%u, %u, e1v=%u, %u, e2v=%u, %u, dirty
                    //         = "
                    //         "%d, is_locked = %d",
                    //         q_patch_info.ev[2 * e0 + 0].id,
                    //         q_patch_info.ev[2 * e0 + 1].id,
                    //         q_patch_info.ev[2 * e1 + 0].id,
                    //         q_patch_info.ev[2 * e1 + 1].id,
                    //         q_patch_info.ev[2 * e2 + 0].id,
                    //         q_patch_info.ev[2 * e2 + 1].id,
                    //         m_context.m_patches_info[q].is_dirty(),
                    //         m_context.m_patches_info[q].lock.is_locked());
                    // }
                    //
                    // if ((!m_s_src_connect_mask_v(v0q) &&
                    //      !m_s_src_mask_v(v0q)) ||
                    //     (!m_s_src_connect_mask_v(v1q) &&
                    //      !m_s_src_mask_v(v1q)) &&
                    //         (!m_s_src_connect_mask_v(v2q) &&
                    //          !m_s_src_mask_v(v2q))) {
                    //
                    //     printf("\n **v0q= %u, v1q= %u, v2q= %u", v0q, v1q,
                    //     v2q);
                    //
                    //     /*if (q_patch_info.is_owned(LocalVertexT(v0q))) {
                    //         printf("\n **v0q = %f, %f, %f",
                    //                coords(VertexHandle(q, v0q), 0),
                    //                coords(VertexHandle(q, v0q), 1),
                    //                coords(VertexHandle(q, v0q), 2));
                    //     }
                    //     if (q_patch_info.is_owned(LocalVertexT(v1q))) {
                    //         printf("\n **v1q = %f, %f, %f",
                    //                coords(VertexHandle(q, v1q), 0),
                    //                coords(VertexHandle(q, v1q), 1),
                    //                coords(VertexHandle(q, v1q), 2));
                    //     }
                    //     if (q_patch_info.is_owned(LocalVertexT(v2q))) {
                    //         printf("\n **v2q = %f, %f, %f",
                    //                coords(VertexHandle(q, v2q), 0),
                    //                coords(VertexHandle(q, v2q), 1),
                    //                coords(VertexHandle(q, v2q), 2));
                    //     }*/
                    // }


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
        for (int e = threadIdx.x; e < int(q_num_edges_up); e += blockThreads) {
            if (m_s_should_slice[0]) {
                return false;
            }
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
                if (!inserted) {
                    m_s_should_slice[0] = true;
                }
            }
            block.sync();
        }
        if (m_s_should_slice[0]) {
            return false;
        }

        if (!lock_patches_to_lock(block)) {
            return false;
        }

        populate_correspondence<FaceHandle>(block,
                                            q_patch_info,
                                            q_stash_id,
                                            m_s_q_correspondence_vf,
                                            m_s_q_correspondence_stash_vf,
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
        for (int f = threadIdx.x; f < int(q_num_faces_up); f += blockThreads) {
            if (m_s_should_slice[0]) {
                return false;
            }
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
                if (!inserted) {
                    m_s_should_slice[0] = true;
                }
            }
            block.sync();
        }
        if (m_s_should_slice[0]) {
            return false;
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
                assert(owner_stash_id < PatchStash::stash_size);
                ret = LPPair(vp, vq, owner_stash_id);

                assert(q_vertex < m_correspondence_size_vf);
                m_s_q_correspondence_vf[q_vertex]       = vp;
                m_s_q_correspondence_stash_vf[q_vertex] = owner_stash_id;

                assert(owner_stash_id < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(owner_stash_id, true);
            } else if (o != q && o != m_patch_info.patch_id &&
                       o_stash != INVALID8) {
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

        auto [v0q, v1q] = q_patch_info.get_edge_vertices(q_edge);

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

                // if (v0p == INVALID16 || v1p == INVALID16) {
                //     printf(
                //         "\n patch_id = %u, q= %u, v0q=%u, v1q= "
                //         "%u,v0p=%u, v1p= %u, q_edge= %u, o= %u, "
                //         "m_s_src_connect_mask_v(v0q)= "
                //         "%d,m_s_src_connect_mask_v(v1q)= %d, "
                //         "m_s_src_mask_v(v0q)= %u, m_s_src_mask_v(v1q)= %u, "
                //         "m_s_src_connect_mask_e=%d, "
                //         "m_s_src_mask_e=%d, "
                //         "q_patch_info.is_owned(v0q)= "
                //         "%d,q_patch_info.is_owned(v1q)= %d,",
                //         patch_id(),
                //         q,
                //         v0q,
                //         v1q,
                //         v0p,
                //         v1p,
                //         q_edge,
                //         o,
                //         m_s_src_connect_mask_v(v0q),
                //         m_s_src_connect_mask_v(v1q),
                //         m_s_src_mask_v(v0q),
                //         m_s_src_mask_v(v1q),
                //         m_s_src_connect_mask_e(q_edge),
                //         m_s_src_mask_e(q_edge),
                //         q_patch_info.is_owned(LocalVertexT(v0q)),
                //         q_patch_info.is_owned(LocalVertexT(v1q)));
                //
                //     /*if (q_patch_info.is_owned(LocalVertexT(v0q))) {
                //         printf("\n v0q = %f, %f, %f",
                //                coords(VertexHandle(q, v0q), 0),
                //                coords(VertexHandle(q, v0q), 1),
                //                coords(VertexHandle(q, v0q), 2));
                //     }
                //     if (q_patch_info.is_owned(LocalVertexT(v1q))) {
                //         printf("\n v1q = %f, %f, %f",
                //                coords(VertexHandle(q, v1q), 0),
                //                coords(VertexHandle(q, v1q), 1),
                //                coords(VertexHandle(q, v1q), 2));
                //     }*/
                // }

                // active bitmask is set in add_element

                // since it is owned by some other patch
                m_s_owned_mask_e.reset(ep, true);

                const uint8_t owner_stash_id =
                    m_s_patch_stash.insert_patch(o, m_s_patch_stash_mutex);

                assert(owner_stash_id < PatchStash::stash_size);
                assert(q_edge < m_correspondence_size_e);
                m_s_q_correspondence_e[q_edge]       = ep;
                m_s_q_correspondence_stash_e[q_edge] = owner_stash_id;

                assert(owner_stash_id != INVALID8);
                ret = LPPair(ep, eq, owner_stash_id);

                assert(owner_stash_id < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(owner_stash_id, true);
            } else if (o != q && o != m_patch_info.patch_id &&
                       o_stash != INVALID8) {
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
                assert(owner_stash_id < PatchStash::stash_size);

                assert(q_face < m_correspondence_size_vf);
                m_s_q_correspondence_vf[q_face]       = fp;
                m_s_q_correspondence_stash_vf[q_face] = owner_stash_id;

                ret = LPPair(fp, fq, owner_stash_id);

                assert(owner_stash_id < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(owner_stash_id, true);
            } else if (o != q && o != m_patch_info.patch_id &&
                       o_stash != INVALID8) {
                assert(o_stash != INVALID8);
                assert(o_stash < m_s_patches_to_lock_mask.size());
                m_s_patches_to_lock_mask.set(o_stash, true);
            }
        }
    }

    return ret;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ uint8_t
CavityManager<blockThreads, cop>::add_new_patch_to_patch_stash(
    const uint32_t new_patch)
{
    // first check if the patch is in the stash already,
    //  then we use mutex, do the check again, and add the patch if it's not
    //  there. if the patch is new, then we indicate this in
    //  m_s_new_patch_added

    uint8_t ret =
        m_s_patch_stash.insert_patch(new_patch, m_s_patch_stash_mutex);

    if (m_s_new_patch_stash.get_patch(ret) == INVALID32) {
        m_s_new_patch_added[0] = true;
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
                                   m_s_q_correspondence_stash_vf,
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
                                 m_s_q_correspondence_stash_e,
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
                                 m_s_q_correspondence_stash_vf,
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
    uint8_t*       q_correspondence_stash,
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
        src_patch_stash_id = q_correspondence_stash[lid];
        // assert(src_patch_stash_id < PatchStash::stash_size);
        src_patch = (src_patch_stash_id == INVALID8) ?
                        m_patch_info.patch_id :
                        m_s_patch_stash.get_patch(src_patch_stash_id);
        return corres;
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
            q_correspondence[lid_in]       = owner.local_id();
            q_correspondence_stash[lid_in] = INVALID8;
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
    // for (int i = 0; i < int(dest_patch_num_elements); ++i) {
    //    assert(i < dest_patch_owned_mask.size());
    //    assert(i < dest_patch_active_mask.size());
    //    assert(i < dest_in_cavity.size());
    //    if (!dest_patch_owned_mask(i) &&
    //        (dest_patch_active_mask(i) || dest_in_cavity(i))) {
    //
    //        const LPPair lp =
    //            m_patch_info.get_lp<HandleT>().find(i, s_table, s_stash);
    //
    //        if (m_s_patch_stash.get_patch(lp) == src_patch &&
    //            lp.local_id_in_owner_patch() == lid) {
    //            q_correspondence[lid_in]       = i;
    //            q_correspondence_stash[lid_in] = lp.patch_stash_id();
    //            assert(q_correspondence_stash[lid_in] <
    //            PatchStash::stash_size); src_patch_stash_id =
    //            lp.patch_stash_id(); src_patch =
    //            m_s_patch_stash.get_patch(src_patch_stash_id); return i;
    //        }
    //    }
    //}
    return INVALID16;
}

template <uint32_t blockThreads, CavityOp cop>
template <typename HandleT>
__device__ __inline__ void
CavityManager<blockThreads, cop>::populate_correspondence(
    cooperative_groups::thread_block& block,
    const PatchInfo&                  q_patch_info,
    const uint8_t                     q_stash,
    uint16_t*                         s_correspondence,
    uint8_t*                          s_correspondence_stash,
    const uint16_t                    s_correspondence_size,
    const LPPair*                     s_table,
    const LPPair*                     s_stash)
{
    using LocalT = typename HandleT::LocalT;

    fill_n<blockThreads>(
        s_correspondence, s_correspondence_size, uint16_t(INVALID16));
    fill_n<blockThreads>(
        s_correspondence_stash, s_correspondence_size, uint8_t(INVALID8));
    block.sync();

    LPHashTable lp = m_patch_info.get_lp<HandleT>();

    for (int b = threadIdx.x; b < int(lp.m_capacity); b += blockThreads) {
        const LPPair pair = s_table[b];
        if (!pair.is_sentinel() && pair.patch_stash_id() == q_stash) {
            assert(pair.local_id_in_owner_patch() < s_correspondence_size);
            s_correspondence[pair.local_id_in_owner_patch()] = pair.local_id();
            s_correspondence_stash[pair.local_id_in_owner_patch()] =
                pair.patch_stash_id();
            assert(s_correspondence_stash[pair.local_id_in_owner_patch()] <
                   PatchStash::stash_size);
        }
    }


    for (int b = threadIdx.x; b < LPHashTable::stash_size; b += blockThreads) {
        auto pair = s_stash[b];
        if (!pair.is_sentinel() && pair.patch_stash_id() == q_stash) {
            assert(pair.local_id_in_owner_patch() < s_correspondence_size);
            s_correspondence[pair.local_id_in_owner_patch()] = pair.local_id();
            s_correspondence_stash[pair.local_id_in_owner_patch()] =
                pair.patch_stash_id();
            assert(s_correspondence_stash[pair.local_id_in_owner_patch()] <
                   PatchStash::stash_size);
        }
    }


    // build patch stash mapper
    build_patch_stash_mapper(block, q_patch_info);
    block.sync();


    // for other not-owned elements in q, we store the local id in the owner
    // patch and patch stash id (in q patch stash). When we store the local id
    // in the owner patch, we set the high bit to one to mark these elements
    // differently for the next for loop, where we check if these elements
    // exists in p

    // The q's not-owned element could be
    // 1) don't exists at all in p.
    // 2) elements where the owner patch is p.
    // 3) elements where the owner patch is k and these
    // elements have a copy in p. This will be handle in the next for loop
    //
    // To handle 1) and 3), we have to convert q's patch stash id to p's patch
    // stash id and this is what we do in this for loop as well
    int q_num_elements = q_patch_info.get_num_elements<HandleT>()[0];
    assert(q_num_elements <= s_correspondence_size);

    for (int b = threadIdx.x; b < q_num_elements; b += blockThreads) {
        const LocalT l = LocalT(b);

        if (!q_patch_info.is_owned(l) && !q_patch_info.is_deleted(l)) {
            const LPPair lp =
                q_patch_info.get_lp<HandleT>().find(b, nullptr, nullptr);

            assert(s_correspondence[b] == INVALID16);
            assert(lp.patch_stash_id() < PatchStash::stash_size);
            assert(lp.patch_stash_id() != INVALID8);


            // set the high bit to one
            uint16_t s = lp.local_id_in_owner_patch();


            // it might sound wise to assert that the k_patch (which is the
            // patch  point to by lp) is the owning patch of s. But, we are not
            // even sure if we are going to use s at all. Also, later on, we
            // check if k_pach is the owner (ensure_ownership) and if it is
            // dirty in migrate() However, in there, we do this check on the
            // patches that we are sure that we are going to change their
            // ownership.
            //
            // const uint32_t k_patch =
            //     q_patch_info.patch_stash.get_patch(lp.patch_stash_id());
            // assert(m_context.m_patches_info[k_patch].is_owned(LocalT(s)));


            // patch stash id in q's patch stash
            const uint8_t p_stash_id =
                m_s_patch_stash_mapper[lp.patch_stash_id()];

            // Case 2
            if (p_stash_id == INVALID8 - 1u) {
                s_correspondence_stash[b] = INVALID8;
            } else {

                // Case 1, early exit, i.e., if the owner patch is not in p's
                // patch stash, then the element is for sure not in p at all

                // Could be either INVALID8 or the actual patch stash ID in p's
                // patch stash
                s_correspondence_stash[b] = p_stash_id;

                if (p_stash_id == INVALID8) {
                    // if this patch is not in p, then reset this correspondence
                    // entry
                    s = INVALID16;
                } else {
                    // otherwise, mark this element by setting its highest bit
                    s |= (1 << 15);
                }
            }

            s_correspondence[b] = s;
        }
    }
    block.sync();


    // Now, in s_correspondence, there are some elements that are stored where
    // we have the local index in the owner patch and the patch stash in p.
    // some of these elements have a corresponding copy in p and we want to
    // replace their entries with the local index in p. Of course these are
    // not-owned elements in p. So, there are different way of doing this.
    // 1) outer for loop on the correspondence, and inner loop on p's table and
    // p's table stash
    // 2) outer loop on p's not-owned elements and find the LPPair, then inner
    // for loop in the correspondence
    // We go with 2) since we have "probably" have smaller number of
    // correspondence to search through than number of not-owned elements


    const int num_elements = get_num_elements<HandleT>();

    // outer
    for (int b = threadIdx.x; b < num_elements; b += blockThreads) {
        if (!is_owned<HandleT>(b) &&
            (is_active<HandleT>(b) || is_in_cavity<HandleT>(b))) {

            const LPPair lp =
                m_patch_info.get_lp<HandleT>().find(b, s_table, s_stash);

            assert(lp.local_id() == b);

            // inner
            for (int c = 0; c < q_num_elements; ++c) {
                uint16_t corres = s_correspondence[c];
                if (corres != INVALID16 && (corres & (1 << 15))) {
                    corres &= ~(1 << 15);
                    if (corres == lp.local_id_in_owner_patch() &&
                        s_correspondence_stash[c] == lp.patch_stash_id()) {
                        s_correspondence[c] = lp.local_id();
                        break;
                    }
                }
            }
        }
    }

    block.sync();

    // finally reset the correspondence element where their high bit is set to
    // one since these elements are not-owned in q but they don't appear in p
    for (int b = threadIdx.x; b < q_num_elements; b += blockThreads) {
        // if the highest bit is set
        const uint16_t corres = s_correspondence[b];
        if (corres != INVALID16 && (corres & (1 << 15))) {
            s_correspondence[b]       = INVALID16;
            s_correspondence_stash[b] = INVALID8;
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::build_patch_stash_mapper(
    cooperative_groups::thread_block& block,
    const PatchInfo&                  q_patch_info)
{
    // build patch stash mapper that maps q's patch stash index to p's patch
    // stash index
    for (int q_stash_id = threadIdx.x; q_stash_id < PatchStash::stash_size;
         q_stash_id += blockThreads) {

        const uint32_t k_patch = q_patch_info.patch_stash.get_patch(q_stash_id);

        if (k_patch == patch_id()) {
            m_s_patch_stash_mapper[q_stash_id] = INVALID8 - 1u;
        } else if (k_patch != INVALID32) {
            const uint8_t p_stash_id =
                m_s_patch_stash.find_patch_index(k_patch);
            m_s_patch_stash_mapper[q_stash_id] = p_stash_id;
        } else {
            m_s_patch_stash_mapper[q_stash_id] = INVALID8;
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

    for (int vp = threadIdx.x; vp < int(num_elements); vp += blockThreads) {
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
__device__ __inline__ bool
CavityManager<blockThreads, cop>::ensure_locked_patches_are_not_dirty()
{
    for (int st = 0; st < PatchStash::stash_size; ++st) {
        assert(st < m_s_locked_patches_mask.size());
        if (m_s_locked_patches_mask(st)) {
            const uint32_t q = m_s_patch_stash.get_patch(st);
            if (m_context.m_patches_info[q].is_dirty()) {
                return false;
            }
        }
    }
    return true;
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
    for (int vp = threadIdx.x; vp < int(num_elements); vp += blockThreads) {
        assert(vp < s_ownership_change.size());
        if (s_ownership_change(vp)) {
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

            for (int attr = 0; attr < int(num_attr); ++attr) {
                attribute(p, vp, attr) = attribute(h, attr);
            }
        };

    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        for (int vp = threadIdx.x; vp < int(m_s_num_vertices[0]);
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
        for (int ep = threadIdx.x; ep < int(m_s_num_edges[0]);
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
        for (int fp = threadIdx.x; fp < int(m_s_num_faces[0]);
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
__device__ __inline__ void CavityManager<blockThreads, cop>::recover_faces()
{
    for (int f = threadIdx.x; f < int(m_s_num_faces[0]); f += blockThreads) {
        if (m_s_recover_f(f)) {
            if (!m_patch_info.is_deleted(LocalFaceT(f))) {
                m_s_active_mask_f.set(f, true);
#ifndef NDEBUG
                for (int i = 0; i < 3; ++i) {
                    const uint16_t e =
                        static_cast<uint16_t>(m_s_fe[3 * f + i] >> 1);
                    assert(m_s_recover_e(e) || m_s_active_mask_e(e));
                }
#endif
            }
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::recover_edges()
{
    for (int e = threadIdx.x; e < int(m_s_num_edges[0]); e += blockThreads) {
        if (m_s_recover_e(e)) {
            if (!m_patch_info.is_deleted(LocalEdgeT(e))) {
                m_s_active_mask_e.set(e, true);
#ifndef NDEBUG
                for (int i = 0; i < 2; ++i) {
                    const uint16_t v = m_s_ev[2 * e + i];
                    assert(m_s_recover_v(v) || m_s_active_mask_v(v));
                }
#endif
            }
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void CavityManager<blockThreads, cop>::recover_vertices()
{
    for (int v = threadIdx.x; v < int(m_s_num_vertices[0]); v += blockThreads) {
        if (m_s_recover_v(v)) {
            if (!m_patch_info.is_deleted(LocalVertexT(v))) {
                m_s_active_mask_v.set(v, true);
            }
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::recover_vertices_through_edges()
{
    // scatter
    for (int e = threadIdx.x; e < int(m_s_num_edges[0]); e += blockThreads) {
        if (m_s_recover_e(e)) {
            for (int i = 0; i < 2; ++i) {
                const uint16_t v = m_s_ev[2 * e + i];
                assert(v < m_s_num_vertices[0]);
                assert(!m_patch_info.is_deleted(LocalVertexT(v)));
                m_s_recover_v.set(v, true);
                m_s_active_mask_v.set(v, true);
            }
        }
    }
}

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::recover_edges_through_faces()
{
    // scatter
    for (int f = threadIdx.x; f < int(m_s_num_faces[0]); f += blockThreads) {
        if (m_s_recover_f(f)) {
            for (int i = 0; i < 3; ++i) {
                const uint16_t e =
                    static_cast<uint16_t>(m_s_fe[3 * f + i] >> 1);
                assert(e < m_s_num_edges[0]);
                assert(!m_patch_info.is_deleted(LocalEdgeT(e)));
                m_s_recover_e.set(e, true);
                m_s_active_mask_e.set(e, true);
            }
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::recover_edges_through_vertices()
{
    // gather
    for (int e = threadIdx.x; e < int(m_s_num_edges[0]); e += blockThreads) {
        if (!m_s_active_mask_e(e) && !m_patch_info.is_deleted(LocalEdgeT(e))) {
            bool recover = false;

            for (int i = 0; i < 2; ++i) {
                const uint16_t v = m_s_ev[2 * e + i];
                if (v < m_s_num_vertices[0]) {
                    if (m_s_recover_v(v)) {
                        recover = true;
                        break;
                    }
                }
            }
            if (recover) {
                m_s_recover_e.set(e, true);
                m_s_active_mask_e.set(e, true);
            }
        }
    }
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ void
CavityManager<blockThreads, cop>::recover_faces_through_edges()
{
    // gather
    for (int f = threadIdx.x; f < int(m_s_num_faces[0]); f += blockThreads) {

        if (!m_s_active_mask_f(f) && !m_patch_info.is_deleted(LocalFaceT(f))) {

            bool recover = false;

            for (int i = 0; i < 3; ++i) {
                const uint16_t e =
                    static_cast<uint16_t>(m_s_fe[3 * f + i] >> 1);
                if (e < m_s_num_edges[0]) {
                    if (m_s_recover_e(e)) {
                        recover = true;
                        break;
                    }
                }
            }

            if (recover) {
                m_s_recover_f.set(f, true);
                m_s_active_mask_f.set(f, true);
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
        if (m_preserve_cavity) {
            for (int v = threadIdx.x; v < int(m_s_active_mask_v.size());
                 v += blockThreads) {
                if (m_s_in_cavity_v(v)) {
                    assert(!m_s_active_mask_v(v));
                }
                if (m_s_fill_in_v(v)) {
                    assert(m_s_active_mask_v(v));
                }
            }
            for (int e = threadIdx.x; e < int(m_s_active_mask_e.size());
                 e += blockThreads) {
                if (m_s_in_cavity_e(e)) {
                    assert(!m_s_active_mask_e(e));
                }
                if (m_s_fill_in_e(e)) {
                    assert(m_s_active_mask_e(e));
                }
            }
            for (int f = threadIdx.x; f < int(m_s_active_mask_f.size());
                 f += blockThreads) {
                if (m_s_in_cavity_f(f)) {
                    assert(!m_s_active_mask_f(f));
                }
                if (m_s_fill_in_f(f)) {
                    assert(m_s_active_mask_f(f));
                }
            }
        } else if (m_s_remove_fill_in[0]) {
            // if we are not preserving cavity and we have to back off, then
            // we have to check if the mesh element is either in-cavity or
            // fill-in. If there is an element that is both (in-cavity and
            // fill-in), that means we have lost its topology/geometry info.
            // A potential solution is to selectively update global memory such
            // that we don't update global memory with these elements
            for (int v = threadIdx.x; v < int(m_s_active_mask_v.size());
                 v += blockThreads) {
                if (m_s_in_cavity_v(v)) {
                    assert(!m_s_fill_in_v(v));
                }
                if (m_s_fill_in_v(v)) {
                    assert(!m_s_in_cavity_v(v));
                }
            }
            for (int e = threadIdx.x; e < int(m_s_active_mask_e.size());
                 e += blockThreads) {
                if (m_s_in_cavity_e(e)) {
                    assert(!m_s_fill_in_e(e));
                }
                if (m_s_fill_in_e(e)) {
                    assert(!m_s_in_cavity_e(e));
                }
            }
            for (int f = threadIdx.x; f < int(m_s_active_mask_f.size());
                 f += blockThreads) {
                if (m_s_in_cavity_f(f)) {
                    assert(!m_s_fill_in_f(f));
                }
                if (m_s_fill_in_f(f)) {
                    assert(!m_s_in_cavity_f(f));
                }
            }
        }
#endif
        if (m_s_recover[0]) {
            if constexpr (cop == CavityOp::V) {
                recover_vertices();
                recover_edges_through_vertices();
                block.sync();
                recover_faces_through_edges();
            }

            if constexpr (cop == CavityOp::EV) {
                recover_vertices_through_edges();
                block.sync();
                recover_edges_through_vertices();
                block.sync();
                recover_faces_through_edges();
            }

            if constexpr (cop == CavityOp::E) {
                recover_edges();
                recover_faces_through_edges();
            }
            block.sync();
        }

        // store bitmask
        if (m_s_remove_fill_in[0]) {
            // TODO optimize this by working on whole 32-bit mask
            //
            //  removing fill-in elements since we were not successful in adding
            //  all of them. Thus, we need to preserve the original mesh by
            //  removing these elements and re-activating the in-cavity ones
            for (int v = threadIdx.x; v < int(m_s_active_mask_v.size());
                 v += blockThreads) {
                if (m_s_in_cavity_v(v)) {
                    m_s_active_mask_v.set(v, true);
                }
                if (m_s_fill_in_v(v)) {
                    m_s_active_mask_v.reset(v, true);
                }
            }

            for (int e = threadIdx.x; e < int(m_s_active_mask_e.size());
                 e += blockThreads) {
                if (m_s_in_cavity_e(e)) {
                    m_s_active_mask_e.set(e, true);
                }
                if (m_s_fill_in_e(e)) {
                    m_s_active_mask_e.reset(e, true);
                }
            }


            for (int f = threadIdx.x; f < int(m_s_active_mask_f.size());
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
    }

    if (m_s_should_slice[0]) {
        if (threadIdx.x == 0) {
            m_context.m_patches_info[patch_id()].should_slice = true;
        }
    }

    // re-add the patch to the queue if there is ownership change
    // or we could not lock all neighbor patches (and thus could not write to
    // global memory)
    if ((m_s_should_slice[0] || !m_write_to_gmem) && get_num_cavities() > 0) {
        push();
    }

    // unlock any neighbor patch we have locked
    unlock_locked_patches();

    // unlock this patch
    unlock();
}
}  // namespace rxmesh
