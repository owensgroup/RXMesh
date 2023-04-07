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
    m_s_cavity_size_prefix =
        shrd_alloc.alloc<uint16_t>(m_s_num_cavities[0] + 1);
    for (uint16_t i = threadIdx.x; i < m_s_num_cavities[0] + 1;
         i += blockThreads) {
        m_s_cavity_size_prefix[i] = 0;
    }


    // load mesh FE and EV
    load_mesh_async(block, shrd_alloc);
    block.sync();

    // Expand cavities by marking incident elements
    /*if constexpr (cop == CavityOp::V) {
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
    block.sync();*/
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
}  // namespace rxmesh