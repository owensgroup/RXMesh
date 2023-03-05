
namespace rxmesh {

template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool Cavity<blockThreads, cop>::migrate_v2(
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
    m_s_ownership_change_mask_v.reset(block);
    m_s_ownership_change_mask_e.reset(block);
    m_s_ownership_change_mask_f.reset(block);
    m_s_patches_to_lock_mask.reset(block);
    m_s_locked_patches_mask.reset(block);
    block.sync();


    // lock the current patch since we are going to update its hashtables
    __shared__ bool s_success;
    if (threadIdx.x == 0) {
        s_success = m_patch_info.lock.acquire_lock(blockIdx.x);
    }
    block.sync();
    if (!s_success) {
        return false;
    }

    auto unlock_this_patch = [&]() {
        // TODO remove from the LPHashTable in global memory any elements that
        // has been added during the migration
        assert(false);
        if (threadIdx.x == 0) {
            m_patch_info.lock.release_lock();
        }
    };

    // make sure the timestamp is the same after locking the patch
    if (!is_same_timestamp(block)) {
        unlock_this_patch();
        return false;
    }


    // mark vertices on the boundary of all active cavities in this patch
    // owned vertices are marked in m_s_owned_cavity_bdry_v and not-owned
    // vertices are marked in m_s_migrate_mask_v (since we need to migrate
    // them)
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


    // mark a face in the ownership change (m_s_ownership_change_mask_f) if
    // one of its edges is connected to a vertex that is marked in
    // m_s_owned_cavity_bdry_v. Then mark that face's three edges in the
    // ownership change (m_s_ownership_change_mask_e)
    for (uint16_t f = threadIdx.x; f < m_s_num_faces[0]; f += blockThreads) {
        if (!m_s_owned_mask_f(f) &&
            (m_s_active_mask_f(f) || m_s_cavity_id_f[f] != INVALID16)) {
            bool change = false;
            for (int i = 0; i < 3; ++i) {
                const uint16_t e = m_s_fe[3 * f + i] >> 1;

                assert(m_s_active_mask_e(e) || m_s_cavity_id_e[e] != INVALID16);

                const uint16_t v0 = m_s_ev[2 * e + 0];
                const uint16_t v1 = m_s_ev[2 * e + 1];

                assert(m_s_active_mask_v(v0) ||
                       m_s_cavity_id_v[v0] != INVALID16);
                assert(m_s_active_mask_v(v1) ||
                       m_s_cavity_id_v[v1] != INVALID16);

                if (m_s_owned_cavity_bdry_v(v0) ||
                    m_s_owned_cavity_bdry_v(v1) || m_s_migrate_mask_v(v0) ||
                    m_s_migrate_mask_v(v1)) {
                    change = true;

                    m_s_ownership_change_mask_f.set(f, true);
                }
                break;
            }

            if (change) {
                for (int i = 0; i < 3; ++i) {
                    const uint16_t e = m_s_fe[3 * f + i] >> 1;
                    if (!m_s_owned_mask_e(e)) {
                        assert(m_s_active_mask_e(e) ||
                               m_s_cavity_id_e[e] != INVALID16);
                        m_s_ownership_change_mask_e.set(e, true);
                    }
                }
            }
        }
    }

    block.sync();

    // construct protection zone
    for (uint32_t p = 0; p < PatchStash::stash_size; ++p) {
        const uint32_t q = m_patch_info.patch_stash.get_patch(p);
        if (q != INVALID32) {
            if (!migrate_from_patch_v2(block, q, m_s_migrate_mask_v, true)) {
                unlock_this_patch();
                return false;
            }
        }
    }
    block.sync();


    // ribbonize protection zone
    for (uint16_t e = threadIdx.x; e < m_s_num_edges[0]; e += blockThreads) {
        if (m_s_active_mask_e(e) || m_s_cavity_id_e[e] != INVALID16) {

            // we only want to ribbonize vertices connected to a vertex on
            // the boundary of a cavity boundaries. If the two vertices are
            // on the cavity boundaries (b0=true and b1=true), then this is
            // an edge on the cavity and we don't to ribbonize any of these
            // two vertices Only when one of the vertices are on the cavity
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
                m_s_ribbonize_v.set(v1, true);
            }

            if (b1 && !b0 && !m_s_owned_mask_v(v0)) {
                m_s_ribbonize_v.set(v0, true);
            }
        }
    }

    block.sync();

    for (uint32_t p = 0; p < PatchStash::stash_size; ++p) {
        const uint32_t q = m_patch_info.patch_stash.get_patch(p);
        if (q != INVALID32) {
            if (!migrate_from_patch_v2(block, q, m_s_ribbonize_v, false)) {
                unlock_this_patch();
                return false;
            }
        }
    }

    return true;
}


template <uint32_t blockThreads, CavityOp cop>
__device__ __inline__ bool Cavity<blockThreads, cop>::migrate_from_patch_v2(
    cooperative_groups::thread_block& block,
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
            const VertexHandle v_owner = m_context.get_owner_vertex_handle(
                {m_patch_info.patch_id, {v}}, nullptr, false);
            if (v_owner.patch_id() == q) {
                ::atomicAdd(&s_ok_q, 1);
                m_s_src_mask_v.set(v_owner.local_id(), true);
            }
        }
    }
    block.sync();

    __shared__ bool s_success;

    auto lock_patch = [&](const uint8_t stash_id, const uint32_t patch) {
        if (threadIdx.x == 0) {
            bool okay = m_s_locked_patches_mask(stash_id);
            if (!okay) {
                okay = m_context.m_patches_info[patch].lock.acquire_lock(
                    blockIdx.x);
                if (okay) {
                    m_s_locked_patches_mask.set(stash_id);
                }
            }

            s_success = okay;
        }
        block.sync();
        return s_success;
    };

    auto lock_patch_p = [&](const uint32_t patch) {
        uint8_t stash_id = m_patch_info.patch_stash.find_patch_index(patch);
        assert(stash_id != INVALID8);
        return lock_patch(stash_id, patch);
    };

    auto lock_patch_st = [&](const uint8_t stash_id) {
        assert(stash_id != INVALID8);
        const uint32_t patch = m_patch_info.patch_stash.get_patch(stash_id);
        return lock_patch(stash_id, patch);
    };

    auto lock_patches_to_lock = [&]() {
        for (uint8_t i = 0; i < PatchStash::stash_size; ++i) {
            if (m_s_patches_to_lock_mask(i) && !m_s_locked_patches_mask(i)) {
                if (!lock_patch_st(i)) {
                    return false;
                }
            }
        }
        return true;
    };

    // unlock patches that has been locked
    auto unlock_locked_patches = [&]() {
        if (threadIdx.x == 0) {
            for (uint8_t i = 0; i < PatchStash::stash_size; ++i) {
                if (m_s_locked_patches_mask(i)) {
                    uint32_t p = m_patch_info.patch_stash.get_patch(i);
                    m_context.m_patches_info[p].lock.release_lock();
                }
            }
        }
    };


    // In every call to migrate_vertex/edge/face, threads make sure that
    // they mark patches they read from in m_s_patches_to_lock_mask.
    // At the end of every round, one thread make sure make sure that all
    // patches marked in m_s_patches_to_lock_mask are actually locked.
    if (s_ok_q != 0) {

        if (!lock_patch_p(q)) {
            unlock_locked_patches();
            return false;
        }

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
            const uint16_t v0q = q_patch_info.ev[2 * e + 0].id;
            const uint16_t v1q = q_patch_info.ev[2 * e + 1].id;

            if (m_s_src_mask_v(v0q)) {
                m_s_src_connect_mask_v.set(v1q, true);
            }

            if (m_s_src_mask_v(v1q)) {
                m_s_src_connect_mask_v.set(v0q, true);
            }
        }
        block.sync();

        // 3.
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


            LPPair lp =
                migrate_vertex(q,
                               q_num_vertices,
                               v,
                               false,  // change_ownership,
                               q_patch_info,
                               [&](const uint16_t vertex) {
                                   return m_s_src_connect_mask_v(vertex);
                               });

            // we need to make sure that no other
            // thread is querying the hashtable while we
            // insert in it
            block.sync();

            if (!lp.is_sentinel()) {
                if (!m_patch_info.lp_v.insert(lp)) {
                    assert(false);
                }
            }
        }
        block.sync();

        if (!lock_patches_to_lock()) {
            unlock_locked_patches();
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

                        // set the bit for this edge in src_e mask so we
                        // can use it for migrating faces
                        m_s_src_mask_e.set(edge, true);
                        return true;
                    }
                    return false;
                });

            block.sync();

            if (!lp.is_sentinel()) {
                if (!m_patch_info.lp_e.insert(lp)) {
                    assert(false);
                }
            }
        }
        block.sync();

        if (!lock_patches_to_lock()) {
            unlock_locked_patches();
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
            const uint16_t e0 = q_patch_info.fe[3 * f + 0].id >> 1;
            const uint16_t e1 = q_patch_info.fe[3 * f + 1].id >> 1;
            const uint16_t e2 = q_patch_info.fe[3 * f + 2].id >> 1;

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
                if (!m_patch_info.lp_e.insert(lp)) {
                    assert(false);
                }
            }
        }

        block.sync();
        if (!lock_patches_to_lock()) {
            unlock_locked_patches();
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
                if (!m_patch_info.lp_f.insert(lp)) {
                    assert(false);
                }
            }
        }

        if (!lock_patches_to_lock()) {
            unlock_locked_patches();
            return false;
        }
    }

    return true;
}


}  // namespace rxmesh