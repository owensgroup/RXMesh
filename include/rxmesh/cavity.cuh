#pragma once

#include <cooperative_groups.h>

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

template <uint32_t blockThreads, CavityOp cop>
struct Cavity
{
    __device__ __inline__ Cavity()
        : m_s_num_cavities(nullptr),
          m_cavity_id_v(nullptr),
          m_cavity_id_e(nullptr),
          m_cavity_id_f(nullptr),
          m_cavity_edge_loop(nullptr),
          m_s_ev(nullptr),
          m_s_fe(nullptr)
    {
    }

    __device__ __inline__ Cavity(cooperative_groups::thread_block& block,
                                 ShmemAllocator&                   shrd_alloc,
                                 const PatchInfo&                  patch_info)
    {
        m_s_num_cavities = shrd_alloc.alloc<int>(1);
        m_cavity_id_v = shrd_alloc.alloc<uint16_t>(patch_info.num_vertices[0]);
        m_cavity_id_e = shrd_alloc.alloc<uint16_t>(patch_info.num_edges[0]);
        m_cavity_id_f = shrd_alloc.alloc<uint16_t>(patch_info.num_faces[0]);
        m_cavity_edge_loop =
            shrd_alloc.alloc<uint16_t>(patch_info.num_edges[0]);

        if (threadIdx.x == 0) {
            m_s_num_cavities[0] = 0;
        }

        // TODO fix the bank conflict
        for (uint16_t v = threadIdx.x; v < patch_info.num_vertices[0];
             v += blockThreads) {
            m_cavity_id_v[v] = INVALID16;
        }

        for (uint16_t e = threadIdx.x; e < patch_info.num_edges[0];
             e += blockThreads) {
            m_cavity_id_e[e] = INVALID16;
        }

        for (uint16_t f = threadIdx.x; f < patch_info.num_faces[0];
             f += blockThreads) {
            m_cavity_id_f[f] = INVALID16;
        }

        block.sync();
    }

    /**
     * @brief create new cavity
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
        if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                      cop == CavityOp::EE || cop == CavityOp::EF) {
            // TODO EV may also mark its vertices immediately
            m_cavity_id_e[handle.unpack().second] = id;
            // m_cavity_id_e[handle.unpack().second - 1] = id;
            // m_cavity_id_e[handle.unpack().second + 1] = id;
        }

        if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                      cop == CavityOp::VE || cop == CavityOp::VF) {
            m_cavity_id_v[handle.unpack().second] = id;
        }

        if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                      cop == CavityOp::FE || cop == CavityOp::FF) {
            // TODO FE may also mark its edges immediately
            m_cavity_id_f[handle.unpack().second] = id;
        }
    }

    /**
     * @brief delete elements by applying the cop operation
     * TODO we probably need to clear any shared memory used for queries during
     * adding elements to cavity
     */
    __device__ __inline__ void process(cooperative_groups::thread_block& block,
                                       ShmemAllocator& shrd_alloc,
                                       PatchInfo&      patch_info)
    {
        // load mesh FE and EV
        load_mesh_async(block, shrd_alloc, patch_info);
        block.sync();

        // Expand cavities
        if constexpr (cop == CavityOp::V) {
            mark_edges_through_vertices(patch_info);
            block.sync();
            mark_faces_through_edges(patch_info);
            block.sync();
        }

        if constexpr (cop == CavityOp::E) {
            mark_faces_through_edges(patch_info);
            block.sync();
        }

        // construct cavity boundary loop
        construct_cavities_edge_loop(block, patch_info);

        // sort each cavity edge loop
        sort_cavities_edge_loop(patch_info);

        block.sync();
    }


    /**
     * @brief load mesh FE and EV into shared memory
     */
    __device__ __inline__ void load_mesh_async(
        cooperative_groups::thread_block& block,
        ShmemAllocator&                   shrd_alloc,
        PatchInfo&                        patch_info)
    {
        // Load mesh info
        m_s_cavity_size = shrd_alloc.alloc<int>(m_s_num_cavities[0] + 1);

        for (uint16_t i = threadIdx.x; i < m_s_num_cavities[0] + 1;
             i += blockThreads) {
            m_s_cavity_size[i] = 0;
        }

        m_s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges[0]);
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(patch_info.ev),
                           2 * patch_info.num_edges[0],
                           m_s_ev,
                           false);
        m_s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces[0]);
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(patch_info.fe),
                           3 * patch_info.num_faces[0],
                           m_s_fe,
                           true);
    }

    /**
     * @brief propagate the cavity tag from vertices to their incident edges
     */
    __device__ __inline__ void mark_edges_through_vertices(
        PatchInfo& patch_info)
    {
        for (uint16_t e = threadIdx.x; e < patch_info.num_edges[0];
             e += blockThreads) {
            if (!detail::is_deleted(e, patch_info.active_mask_e)) {

                // vertices tag
                const uint16_t v0 = m_s_ev[2 * e];
                const uint16_t v1 = m_s_ev[2 * e + 1];

                if (m_cavity_id_v[v0] != INVALID16) {
                    // TODO possible race condition
                    m_cavity_id_e[e] = v0;
                }

                if (m_cavity_id_v[v1] != INVALID16) {
                    // TODO possible race condition
                    m_cavity_id_e[e] = v1;
                }
            }
        }
    }

    /**
     * @brief propagate the cavity tag from edges to their incident faces
     */
    __device__ __inline__ void mark_faces_through_edges(PatchInfo& patch_info)
    {
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces[0];
             f += blockThreads) {
            if (!detail::is_deleted(f, patch_info.active_mask_f)) {

                // edges tag
                const uint16_t e0 = m_s_fe[3 * f] >> 1;
                const uint16_t e1 = m_s_fe[3 * f + 1] >> 1;
                const uint16_t e2 = m_s_fe[3 * f + 2] >> 1;

                const uint16_t c0 = m_cavity_id_e[e0];
                const uint16_t c1 = m_cavity_id_e[e1];
                const uint16_t c2 = m_cavity_id_e[e2];

                if (c0 != INVALID16) {
                    m_cavity_id_f[f] = c0;
                }

                if (c1 != INVALID16) {
                    m_cavity_id_f[f] = c1;
                }

                if (c2 != INVALID16) {
                    m_cavity_id_f[f] = c2;
                }
            }
        }
    }

    /**
     * @brief construct the cavities boundary loop
     */
    template <uint32_t itemPerThread = 5>
    __device__ __inline__ void construct_cavities_edge_loop(
        cooperative_groups::thread_block& block,
        PatchInfo&                        patch_info)
    {


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
            if (f < patch_info.num_faces[0]) {
                face_cavity = m_cavity_id_f[f];
            }

            // if the face is inside a cavity
            // we could check on if the face is deleted but we only mark faces
            // that are not deleted so no need to double check this
            if (face_cavity != INVALID16) {
                const uint16_t c0 = m_cavity_id_e[m_s_fe[3 * f + 0] >> 1];
                const uint16_t c1 = m_cavity_id_e[m_s_fe[3 * f + 1] >> 1];
                const uint16_t c2 = m_cavity_id_e[m_s_fe[3 * f + 2] >> 1];

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

                const uint16_t face_cavity = m_cavity_id_f[f];

                int num_added = 0;

                const uint16_t e0 = m_s_fe[3 * f + 0];
                const uint16_t e1 = m_s_fe[3 * f + 1];
                const uint16_t e2 = m_s_fe[3 * f + 2];

                const uint16_t c0 = m_cavity_id_e[e0 >> 1];
                const uint16_t c1 = m_cavity_id_e[e1 >> 1];
                const uint16_t c2 = m_cavity_id_e[e2 >> 1];


                auto check_and_add = [&](const uint16_t c, const uint16_t e) {
                    if (c == INVALID16) {
                        uint16_t offset = m_s_cavity_size[face_cavity] +
                                          local_offset[i] + num_added;
                        m_cavity_edge_loop[offset] = e;
                        num_added++;
                    }
                };

                check_and_add(c0, e0);
                check_and_add(c1, e1);
                check_and_add(c2, e2);
            }
        }
    }


    /**
     * @brief sort cavity edge loop
     */
    __device__ __inline__ void sort_cavities_edge_loop(PatchInfo& patch_info)
    {

        // TODO need to increase the parallelism in this part. It should be at
        // least one warp processing one cavity
        for (uint16_t c = threadIdx.x; c < m_s_num_cavities[0];
             c += blockThreads) {

            // Specify the starting edge of the cavity before sorting everyhing
            // TODO this may be tuned for different CavityOp's
            uint16_t cavity_edge_src_vertex;
            for (uint16_t e = 0; e < patch_info.num_edges[0]; ++e) {
                if (m_cavity_id_e[e] == c) {
                    cavity_edge_src_vertex = m_s_ev[2 * e];
                    break;
                }
            }

            const uint16_t start = m_s_cavity_size[c];
            const uint16_t end   = m_s_cavity_size[c + 1];


            for (uint16_t e = start; e < end; ++e) {
                uint32_t edge = m_cavity_edge_loop[e];

                if (m_s_ev[2 * (edge >> 1) + 0] == cavity_edge_src_vertex ||
                    m_s_ev[2 * (edge >> 1) + 1] == cavity_edge_src_vertex) {
                    uint16_t temp             = m_cavity_edge_loop[start];
                    m_cavity_edge_loop[start] = edge;
                    m_cavity_edge_loop[e]     = temp;
                    break;
                }
            }


            for (uint16_t e = start; e < end; ++e) {
                uint16_t edge;
                uint8_t  dir;
                Context::unpack_edge_dir(m_cavity_edge_loop[e], edge, dir);
                uint16_t end_vertex = m_s_ev[2 * edge + 1];
                if (dir) {
                    end_vertex = m_s_ev[2 * edge];
                }

                for (uint16_t i = e + 1; i < end; ++i) {
                    uint32_t ee = m_cavity_edge_loop[i] >> 1;
                    uint32_t v0 = m_s_ev[2 * ee + 0];
                    uint32_t v1 = m_s_ev[2 * ee + 1];

                    if (v0 == end_vertex || v1 == end_vertex) {
                        uint16_t temp             = m_cavity_edge_loop[e + 1];
                        m_cavity_edge_loop[e + 1] = m_cavity_edge_loop[i];
                        m_cavity_edge_loop[i]     = temp;
                        break;
                    }
                }
            }
        }
    }


    /**
     * @brief apply a lambda function on each cavity to fill it in with edges
     * and then faces
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

            FillInFunc(c, get_cavity_size(c));
        }

        block.sync();
    }

    /**
     * @brief return number of cavities in this patch
     */
    __device__ __inline__ int get_num_cavities() const
    {
        return m_s_num_cavities[0];
    }

    /**
     * @brief return the size of the c-th cavity. The size is the number of
     * edges surrounding the cavity
     */
    __device__ __inline__ uint16_t get_cavity_size(uint16_t c) const
    {
        return m_s_cavity_size[c + 1] - m_s_cavity_size[c];
    }

    /**
     * @brief get an edge handle to the i-th edges in the c-th cavity
     */
    __device__ __inline__ DEdgeHandle get_cavity_edge(PatchInfo& patch_info,
                                                      uint16_t   c,
                                                      uint16_t   i) const
    {
        assert(c < m_s_num_cavities[0]);
        assert(i < get_cavity_size(c));
        return DEdgeHandle(patch_info.patch_id,
                           m_cavity_edge_loop[m_s_cavity_size[c] + i]);
    }


    /**
     * @brief get a vertex handle to the i-th vertex in the c-th cavity
     */
    __device__ __inline__ VertexHandle get_cavity_vertex(PatchInfo& patch_info,
                                                         uint16_t   c,
                                                         uint16_t   i) const
    {
        assert(c < m_s_num_cavities[0]);
        assert(i < get_cavity_size(c));

        uint16_t edge;
        flag_t   dir;
        Context::unpack_edge_dir(
            m_cavity_edge_loop[m_s_cavity_size[c] + i], edge, dir);

        const uint16_t v0 = m_s_ev[2 * edge];
        const uint16_t v1 = m_s_ev[2 * edge + 1];

        return VertexHandle(patch_info.patch_id, ((dir == 0) ? v0 : v1));
    }

    /**
     * @brief should be called by a single thread
     */
    __device__ __inline__ VertexHandle add_vertex(PatchInfo&     patch_info,
                                                  const uint16_t cavity_id)
    {
        // First try to reuse a vertex in the cavity or a deleted vertex
        uint16_t v_id = add_element(cavity_id,
                                    m_cavity_id_v,
                                    patch_info.active_mask_v,
                                    patch_info.num_vertices[0]);

        if (v_id == INVALID16) {
            // if this fails, then add a new vertex to the mesh
            v_id = atomicAdd(patch_info.num_vertices, 1);
            assert(v_id < patch_info.vertices_capacity[0]);
        }

        detail::bitmask_set_bit(v_id, patch_info.active_mask_v);
        detail::bitmask_set_bit(v_id, patch_info.owned_mask_v);
        return {patch_info.patch_id, v_id};
    }


    /**
     * @brief should be called by a single thread
     */
    __device__ __inline__ DEdgeHandle add_edge(PatchInfo&         patch_info,
                                               const uint16_t     cavity_id,
                                               const VertexHandle src,
                                               const VertexHandle dest)
    {
        assert(src.unpack().first == patch_info.patch_id);
        assert(dest.unpack().first == patch_info.patch_id);

        // First try to reuse an edge in the cavity or a deleted edge
        uint16_t e_id = add_element(cavity_id,
                                    m_cavity_id_e,
                                    patch_info.active_mask_e,
                                    patch_info.num_edges[0]);
        if (e_id == INVALID16) {
            // if this fails, then add a new edge to the mesh
            e_id = atomicAdd(patch_info.num_edges, 1);
            assert(e_id < patch_info.edges_capacity[0]);
        }
        m_s_ev[2 * e_id + 0] = src.unpack().second;
        m_s_ev[2 * e_id + 1] = dest.unpack().second;
        detail::bitmask_set_bit(e_id, patch_info.active_mask_e);
        detail::bitmask_set_bit(e_id, patch_info.owned_mask_e);
        return {patch_info.patch_id, e_id, 0};
    }


    /**
     * @brief should be called by a single thread
     */
    __device__ __inline__ FaceHandle add_face(PatchInfo&        patch_info,
                                              const uint16_t    cavity_id,
                                              const DEdgeHandle e0,
                                              const DEdgeHandle e1,
                                              const DEdgeHandle e2)
    {
        assert(e0.unpack().first == patch_info.patch_id);
        assert(e1.unpack().first == patch_info.patch_id);
        assert(e2.unpack().first == patch_info.patch_id);

        // First try to reuse a face in the cavity or a deleted face
        uint16_t f_id = add_element(cavity_id,
                                    m_cavity_id_f,
                                    patch_info.active_mask_f,
                                    patch_info.num_faces[0]);

        if (f_id == INVALID16) {
            // if this fails, then add a new face to the mesh
            f_id = atomicAdd(patch_info.num_faces, 1);
            assert(f_id < patch_info.faces_capacity[0]);
        }

        m_s_fe[3 * f_id + 0] = e0.unpack().second;
        m_s_fe[3 * f_id + 1] = e1.unpack().second;
        m_s_fe[3 * f_id + 2] = e2.unpack().second;
        detail::bitmask_set_bit(f_id, patch_info.active_mask_f);
        detail::bitmask_set_bit(f_id, patch_info.owned_mask_f);
        return {patch_info.patch_id, f_id};
    }

    /**
     * @brief cleanup by moving data from shared memory to global memory
     */
    __device__ __inline__ void cleanup(cooperative_groups::thread_block& block,
                                       PatchInfo& patch_info)
    {
        detail::store<blockThreads>(m_s_ev,
                                    2 * patch_info.num_edges[0],
                                    reinterpret_cast<uint16_t*>(patch_info.ev));

        detail::store<blockThreads>(m_s_fe,
                                    3 * patch_info.num_faces[0],
                                    reinterpret_cast<uint16_t*>(patch_info.fe));
    }


    /**
     * @brief find the index of the next element to add. First search within the
     * cavity and find the first element that has it cavity set to cavity_id. If
     * nothing found, search for the first element that has its bitmask set to 0
     */
    __device__ __inline__ uint16_t add_element(const uint16_t cavity_id,
                                               uint16_t*      element_cavity_id,
                                               uint32_t*      active_bitmask,
                                               const uint16_t num_elements)
    {

        for (uint16_t i = 0; i < num_elements; ++i) {
            if (element_cavity_id[i] == cavity_id) {
                element_cavity_id[i] = INVALID16;
                return i;
            }
        }

        for (uint16_t i = 0; i < num_elements; ++i) {
            if (!detail::is_set_bit(i, active_bitmask)) {
                return i;
            }
        }

        return INVALID16;
    }


    int*      m_s_num_cavities;
    uint16_t *m_cavity_id_v, *m_cavity_id_e, *m_cavity_id_f;
    uint16_t* m_cavity_edge_loop;
    uint16_t *m_s_ev, *m_s_fe;
    int*      m_s_cavity_size;
};

}  // namespace rxmesh