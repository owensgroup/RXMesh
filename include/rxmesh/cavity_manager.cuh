#pragma once

#include <cooperative_groups.h>

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/patch_info.h"

#include "rxmesh/attribute.h"


namespace rxmesh {

template <uint32_t blockThreads, CavityOp cop>
struct CavityManager
{
    /**
     * @brief default constructor
     */
    __device__ __inline__ CavityManager()
        : m_s_num_cavities(nullptr),
          m_s_cavity_size_prefix(nullptr),
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


    /**
     * @brief constructor
     * @param block
     * @param context
     * @param shrd_alloc
     * @return
     */
    __device__ __inline__ CavityManager(cooperative_groups::thread_block& block,
                                        Context&        context,
                                        ShmemAllocator& shrd_alloc);

    /**
     * @brief create new cavity from a seed element. The seed element type
     * should match the CavityOp type
     * @param seed
     */
    template <typename HandleT>
    __device__ __inline__ void create(HandleT seed);

    /**
     * @brief processes all cavities created using create() by removing elements
     * in these cavities, update the patch layout for subsequent cavity fill-in.
     * In the event of failure (due to failure of locking neighbor patches),
     * this function returns false.
     * @param block
     * @param shrd_alloc
     * @return
     */
    __device__ __inline__ bool prologue(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc);


    /**
     * @brief cleanup and store updated patch to global memory
     * @param block
     * @return
     */
    __device__ __inline__ void epilogue(
        cooperative_groups::thread_block& block);

    /**
     * @brief return the patch id that this cavity manager operates on
     */
    __device__ __forceinline__ uint32_t patch_id() const
    {
        return m_patch_info.patch_id;
    }

   private:
    /**
     * @brief allocate memory and then load the mesh FE and EV into shared
     * memory
     */
    __device__ __inline__ void load_mesh_async(
        cooperative_groups::thread_block& block,
        ShmemAllocator&                   shrd_alloc);


    // num_cavities could be uint16_t but we use int since we need atomicAdd
    int* m_s_num_cavities;

    // the prefix sum of the cavities sizes. the size of the cavity is the
    // number of boundary edges in the cavity
    uint16_t* m_s_cavity_size_prefix;

    // the maximum number of vertices, edges, and faces that this patch can hold
    uint16_t m_vert_cap, m_edge_cap, m_face_cap;

    // some cavities are inactive since they overlap without cavities.
    // we use this bitmask to indicate if the cavity is active or not
    Bitmask m_s_active_cavity_bitmask;

    // element ownership bitmask
    Bitmask m_s_owned_mask_v, m_s_owned_mask_e, m_s_owned_mask_f;

    // active elements bitmask
    Bitmask m_s_active_mask_v, m_s_active_mask_e, m_s_active_mask_f;

    // indicate if a the vertex should be migrated
    Bitmask m_s_migrate_mask_v;

    Bitmask m_s_src_mask_v, m_s_src_mask_e;
    Bitmask m_s_src_connect_mask_v, m_s_src_connect_mask_e;

    // indicate if the mesh element should change ownership
    Bitmask m_s_ownership_change_mask_v, m_s_ownership_change_mask_e,
        m_s_ownership_change_mask_f;

    Bitmask m_s_owned_cavity_bdry_v;

    // indicate if the vertex should be ribbonized
    Bitmask m_s_ribbonize_v;

    // indicate which patch (in the patch stash) should be locked
    Bitmask m_s_patches_to_lock_mask;

    // indicate which patch (in the patch stash) is actually locked
    Bitmask m_s_locked_patches_mask;

    Bitmask m_s_added_to_lp_v, m_s_added_to_lp_e, m_s_added_to_lp_f;

    // indicate if the mesh element is in the interior of the cavity
    Bitmask m_s_in_cavity_v, m_s_in_cavity_e, m_s_in_cavity_f;

    bool* m_s_readd_to_queue;

    // mesh connectivity
    uint16_t *m_s_ev, *m_s_fe;

    // store the cavity ID each mesh element belong to. If the mesh element
    // does not belong to any cavity, then it stores INVALID32
    uint16_t *m_s_cavity_id_v, *m_s_cavity_id_e, *m_s_cavity_id_f;

    // TODO this should be the hashtable pointers
    uint16_t *m_s_owner_v, *m_s_owner_e, *m_s_owner_f;

    // store the number of elements. we use pointers since the number of mesh
    // elements could change
    uint16_t *m_s_num_vertices, *m_s_num_edges, *m_s_num_faces;

    uint16_t* m_s_cavity_edge_loop;

    PatchInfo m_patch_info;
    Context   m_context;
};

}  // namespace rxmesh

#include "rxmesh/cavity_manager_impl.cuh"