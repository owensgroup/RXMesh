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


    /**
     * @brief apply a lambda function on each cavity to fill it in with edges
     * and then faces
     */
    template <typename FillInT>
    __device__ __inline__ void for_each_cavity(
        cooperative_groups::thread_block& block,
        FillInT                           FillInFunc);


    /**
     * @brief return number of active non-conflicting cavities in this patch
     */
    __device__ __inline__ int get_num_cavities() const
    {
        return m_s_num_cavities[0];
    }

    /**
     * @brief return the size of the c-th cavity. The size is the number of
     * edges on the cavity boundary
     */
    __device__ __inline__ uint16_t get_cavity_size(uint16_t c) const
    {
        return m_s_cavity_size_prefix[c + 1] - m_s_cavity_size_prefix[c];
    }

    /**
     * @brief get an edge handle to the i-th edge in the c-th cavity
     */
    __device__ __inline__ DEdgeHandle get_cavity_edge(uint16_t c,
                                                      uint16_t i) const;

    /**
     * @brief get a vertex handle to the i-th vertex in the c-th cavity
     */
    __device__ __inline__ VertexHandle get_cavity_vertex(uint16_t c,
                                                         uint16_t i) const;

    /**
     * @brief add a new vertex to the patch. The returned handle could be use
     * to access the vertex attributes
     */
    __device__ __inline__ VertexHandle add_vertex();


    /**
     * @brief add a new edge to the patch defined by the src and dest vertices.
     * The src and dest vertices could either come from get_cavity_vertex() or
     * add_vertex(). The returned handle could be used to access the edge
     * attributes
     */
    __device__ __inline__ DEdgeHandle add_edge(const VertexHandle src,
                                               const VertexHandle dest);


    /**
     * @brief add a new face to the patch defined by three edges. These edges
     * handles could come from get_cavity_edge() or add_edge(). The returned
     * handle could be used to access the face attributes
     */
    __device__ __inline__ FaceHandle add_face(const DEdgeHandle e0,
                                              const DEdgeHandle e1,
                                              const DEdgeHandle e2);

   private:
    /**
     * @brief allocate memory and then load the mesh FE and EV into shared
     * memory
     */
    __device__ __inline__ void load_mesh_async(
        cooperative_groups::thread_block& block,
        ShmemAllocator&                   shrd_alloc);


    /**
     * @brief propage the cavity ID from the seeds to indicent/adjacent elements
     * based on CavityOp
     */
    __device__ __inline__ void propagate(
        cooperative_groups::thread_block& block);

    /**
     * @brief propagate the cavity tag from vertices to their incident edges
     */
    __device__ __inline__ void mark_edges_through_vertices();


    /**
     * @brief propagate the cavity tag from edges to their incident faces
     */
    __device__ __inline__ void mark_faces_through_edges();


    /**
     * @brief mark element and deactivate cavities if there is a conflict. Each
     * element should be marked by one cavity. In case of conflict, the cavity
     * with min id wins. If the element has been marked previously with cavity
     * of higher ID, this higher ID cavity will be deactivated. If the element
     * has been already been marked with a cavity of lower ID, the current
     * cavity (cavity_id) will be deactivated
     * This function assumes no other thread is trying to update element_id's
     * cavity ID
     */
    __device__ __inline__ void mark_element(uint16_t*      element_cavity_id,
                                            const uint16_t element_id,
                                            const uint16_t cavity_id);

    /**
     * @brief deactivate the cavities that has been marked as inactivate in the
     * bitmask (m_s_active_cavity_bitmask) by reverting all mesh element ID
     * assigned to these cavities to be INVALID16
     */
    __device__ __inline__ void deactivate_conflicting_cavities();


    /**
     * @brief revert the element cavity ID to INVALID16 if the element's cavity
     * ID is a cavity that has been marked as inactive in
     * m_s_active_cavity_bitmask
     */
    __device__ __inline__ void deactivate_conflicting_cavities(
        const uint16_t num_elements,
        uint16_t*      element_cavity_id);


    /**
     * @brief clear the bit corresponding to an element in the active bitmask if
     * the element is in a cavity. Apply this for vertices, edges and face
     */
    __device__ __inline__ void clear_bitmask_if_in_cavity();

    /**
     * @brief clear the bit corresponding to an element in the bitmask if the
     * element is in a cavity
     */
    __device__ __inline__ void clear_bitmask_if_in_cavity(
        Bitmask&        active_bitmask,
        Bitmask&        in_cavity,
        const uint16_t* element_cavity_id,
        const uint16_t  num_elements);

    /**
     * @brief construct the cavities boundary loop for all cavities created in
     * this patch
     */
    template <uint32_t itemPerThread = 5>
    __device__ __inline__ void construct_cavities_edge_loop(
        cooperative_groups::thread_block& block);

    /**
     * @brief sort cavities edge loop for all cavities created in this patch
     */
    __device__ __inline__ void sort_cavities_edge_loop();

    /**
     * @brief find the index of the next element to add. We do this by
     * atomically attempting to set the active_bitmask until we find an element
     * where we successfully flipped its status from inactive to active.
     */
    __device__ __inline__ uint16_t add_element(Bitmask        active_bitmask,
                                               const uint16_t num_elements);

    //
    // num_cavities could be uint16_t but we use int since we need atomicAdd
    int* m_s_num_cavities;

    // the prefix sum of the cavities sizes. the size of the cavity is the
    // number of boundary edges in the cavity
    // this also could have be uint16_t but we use itn since we do atomicAdd on
    // it
    int* m_s_cavity_size_prefix;

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