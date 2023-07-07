#pragma once

#include <cooperative_groups.h>

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/patch_info.h"

#include "rxmesh/attribute.h"

#include "rxmesh/kernels/shmem_mutex.cuh"

namespace rxmesh {

template <uint32_t blockThreads, CavityOp cop>
struct CavityManager
{
    /**
     * @brief default constructor
     */
    __device__ __inline__ CavityManager()
        : m_write_to_gmem(true),
          m_s_num_cavities(nullptr),
          m_s_cavity_size_prefix(nullptr),
          m_s_readd_to_queue(nullptr),
          m_s_ev(nullptr),
          m_s_fe(nullptr),
          m_s_cavity_id_v(nullptr),
          m_s_cavity_id_e(nullptr),
          m_s_cavity_id_f(nullptr),
          m_s_table_v(nullptr),
          m_s_table_e(nullptr),
          m_s_table_f(nullptr),
          m_s_num_vertices(nullptr),
          m_s_num_edges(nullptr),
          m_s_num_faces(nullptr),
          m_s_cavity_boundary_edges(nullptr)
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
     * @brief check if a seed was successful in creating its cavity
     * Note that not all cavities can be created (by calling create()) since
     * some of them could be conflicting, rather we select (hopefully maximal)
     * set of non-conflicting cavities. This function can be used to check if a
     * specific seed is in this set. This function can only be called after
     * calling prologue()
     *
     */
    template <typename HandleT>
    __device__ __inline__ bool is_successful(HandleT seed);

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
     * @brief return the patch id that this cavity manager operates on
     */
    __device__ __forceinline__ uint32_t patch_id() const
    {
        return m_patch_info.patch_id;
    }

    /**
     * @brief return the patch info assigned to this cavity manager
     */
    __device__ __forceinline__ const PatchInfo& patch_info() const
    {
        return m_patch_info;
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
        assert(c < m_s_num_cavities[0]);
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


    /**
     * @brief update all attributes such that it can be used after the topology
     * changes. This function takes as many attributes as you want
     */
    template <typename... AttributesT>
    __device__ __inline__ void update_attributes(
        cooperative_groups::thread_block& block,
        AttributesT&&... attributes)
    {
        // use fold expersion to iterate over each attribute
        // https://stackoverflow.com/a/60136761
        ([&] { update_attribute(attributes); }(), ...);
        block.sync();
    }

    /**
     * @brief cleanup and store updated patch to global memory
     * @param block
     * @return
     */
    __device__ __inline__ void epilogue(
        cooperative_groups::thread_block& block);

    /**
     * @brief should this patch be sliced. Populated after calling prologue()
     */
    __device__ __inline__ bool should_slice() const
    {
        return m_s_should_slice[0];
    }

   private:
    /**
     * @brief allocate shared memory
     */
    __device__ __inline__ void alloc_shared_memory(
        cooperative_groups::thread_block& block,
        ShmemAllocator&                   shrd_alloc);

    /**
     * @brief load hashtable into shared memory
     */
    __device__ __inline__ void load_hashtable(
        cooperative_groups::thread_block& block);

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
     * @brief deactivate a cavity if it requires changing ownership of elements
     * from another patch
     */
    __device__ __inline__ void deactivate_boundary_cavities(
        cooperative_groups::thread_block& block);

    /**
     * @brief revert the element cavity ID to INVALID16 if the element's cavity
     * ID is a cavity that has been marked as inactive in
     * m_s_active_cavity_bitmask
     */
    __device__ __inline__ void deactivate_conflicting_cavities(
        const uint16_t num_elements,
        uint16_t*      element_cavity_id,
        const Bitmask& active_bitmask);

    /**
     * @brief reactivate (set to active) elements if they have been to fall in a
     * cavity that has been deactivated
     */
    __device__ __inline__ void reactivate_elements();

    __device__ __inline__ void reactivate_elements(Bitmask&  active_bitmask,
                                                   Bitmask&  in_cavity,
                                                   uint16_t* element_cavity_id,
                                                   const uint16_t num_elements);


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
     * If we fail, we just atomically increment num_elements and set the
     * corresponding bit in active_bitmask. There is a special case when it
     * comes to fill in spots of in-cavity elements that are not-owned. We need
     * to leave these spots clear since we use them as a key in the hashtable
     * in order to change their ownership. If we filled them in, and added an
     * element (which are initially not-owned), we will pollute the hashtable
     * and won't be able to get the owner element in order to change their
     * ownership flag during ownership_change(). However, after
     * ownership_change, we should leave only spots that are in-cavity and
     * not-owned since we use the corresponding entries in the hashtable of
     * these spots in hashtable calibration
     */
    __device__ __inline__ uint16_t add_element(Bitmask&       active_bitmask,
                                               uint16_t*      num_elements,
                                               const uint16_t capacity,
                                               const Bitmask& in_cavity,
                                               const Bitmask& owned,
                                               bool avoid_not_owned_in_cavity);

    /**
     * @brief enqueue patch in the patch scheduler so that it can be scheduled
     * latter
     */
    __device__ __inline__ void push();

    /**
     * @brief release the lock of this patch
     */
    __device__ __forceinline__ void unlock();


    /**
     * @brief try to acquire the lock of the patch q. The block call this
     * function while only one thread is needed to do the job but we broadcast
     * the results to all threads
     */
    __device__ __forceinline__ bool lock(
        cooperative_groups::thread_block& block,
        const uint8_t                     stash_id,
        const uint32_t                    q);


    /**
     * @brief release the lock acquired earlier for the patch q.
     */
    __device__ __forceinline__ void unlock(const uint8_t  stash_id,
                                           const uint32_t q);

    /**
     * @brief unlock all locked patches done by this cavity manager
     * @return
     */
    __device__ __inline__ void unlock_locked_patches();

    /**
     * @brief prepare m_s_migrate_mask_v, m_s_owned_cavity_bdry_v,
     * m_s_ownership_change_mask_v, m_s_ownership_change_mask_f,
     * m_s_ownership_change_mask_e with vertices/edge/faces that on the boundary
     * or inside a cavity but not owned by this patch
     */
    __device__ __inline__ void pre_migrate(
        cooperative_groups::thread_block& block);

    /**
     * @brief migrate vertices/edges/faces from neighbor patches to this patch
     */
    __device__ __inline__ bool migrate(cooperative_groups::thread_block& block);

    /**
     * @brief given a neighbor patch (q), migrate vertices (and edges and faces
     * connected to these vertices) marked in migrate_mask_v to the patch
     * managed by this cavity manager
     */
    __device__ __inline__ bool migrate_from_patch(
        cooperative_groups::thread_block& block,
        const uint8_t                     q_stash_id,
        const uint32_t                    q,
        const Bitmask&                    migrate_mask_v,
        const bool                        change_ownership);

    /**
     * @brief give a neighbor patch q and a vertex in it q_vertex, find the copy
     * of q_vertex in this patch. If it does not exist, create such a copy.
     */
    template <typename FuncT>
    __device__ __inline__ LPPair migrate_vertex(
        const uint32_t q,
        const uint16_t q_num_vertices,
        const uint16_t q_vertex,
        const bool     require_ownership_change,
        PatchInfo&     q_patch_info,
        FuncT          should_migrate);


    /**
     * @brief give a neighbor patch q and an edge in it q_edge, find the copy
     * of q_edge in this patch. If it does not exist, create such a copy.
     */
    template <typename FuncT>
    __device__ __inline__ LPPair migrate_edge(
        const uint32_t q,
        const uint16_t q_num_edges,
        const uint16_t q_edge,
        const bool     require_ownership_change,
        PatchInfo&     q_patch_info,
        FuncT          should_migrate);


    /**
     * @brief give a neighbor patch q and a face in it q_face, find the copy
     * of q_face in this patch. If it does not exist, create such a copy.
     */
    template <typename FuncT>
    __device__ __inline__ LPPair migrate_face(
        const uint32_t q,
        const uint16_t q_num_faces,
        const uint16_t q_face,
        const bool     require_ownership_change,
        PatchInfo&     q_patch_info,
        FuncT          should_migrate);

    /**
     * @brief given a local vertex in a patch, find its corresponding local
     * index in the patch associated with this cavity i.e., m_patch_info.
     * If the given vertex (local_id) is not owned by the given patch, they will
     * be mapped to their owner patch and local index in the owner patch.
     */
    __device__ __inline__ uint16_t find_copy_vertex(uint16_t& local_id,
                                                    uint32_t& patch);

    /**
     * @brief given a local edge in a patch, find its corresponding local
     * index in the patch associated with this cavity i.e., m_patch_info.
     * If the given edge (local_id) is not owned by the given patch, they will
     * be mapped to their owner patch and local index in the owner patch
     */
    __device__ __inline__ uint16_t find_copy_edge(uint16_t& local_id,
                                                  uint32_t& patch);

    /**
     * @brief given a local face in a patch, find its corresponding local
     * index in the patch associated with this cavity i.e., m_patch_info.
     * If the given face (local_id) is not owned by the given patch, they will
     * be mapped to their owner patch and local index in the owner patch
     */
    __device__ __inline__ uint16_t find_copy_face(uint16_t& local_id,
                                                  uint32_t& patch);


    /**
     * @brief find a copy of mesh element from a src_patch in a dest_patch i.e.,
     * the lid lives in src_patch and we want to find the corresponding local
     * index in dest_patch
     */
    template <typename HandleT>
    __device__ __inline__ uint16_t find_copy(
        uint16_t&      lid,
        uint32_t&      src_patch,
        const uint16_t dest_patch_num_elements,
        const Bitmask& dest_patch_owned_mask,
        const Bitmask& dest_patch_active_mask,
        const Bitmask& dest_in_cavity,
        const LPPair*  s_table,
        const LPPair*  s_stash);


    /**
     * @brief change vertices, edges, and faces ownership as marked in
     * m_s_ownership_change_mask_v/e/f
     */
    __device__ __inline__ void change_ownership(
        cooperative_groups::thread_block& block);

    /**
     * @brief change ownership for mesh elements of type HandleT marked in
     * s_ownership_change. We can remove these mesh elements from the
     * hashtable, but we delay this (do it in cleanup) since we need to get
     * these mesh elements' original owner patch in update_attributes()
     */
    template <typename HandleT>
    __device__ __inline__ void change_ownership(
        cooperative_groups::thread_block& block,
        const uint16_t                    num_elements,
        const Bitmask&                    s_ownership_change,
        const LPPair*                     s_table,
        const LPPair*                     s_stash,
        Bitmask&                          s_owned_bitmask);

    /**
     * @brief update an attribute such that it can be used after the topology
     * changes
     */
    template <typename AttributeT>
    __device__ __inline__ void update_attribute(AttributeT& attribute);

    /**
     * @brief lock patches marked in m_s_patches_to_lock_mask. Return true
     * if all patches were locked and false otherwise. Update
     */
    __device__ __inline__ bool lock_patches_to_lock(
        cooperative_groups::thread_block& block);

    /**
     * @brief prepare m_s_ribbonize_v with vertices that need to be ribbonize
     */
    __device__ __inline__ void pre_ribbonize(
        cooperative_groups::thread_block& block);

    /**
     * @brief for each element we ask to change their ownership (to be owned
     * by this patch m_patch_info), we store the owner patch in the hashtable.
     * This happens during migrate(). Here, we make sure that the owner patch
     * stored in the hashtable is 1. locked 2. the actual owner. The reason
     * behind this check is we want to always read from locked patches. In
     * find_copy(), the other patch q may have given up the ownership of the
     * element earlier. Thus, we need to jump from q to the other new owner s.
     * However, s is no longer locked and we can not read it from it. In this
     * cases, we need a hashtable_calibration first.
     */
    template <typename HandleT>
    __device__ __inline__ bool ensure_ownership(
        cooperative_groups::thread_block& block,
        const uint16_t                    num_elements,
        const Bitmask&                    s_ownership_change,
        const LPPair*                     s_table,
        const LPPair*                     s_stash);

    /**
     * @brief give a patch q, set a bit in s_in_patch if element x appears in
     * s_table/s_stash.
     */
    template <typename HandleT>
    __device__ __inline__ void populate_in_patch(
        cooperative_groups::thread_block& block,
        uint8_t                           q_stash,
        Bitmask&                          s_in_patch,
        const LPPair*                     s_table,
        const LPPair*                     s_stash);

    // indicate if this block can write its updates to global memory during
    // epilogue
    bool m_write_to_gmem;

    // num_cavities could be uint16_t but we use int since we need atomicAdd
    int* m_s_num_cavities;

    // the prefix sum of the cavities sizes. the size of the cavity is the
    // number of boundary edges in the cavity
    // this also could have be uint16_t but we use itn since we do atomicAdd on
    // it
    int* m_s_cavity_size_prefix;


    // some cavities are inactive since they overlap with other cavities.
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


    // indicate if the mesh element is in the interior of the cavity
    Bitmask m_s_in_cavity_v, m_s_in_cavity_e, m_s_in_cavity_f;

    // given a patch q, this bitmask store whether a vertex/edge/face in q is
    // stored in the hashtable of this patch (p). Thus, this bitmask used the
    // index space of q
    Bitmask m_s_in_patch_v, m_s_in_patch_e, m_s_in_patch_f;

    bool* m_s_readd_to_queue;

    // mesh connectivity
    uint16_t *m_s_ev, *m_s_fe;

    // store the cavity ID each mesh element belong to. If the mesh element
    // does not belong to any cavity, then it stores INVALID32
    uint16_t *m_s_cavity_id_v, *m_s_cavity_id_e, *m_s_cavity_id_f;

    // the hashtable (this memory overlaps with m_s_cavity_id_v/e/f)
    LPPair *m_s_table_v, *m_s_table_e, *m_s_table_f;
    LPPair *m_s_table_stash_v, *m_s_table_stash_e, *m_s_table_stash_f;

    // store the number of elements. we use pointers since the number of mesh
    // elements could change
    uint16_t *m_s_num_vertices, *m_s_num_edges, *m_s_num_faces;

    // store the boundary edges of all cavities in compact format (similar to
    // CSR for sparse matrices using m_s_cavity_size_prefix but no value ptr)
    uint16_t* m_s_cavity_boundary_edges;

    // patch stash stored in shared memory
    PatchStash m_s_patch_stash;

    PatchInfo m_patch_info;
    Context   m_context;

    bool*      m_s_should_slice;
    ShmemMutex m_s_patch_stash_mutex;
};

}  // namespace rxmesh

#include "rxmesh/cavity_manager_impl.cuh"