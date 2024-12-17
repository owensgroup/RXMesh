#pragma once
#include <memory>
#include <unordered_map>
#include <vector>
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/patch_info.h"
#include "rxmesh/patcher/patcher.h"
#include "rxmesh/types.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include "rxmesh/util/timer.h"

class RXMeshTest;

namespace rxmesh {

/**
 * @brief The main class for creating RXMesh data structure. It takes an input
 * mesh on the host, computes the patches, and creates the data structure on the
 * GPU. It is not mean to be used directly by the user. Users should use
 * RXMeshStatic instead
 */
class RXMesh
{
   public:
    /**
     * @brief Total number of vertices in the mesh
     */
    uint32_t get_num_vertices() const
    {
        return m_num_vertices;
    }

    /**
     * @brief return the number of colors from the 2-ring graph coloring
     * @return
     */
    uint32_t get_num_colors() const
    {
        return m_num_colors;
    }

    uint32_t get_num_vertices(bool from_device)
    {
        if (from_device) {
            CUDA_ERROR(cudaMemcpy(&m_num_vertices,
                                  m_rxmesh_context.m_num_vertices,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
        return m_num_vertices;
    }

    /**
     * @brief Total number of edges in the mesh
     */
    uint32_t get_num_edges() const
    {
        return m_num_edges;
    }

    uint32_t get_num_edges(bool from_device)
    {
        if (from_device) {
            CUDA_ERROR(cudaMemcpy(&m_num_edges,
                                  m_rxmesh_context.m_num_edges,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
        return m_num_edges;
    }

    /**
     * @brief Total number of faces in the mesh
     */
    uint32_t get_num_faces() const
    {
        return m_num_faces;
    }
    uint32_t get_num_faces(bool from_device)
    {
        if (from_device) {
            CUDA_ERROR(cudaMemcpy(&m_num_faces,
                                  m_rxmesh_context.m_num_faces,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
        return m_num_faces;
    }

    /**
     * @brief return the number of mesh elements (vertices, edges, or faces)
     * based on a template paramter input.
     */
    template <typename HandleT>
    uint32_t get_num_elements() const
    {
        static_assert(
            std::is_same_v<HandleT, VertexHandle> ||
                std::is_same_v<HandleT, EdgeHandle> ||
                std::is_same_v<HandleT, FaceHandle>,
            "Template paramter should be either Vertex/Edge/FaceHandle");
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            return get_num_vertices();
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            return get_num_edges();
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            return get_num_faces();
        }
    }

    /**
     * @brief Maximum valence in the input mesh
     */
    uint32_t get_input_max_valence() const
    {
        return m_input_max_valence;
    }

    /**
     * @brief Maximum number of incident faces to an edge in the input mesh
     */
    uint32_t get_input_max_edge_incident_faces() const
    {
        return m_input_max_edge_incident_faces;
    }

    /**
     * @brief Maximum number of adjacent faces to a face in the input mesh
     */
    uint32_t get_input_max_face_adjacent_faces() const
    {
        return m_input_max_face_adjacent_faces;
    }

    /**
     * @brief Return a context that store various information about the mesh on
     * the GPU
     */
    const Context& get_context() const
    {
        return m_rxmesh_context;
    }

    /**
     * @brief returns true if the input mesh is manifold
     */
    bool is_edge_manifold() const
    {
        return m_is_input_edge_manifold;
    }

    /**
     * @brief returns true if the input mesh is closed
     */
    bool is_closed() const
    {
        return m_is_input_closed;
    }

    /**
     * @brief returns the patch size used during partitioning the input mesh
     */
    uint32_t get_patch_size() const
    {
        return m_patch_size;
    }

    /**
     * @brief Total number of patches of the input mesh
     */
    uint32_t get_num_patches() const
    {
        return m_num_patches;
    }
    uint32_t get_num_patches(bool from_device)
    {
        if (from_device) {
            CUDA_ERROR(cudaMemcpy(&m_num_patches,
                                  m_rxmesh_context.m_num_patches,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
        return m_num_patches;
    }

    /**
     * @brief the maximum number of patches
     */
    uint32_t get_max_num_patches() const
    {
        return m_max_num_patches;
    }

    /**
     * @brief Returns the number of disconnected component the input mesh is
     * composed of
     */
    uint32_t get_num_components() const
    {
        return m_patcher->get_num_components();
    }

    /**
     * @brief Return the max, min, and average patch size of the input mesh
     */
    void get_max_min_avg_patch_size(uint32_t& min_p,
                                    uint32_t& max_p,
                                    uint32_t& avg_p) const
    {
        return m_patcher->get_max_min_avg_patch_size(min_p, max_p, avg_p);
    }

    /**
     * @brief Return (approximate) overhead due to ribbons
     */
    double get_ribbon_overhead() const
    {
        return m_patcher->get_ribbon_overhead();
    }

    /**
     * @brief Maximum number of vertices in a patch
     */
    uint32_t get_per_patch_max_vertices() const
    {
        return m_max_vertices_per_patch;
    }

    uint32_t get_per_patch_max_vertices(bool from_device)
    {
        if (from_device) {
            CUDA_ERROR(cudaMemcpy(&m_max_vertices_per_patch,
                                  m_rxmesh_context.m_max_num_vertices,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
        return m_max_vertices_per_patch;
    }

    /**
     * @brief Maximum number of edges in a patch
     */
    uint32_t get_per_patch_max_edges() const
    {
        return m_max_edges_per_patch;
    }

    uint32_t get_per_patch_max_edges(bool from_device)
    {
        if (from_device) {
            CUDA_ERROR(cudaMemcpy(&m_max_edges_per_patch,
                                  m_rxmesh_context.m_max_num_edges,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
        return m_max_edges_per_patch;
    }
    /**
     * @brief Maximum number of faces in a patch
     */
    uint32_t get_per_patch_max_faces() const
    {
        return m_max_faces_per_patch;
    }

    uint32_t get_per_patch_max_faces(bool from_device)
    {
        if (from_device) {
            CUDA_ERROR(cudaMemcpy(&m_max_faces_per_patch,
                                  m_rxmesh_context.m_max_num_faces,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
        return m_max_faces_per_patch;
    }

    /**
     * @brief The time used to construct the patches on the GPU
     */
    float get_patching_time() const
    {
        return m_patcher->get_patching_time();
    }

    /**
     * @brief The number of Lloyd iterations run to partition the mesh into
     * patches
     */
    uint32_t get_num_lloyd_run() const
    {
        return m_patcher->get_num_lloyd_run();
    }

    /**
     * @brief Return the edge id given two vertices. Edges are undirected.
     * @param v0 first input vertex
     * @param v1 second input vertex
     * @return edge id composed by v0-v1 (same as edge id for v1-v0)
     */
    uint32_t get_edge_id(const uint32_t v0, const uint32_t v1) const;

    /**
     * @brief save/seralize the patcher info to a file
     * @param filename
     */
    virtual void save(std::string filename)
    {
        m_patcher->save(filename);
    }

    /**
     * @brief map a global vertex index to a VertexHandle i.e., a local vertex
     * Note: The mapping is different than the mapping from the input
     * @param i global vertex index
     */
    const VertexHandle map_to_local_vertex(uint32_t i) const;

    /**
     * @brief map a global vertex index to a EdgexHandle i.e., a local edge.
     * Note: The mapping is different than the mapping from the input
     * @param i global edge index
     */
    const EdgeHandle map_to_local_edge(uint32_t i) const;

    /**
     * @brief map a global vertex index to an FaceHandle i.e., a local face
     * Note: The mapping is different than the mapping from the input
     * @param i global face index
     */
    const FaceHandle map_to_local_face(uint32_t i) const;

    /**
     * @brief return the number of owned vertices in a patch
     */
    uint16_t get_num_owned_vertices(const uint32_t p) const
    {
        return m_h_num_owned_v[p];
    }

    /**
     * @brief return the number of owned edges in a patch
     */
    uint16_t get_num_owned_edges(const uint32_t p) const
    {
        return m_h_num_owned_e[p];
    }

    /**
     * @brief return the number of owned faces in a patch
     */
    uint16_t get_num_owned_faces(const uint32_t p) const
    {
        return m_h_num_owned_f[p];
    }


    /**
     * @brief return the number of vertices in a patch
     */
    uint16_t get_num_vertices(const uint32_t p) const
    {
        return m_h_patches_info[p].num_vertices[0];
    }

    /**
     * @brief return the number of edges in a patch
     */
    uint16_t get_num_edges(const uint32_t p) const
    {
        return m_h_patches_info[p].num_edges[0];
    }

    /**
     * @brief return the number of faces in a patch
     */
    uint16_t get_num_faces(const uint32_t p) const
    {
        return m_h_patches_info[p].num_faces[0];
    }

    const PatchInfo& get_patch(uint32_t p) const
    {
        assert(p < get_num_patches());
        return m_h_patches_info[p];
    }

    template <typename LocalT>
    uint32_t max_bitmask_size() const
    {
        if constexpr (std::is_same_v<LocalT, LocalVertexT>) {
            return detail::mask_num_bytes(this->m_max_vertices_per_patch);
        }

        if constexpr (std::is_same_v<LocalT, LocalEdgeT>) {
            return detail::mask_num_bytes(this->m_max_edges_per_patch);
        }

        if constexpr (std::is_same_v<LocalT, LocalFaceT>) {
            return detail::mask_num_bytes(this->m_max_faces_per_patch);
        }
    }

    /**
     * @brief return the amount of allocated memory for topology information in
     * megabytes
     */
    double get_topology_memory_mg() const
    {
        return m_topo_memory_mega_bytes;
    }

   protected:
    // Edge hash map that takes two vertices and return their edge id
    using EdgeMapT = std::unordered_map<std::pair<uint32_t, uint32_t>,
                                        uint32_t,
                                        detail::edge_key_hash>;

    virtual ~RXMesh();

    RXMesh(const RXMesh&) = delete;

    RXMesh(uint32_t patch_size);

    /**
     * @brief init all the data structures
     * @param fv the mesh connectivity as an index triangle
     * @param patcher_file optional file to load the patches
     * @param capacity_factor capacity factor the determine the max allocation
     * size of a patch as a fraction of its size. For example, a patch with x
     * faces will be allocated with size that fits capactiy_factor*x faces
     * @param patch_alloc_factor determine the max number of patches. If the
     * input mesh is patched into x patches, we will allocate a space for
     * patch_alloc_factor*x patches
     * @param lp_hashtable_load_factor loading factor for the hashtable use for
     * the not-owned vertices/edges/faces
     */
    void init(const std::vector<std::vector<uint32_t>>& fv,
              const std::string                         patcher_file    = "",
              const float                               capacity_factor = 1.8,
              const float patch_alloc_factor                            = 5.0,
              const float lp_hashtable_load_factor                      = 0.5);

    /**
     * @brief build different supporting data structure used to build RXMesh
     *
     * Set the number of vertices, edges, and faces, populate edge_map (which
     * takes two connected vertices and returns their edge id), build
     * face-incident-faces data structure (used to in creating patches). This is
     * done using a single pass over FV
     *
     * @param fv input face incident vertices
     * @param ef output edge incident faces
     * @param ef output face adjacent faces
     */
    void build_supporting_structures(
        const std::vector<std::vector<uint32_t>>& fv,
        std::vector<std::vector<uint32_t>>&       ev,
        std::vector<std::vector<uint32_t>>&       ef,
        std::vector<uint32_t>&                    ff_offset,
        std::vector<uint32_t>&                    ff_values);

    /**
     * @brief Calculate various statistics for the input mesh
     *
     * Calculate max valence, max edge incident faces, max face adjacent faces,
     * if the input is closed, if the input is edge manifold, and max number of
     * vertices/edges/faces per patch
     *
     * @param fv input face incident vertices
     * @param ef input edge incident faces
     */
    void calc_input_statistics(const std::vector<std::vector<uint32_t>>& fv,
                               const std::vector<std::vector<uint32_t>>& ef);

    /**
     * @brief count the max number of vertices/edges/faces per patch and
     * the max number of not-owned vertices/edges/faces per patch
     * @param on_device either do the computation on device or host
     */
    void calc_max_elements();

    /**
     * @brief allocate extra patches needed in cases the number of patches
     * increases. We allocate these patches space such that they can occupy the
     * same size as the largest patch in the input mesh
     */
    void allocate_extra_patches();

    template <typename HandleT>
    const std::pair<uint32_t, uint16_t> map_to_local(
        const uint32_t  i,
        const uint32_t* element_prefix) const;

    template <typename LocalT>
    uint16_t max_lp_hashtable_capacity() const
    {
        if constexpr (std::is_same_v<LocalT, LocalVertexT>) {
            // return find_next_prime_number(static_cast<uint16_t>(
            //     std::ceil(m_capacity_factor *
            //               static_cast<float>(m_max_not_owned_vertices) /
            //               m_lp_hashtable_load_factor)));
            return m_max_capacity_lp_v;
        }

        if constexpr (std::is_same_v<LocalT, LocalEdgeT>) {
            // return find_next_prime_number(static_cast<uint16_t>(std::ceil(
            //     m_capacity_factor * static_cast<float>(m_max_not_owned_edges)
            //     / m_lp_hashtable_load_factor)));
            return m_max_capacity_lp_e;
        }

        if constexpr (std::is_same_v<LocalT, LocalFaceT>) {
            // return find_next_prime_number(static_cast<uint16_t>(std::ceil(
            //     m_capacity_factor * static_cast<float>(m_max_not_owned_faces)
            //     / m_lp_hashtable_load_factor)));
            return m_max_capacity_lp_f;
        }
    }

    void build(const std::vector<std::vector<uint32_t>>& fv,
               const std::string                         patcher_file);

    void build_single_patch_ltog(const std::vector<std::vector<uint32_t>>& fv,
                                 const std::vector<std::vector<uint32_t>>& ev,
                                 const uint32_t patch_id);

    void build_single_patch_topology(
        const std::vector<std::vector<uint32_t>>& fv,
        const uint32_t                            patch_id);

    // get the max vertex/edge/face capacity i.e., the max number of
    // vertices/edges/faces allowed in a patch (for allocation purposes)
    uint16_t get_per_patch_max_vertex_capacity() const;
    uint16_t get_per_patch_max_edge_capacity() const;
    uint16_t get_per_patch_max_face_capacity() const;

    void build_device();
    void build_device_single_patch(const uint32_t patch_id,
                                   const uint16_t p_num_vertices,
                                   const uint16_t p_num_edges,
                                   const uint16_t p_num_faces,
                                   const uint16_t p_vertices_capacity,
                                   const uint16_t p_edges_capacity,
                                   const uint16_t p_faces_capacity,
                                   const uint16_t p_num_owned_vertices,
                                   const uint16_t p_num_owned_edges,
                                   const uint16_t p_num_owned_faces,
                                   const std::vector<uint32_t>& ltog_v,
                                   const std::vector<uint32_t>& ltog_e,
                                   const std::vector<uint32_t>& ltog_f,
                                   PatchInfo&                   h_patch_info,
                                   PatchInfo&                   d_patch_info);

    void patch_graph_coloring();

    void populate_patch_stash();

    uint32_t get_edge_id(const std::pair<uint32_t, uint32_t>& edge) const;

    friend class ::RXMeshTest;

    template <typename T, typename HandleT>
    friend class Attribute;


    Context  m_rxmesh_context;
    EdgeMapT m_edges_map;

    // Should be updated with update_host
    uint32_t m_num_edges, m_num_faces, m_num_vertices;

    uint32_t m_max_edge_capacity, m_max_face_capacity, m_max_vertex_capacity;

    uint32_t m_input_max_valence, m_input_max_edge_incident_faces,
        m_input_max_face_adjacent_faces;
    bool m_is_input_edge_manifold;
    bool m_is_input_closed;


    uint32_t       m_num_patches, m_max_num_patches;
    const uint32_t m_patch_size;

    // pointer to the patcher class responsible for everything related to
    // patching the mesh into small pieces
    std::unique_ptr<patcher::Patcher> m_patcher;

    // the number of owned mesh elements per patch
    std::vector<uint16_t> m_h_num_owned_f, m_h_num_owned_e, m_h_num_owned_v;

    // uint16_t m_max_not_owned_vertices, m_max_not_owned_edges,
    //    m_max_not_owned_faces;

    uint16_t m_max_capacity_lp_v, m_max_capacity_lp_e, m_max_capacity_lp_f;

    uint32_t m_max_vertices_per_patch, m_max_edges_per_patch,
        m_max_faces_per_patch;

    // mappings
    // local to global map for (v)ertices (e)dges and (f)aces
    // Should be invalidated with update_host
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_v;
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_e;
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_f;

    // the prefix sum of the owned vertices/edges/faces in patches
    uint32_t* m_h_vertex_prefix;
    uint32_t* m_h_edge_prefix;
    uint32_t* m_h_face_prefix;

    uint32_t *m_d_vertex_prefix, *m_d_edge_prefix, *m_d_face_prefix;

    PatchInfo *m_d_patches_info, *m_h_patches_info;

    float m_capacity_factor, m_lp_hashtable_load_factor, m_patch_alloc_factor;

    double m_topo_memory_mega_bytes;

    uint32_t m_num_colors;

    Timers<CPUTimer> m_timers;
};
}  // namespace rxmesh
