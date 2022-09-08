#pragma once
#include <memory>
#include <unordered_map>
#include <vector>
#include "rxmesh/context.h"
#include "rxmesh/patch_info.h"
#include "rxmesh/patcher/patcher.h"
#include "rxmesh/types.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

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
     * @brief Total number of edges in the mesh
     */
    uint32_t get_num_edges() const
    {
        return m_num_edges;
    }

    /**
     * @brief Total number of faces in the mesh
     */
    uint32_t get_num_faces() const
    {
        return m_num_faces;
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

    /**
     * @brief Maximum number of edges in a patch
     */
    uint32_t get_per_patch_max_edges() const
    {
        return m_max_edges_per_patch;
    }

    /**
     * @brief Maximum number of faces in a patch
     */
    uint32_t get_per_patch_max_faces() const
    {
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
    void save(std::string filename)
    {
        m_patcher->save(filename);
    }

   protected:
    // Edge hash map that takes two vertices and return their edge id
    using EdgeMapT = std::unordered_map<std::pair<uint32_t, uint32_t>,
                                        uint32_t,
                                        detail::edge_key_hash>;

    virtual ~RXMesh();

    RXMesh(const RXMesh&) = delete;

    RXMesh();

    void init(const std::vector<std::vector<uint32_t>>& fv,
              const std::string                         patcher_file    = "",
              const bool                                quite           = false,
              const float                               capacity_factor = 1.2,
              const float lp_hashtable_load_factor                      = 0.9);

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

    uint32_t max_bitmask_size(ELEMENT ele) const;

    uint32_t max_lp_hashtable_size(ELEMENT ele) const;

    void build(const std::vector<std::vector<uint32_t>>& fv,
               const std::string                         patcher_file);
    void build_single_patch(const std::vector<std::vector<uint32_t>>& fv,
                            const uint32_t                            patch_id);

    void build_single_patch_ltog(const std::vector<std::vector<uint32_t>>& fv,
                                 const uint32_t patch_id);

    void build_single_patch_topology(
        const std::vector<std::vector<uint32_t>>& fv,
        const uint32_t                            patch_id);


    void build_device();

    uint32_t get_edge_id(const std::pair<uint32_t, uint32_t>& edge) const;

    friend class ::RXMeshTest;

    template <typename T>
    friend class VertexAttribute;
    template <typename T>
    friend class EdgeAttribute;
    template <typename T>
    friend class FaceAttribute;

    Context  m_rxmesh_context;
    EdgeMapT m_edges_map;

    // Should be updated with update_host
    uint32_t m_num_edges, m_num_faces, m_num_vertices;

    bool     m_quite;
    uint32_t m_input_max_valence, m_input_max_edge_incident_faces,
        m_input_max_face_adjacent_faces;
    bool m_is_input_edge_manifold;
    bool m_is_input_closed;

    // May need to be updated with update_host
    uint32_t       m_num_patches;
    const uint32_t m_patch_size;

    // pointer to the patcher class responsible for everything related to
    // patching the mesh into small pieces
    std::unique_ptr<patcher::Patcher> m_patcher;

    // the number of owned mesh elements per patch
    // Should be updated with update_host
    std::vector<uint16_t> m_h_num_owned_f, m_h_num_owned_e, m_h_num_owned_v;

    // Should be updated with update_host
    uint32_t m_max_not_owned_vertices, m_max_not_owned_edges,
        m_max_not_owned_faces;
    // Should be updated with update_host
    uint32_t m_max_vertices_per_patch, m_max_edges_per_patch,
        m_max_faces_per_patch;

    // mappings
    // local to global map for (v)ertices (e)dges and (f)aces
    // Should be invalidated with update_host
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_v;
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_e;
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_f;

    //the prefix sum of the owned vertices/edges/faces in patches 
    std::vector<uint32_t> m_h_vertex_prefix;
    std::vector<uint32_t> m_h_edge_prefix;
    std::vector<uint32_t> m_h_face_prefix;


    // Should be updated with update_host
    PatchInfo *m_d_patches_info, *m_h_patches_info;

    float m_capacity_factor, m_lp_hashtable_load_factor;
};
}  // namespace rxmesh
