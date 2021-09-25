#pragma once

#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>
#include "rxmesh/patcher/patcher.h"
#include "rxmesh/rxmesh_context.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"

class RXMeshTest;

namespace rxmesh {

enum class Op
{
    VV = 0,
    VE = 1,
    VF = 2,
    FV = 3,
    FE = 4,
    FF = 5,
    EV = 6,
    EE = 7,
    EF = 8,
};

inline std::string op_to_string(const Op& op)
{
    switch (op) {
        case rxmesh::Op::VV:
            return "VV";
        case rxmesh::Op::VE:
            return "VE";
        case rxmesh::Op::VF:
            return "VF";
        case rxmesh::Op::FV:
            return "FV";
        case rxmesh::Op::FE:
            return "FE";
        case rxmesh::Op::FF:
            return "FF";
        case rxmesh::Op::EV:
            return "EV";
        case rxmesh::Op::EF:
            return "EF";
        case rxmesh::Op::EE:
            return "EE";
        default: {
            RXMESH_ERROR("op_to_string() unknown input operation");
            return "";
        }
    }
}

enum class ELEMENT
{
    VERTEX = 0,
    EDGE   = 1,
    FACE   = 2
};

class RXMesh
{
   public:
    /**
     * @brief Export the mesh to obj file
     *
     * @tparam VertT Lambda function type [inferred]
     * @param filename the output file
     * @param getCoords lambda function that takes two uint32_t/int parameters
     * and return a real number. The first parameter is the vertex id. The
     * second parameter is dimension (0,1, or 2)
     */
    template <typename VertT>
    void export_obj(const std::string& filename, VertT getCoords)
    {
        std::string  fn = STRINGIFY(OUTPUT_DIR) + filename;
        std::fstream file(fn, std::ios::out);
        file.precision(30);

        // write vertices
        for (uint32_t v = 0; v < m_num_vertices; ++v) {
            uint32_t v_id = v;

            file << "v  ";
            for (uint32_t i = 0; i < 3; ++i) {
                file << getCoords(v_id, i) << "  ";
            }
            file << std::endl;
        }
        // write connectivity
        write_connectivity(file);
        file.close();
    }


    uint32_t get_num_vertices() const
    {
        return m_num_vertices;
    }
    uint32_t get_num_edges() const
    {
        return m_num_edges;
    }
    uint32_t get_num_faces() const
    {
        return m_num_faces;
    }

    uint32_t get_max_valence() const
    {
        return m_max_valence;
    }

    uint32_t get_max_edge_incident_faces() const
    {
        return m_max_edge_incident_faces;
    }

    uint32_t get_max_edge_adjacent_faces() const
    {
        return m_max_face_adjacent_faces;
    }
    uint32_t get_face_degree() const
    {
        return m_face_degree;
    }

    const RXMeshContext& get_context() const
    {
        return m_rxmesh_context;
    }

    bool is_edge_manifold() const
    {
        return m_is_input_edge_manifold;
    }

    bool is_closed() const
    {
        return m_is_input_closed;
    }

    uint32_t get_patch_size() const
    {
        return m_patch_size;
    }

    uint32_t get_num_patches() const
    {
        return m_num_patches;
    }

    uint32_t get_num_components() const
    {
        return m_patcher->get_num_components();
    }

    void get_max_min_avg_patch_size(uint32_t& min_p,
                                    uint32_t& max_p,
                                    uint32_t& avg_p) const
    {
        return m_patcher->get_max_min_avg_patch_size(min_p, max_p, avg_p);
    }

    double get_ribbon_overhead() const
    {
        return m_patcher->get_ribbon_overhead();
    }

    uint32_t get_per_patch_max_vertices() const
    {
        return m_max_vertices_per_patch;
    }

    uint32_t get_per_patch_max_edges() const
    {
        return m_max_edges_per_patch;
    }

    uint32_t get_per_patch_max_faces() const
    {
        return m_max_faces_per_patch;
    }

    uint32_t get_per_patch_max_owned_vertices() const
    {
        return m_max_owned_vertices_per_patch;
    }

    uint32_t get_per_patch_max_owned_edges() const
    {
        return m_max_owned_edges_per_patch;
    }

    uint32_t get_per_patch_max_owned_faces() const
    {
        return m_max_owned_faces_per_patch;
    }

    float get_patching_time() const
    {
        return m_patcher->get_patching_time();
    }

    uint32_t get_num_lloyd_run() const
    {
        return m_patcher->get_num_lloyd_run();
    }

    uint32_t get_edge_id(const uint32_t v0, const uint32_t v1) const;

    double get_gpu_storage_mb() const
    {
        return m_total_gpu_storage_mb;
    }

    const std::unique_ptr<patcher::Patcher>& get_patcher() const
    {
        return m_patcher;
    };

   protected:
    virtual ~RXMesh();

    RXMesh(const RXMesh&) = delete;

    RXMesh(std::vector<std::vector<uint32_t>>& fv, const bool quite = true);

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
     */
    void build_supporting_structures(
        const std::vector<std::vector<uint32_t>>& fv,
        std::vector<std::vector<uint32_t>>&       ef);

    /**
     * @brief Calculate various statistics for the input mesh
     *
     * Calculate max valence, max edge incident faces, max face adjacent faces,
     * if the input is closed, and if the input is edge manifold
     * 
     * @param fv input face incident vertices
     * @param ef input edge incident faces
     */
    void calc_statistics(const std::vector<std::vector<uint32_t>>& fv,
                         const std::vector<std::vector<uint32_t>>& ef);

    void     build_local(std::vector<std::vector<uint32_t>>& fv);
    void     build_patch_locally(const uint32_t patch_id);
    
    uint16_t create_new_local_face(const uint32_t               patch_id,
                                   const uint32_t               global_f,
                                   const std::vector<uint32_t>& fv,
                                   uint16_t&                    faces_count,
                                   uint16_t&      edges_owned_count,
                                   uint16_t&      edges_not_owned_count,
                                   uint16_t&      vertices_owned_count,
                                   uint16_t&      vertices_not_owned_count,
                                   const uint16_t num_edges_owned,
                                   const uint16_t num_vertices_owned,
                                   std::vector<uint32_t>& f_ltog,
                                   std::vector<uint32_t>& e_ltog,
                                   std::vector<uint32_t>& v_ltog,
                                   std::vector<uint16_t>& fp,
                                   std::vector<uint16_t>& ep);

    inline std::pair<uint32_t, uint32_t> edge_key(const uint32_t v0,
                                                  const uint32_t v1) const
    {
        uint32_t i = std::max(v0, v1);
        uint32_t j = std::min(v0, v1);
        return std::make_pair(i, j);
    }

    void device_alloc_local();

    template <typename Tin, typename Tst>
    void get_starting_ids(const std::vector<std::vector<Tin>>& input,
                          std::vector<Tst>&                    starting_id);

    template <typename T>
    void padding_to_multiple(std::vector<std::vector<T>>& input,
                             const uint32_t               multiple,
                             const T                      init_val);

    template <typename Tin, typename Tad>
    void get_size(const std::vector<std::vector<Tin>>& input,
                  std::vector<Tad>&                    ad);

    struct edge_key_hash
    {
        // www.techiedelight.com/use-std-pair-key-std-unordered_map-cpp/
        template <class T>
        inline std::size_t operator()(const std::pair<T, T>& e_key) const
        {
            return std::hash<T>()(e_key.first * 8191 + e_key.second * 11003);
        }
    };

    uint32_t get_edge_id(const std::pair<uint32_t, uint32_t>& edge) const;

    virtual void write_connectivity(std::fstream& file) const;

    // our friend tester class
    friend class ::RXMeshTest;

    RXMeshContext m_rxmesh_context;

    uint32_t m_num_edges, m_num_faces, m_num_vertices, m_max_ele_count,
        m_max_valence, m_max_edge_incident_faces, m_max_face_adjacent_faces;
    const uint32_t m_face_degree;

    // patches
    uint32_t       m_num_patches;
    const uint32_t m_patch_size;

    bool m_is_input_edge_manifold;
    bool m_is_input_closed;
    bool m_quite;

    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, edge_key_hash>
        m_edges_map;

    // store a copy of face incident vertices along with the neighbor
    // faces of that face
    std::vector<std::vector<uint32_t>> m_fvn;

    // pointer to the patcher class responsible for everything related to
    // patching the mesh into small pieces
    std::unique_ptr<patcher::Patcher> m_patcher;


    //*************** Patch sub-matrices

    //****** Host
    uint32_t m_max_vertices_per_patch, m_max_edges_per_patch,
        m_max_faces_per_patch;
    uint32_t m_max_owned_vertices_per_patch, m_max_owned_edges_per_patch,
        m_max_owned_faces_per_patch;
    //** main incident relations
    std::vector<std::vector<uint16_t>> m_h_patches_edges;
    std::vector<std::vector<uint16_t>> m_h_patches_faces;
    //.x edge address
    //.y edge size
    //.z face address
    //.w face size
    std::vector<uint4> m_h_ad_size;

    // the size of owned mesh elements per patch
    //.x faces
    //.y edges
    //.z vertex
    std::vector<uint4> m_h_owned_size;

    uint2 m_max_size;  // max number of edges(*2) and faces(*face_degree)
                       // in a patch
                       // this counts the size of edges and faces arrays
                       // rounded up to multiple of 32

    //** mappings
    // local to global map for (v)ertices (e)dges and (f)aces
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_v;
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_e;
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_f;

    // storing the start id(x) and element count(y)
    std::vector<uint2> m_h_ad_size_ltog_v, m_h_ad_size_ltog_e,
        m_h_ad_size_ltog_f;


    //****** Device
    // Each device pointer points to a long array that holds specific data
    // separated by patch id
    //       ____________ _____________ ____________
    //      |____________|_____________|____________|
    //           ^^            ^^            ^^
    //      patch 1 data  patch 2 data   patch 3 data

    // We store the starting id and the size of mesh elements for each patch
    // in m_d_ad_size_ltog_MESHELE (ad for address) where MESHELE could be
    // v,e, or f. This is for the mapping pointers
    // For incidence pointers, we only need store the starting id

    //** face/vertex/edge patch (indexed by in global space)
    uint32_t *m_d_face_patch, *m_d_vertex_patch, *m_d_edge_patch;

    //** mapping
    uint32_t *m_d_patches_ltog_v, *m_d_patches_ltog_e, *m_d_patches_ltog_f;
    uint2 *   m_d_ad_size_ltog_v, *m_d_ad_size_ltog_e, *m_d_ad_size_ltog_f;

    //** incidence
    uint16_t *m_d_patches_edges, *m_d_patches_faces;

    //*** Scanned histogram of the number of mesh elements per patch
    std::vector<uint32_t> m_h_patch_distribution_v, m_h_patch_distribution_e,
        m_h_patch_distribution_f;
    uint32_t *m_d_patch_distribution_v, *m_d_patch_distribution_e,
        *m_d_patch_distribution_f;

    //.x edge address
    //.y edge size
    //.z face address
    //.w face size
    uint4* m_d_ad_size;

    // the size of owned mesh elements per patch
    //.x faces
    //.y edges
    //.z vertex
    uint4* m_d_owned_size;

    // neighbour patches
    uint32_t *m_d_neighbour_patches, *m_d_neighbour_patches_offset;

    double m_total_gpu_storage_mb;
};
}  // namespace rxmesh
