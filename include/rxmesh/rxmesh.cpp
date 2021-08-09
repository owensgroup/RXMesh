#include "rxmesh.h"

#include <assert.h>
#include <exception>
#include <memory>
#include <queue>
#include "patcher/patcher.h"
#include "rxmesh/rxmesh_context.h"
#include "rxmesh/util/export_tools.h"
#include "rxmesh/util/math.h"

namespace RXMESH {
// extern std::vector<std::vector<RXMESH::float>> Verts; // TODO remove this

//********************** Constructors/Destructors
template <uint32_t patchSize>
RXMesh<patchSize>::RXMesh(std::vector<std::vector<uint32_t>>& fv,
                          std::vector<std::vector<coordT>>&   coordinates,
                          const bool                          sort /*= false*/,
                          const bool                          quite /*= true*/)
    : m_num_edges(0), m_num_faces(0), m_num_vertices(0), m_max_ele_count(0),
      m_max_valence(0), m_max_valence_vertex_id(INVALID32), 
      m_max_edge_incident_faces(0), m_max_face_adjacent_faces(0),
      m_face_degree(3), m_num_patches(0), m_is_input_edge_manifold(true),
      m_is_input_closed(true), m_is_sort(sort), m_quite(quite),
      m_max_vertices_per_patch(0), m_max_edges_per_patch(0),
      m_max_faces_per_patch(0), m_d_patches_ltog_v(nullptr),
      m_d_patches_ltog_e(nullptr), m_d_patches_ltog_f(nullptr),
      m_d_ad_size_ltog_v(nullptr), m_d_ad_size_ltog_e(nullptr),
      m_d_ad_size_ltog_f(nullptr), m_d_patches_edges(nullptr),
      m_d_patches_faces(nullptr), m_d_patch_distribution_v(nullptr),
      m_d_patch_distribution_e(nullptr), m_d_patch_distribution_f(nullptr),
      m_d_ad_size(nullptr), m_d_patches_face_ribbon_flag(nullptr)
{
    // Build everything from scratch including patches
    build_local(fv, coordinates);
    device_alloc_local();
}

template <uint32_t patchSize>
RXMesh<patchSize>::~RXMesh()
{
    GPU_FREE(m_d_patches_ltog_v);
    GPU_FREE(m_d_patches_ltog_e);
    GPU_FREE(m_d_patches_ltog_f);
    GPU_FREE(m_d_patches_edges);
    GPU_FREE(m_d_patches_faces);
    GPU_FREE(m_d_patches_face_ribbon_flag);
    GPU_FREE(m_d_ad_size_ltog_v);
    GPU_FREE(m_d_ad_size_ltog_e);
    GPU_FREE(m_d_ad_size_ltog_f);
    GPU_FREE(m_d_ad_size);
    GPU_FREE(m_d_patch_distribution_v);
    GPU_FREE(m_d_patch_distribution_e);
    GPU_FREE(m_d_patch_distribution_f);
    GPU_FREE(m_d_vertex_patch);
    GPU_FREE(m_d_edge_patch);
    GPU_FREE(m_d_face_patch);
};
//**************************************************************************


//********************** Builders
template <uint32_t patchSize>
void RXMesh<patchSize>::build_local(
    std::vector<std::vector<uint32_t>>& fv,
    std::vector<std::vector<coordT>>&   coordinates)
{
    // we build everything here from scratch
    // 1) set num vertices
    // 2) populate edge_map
    // 3) for each edge, store a list of faces that are incident to that edge
    // 4) copy fv to m_fvn and append the adjacent faces for each face using
    // info from 3)
    // 5) patch the mesh
    // 6) populate the local mesh

    //=========== 1)
    m_num_faces = static_cast<uint32_t>(fv.size());
    set_num_vertices(fv);
    //===============================


    //=========== 2)
    populate_edge_map(fv);
    m_num_edges = static_cast<uint32_t>(m_edges_map.size());
    //===============================


    //=========== 3)
    std::vector<std::vector<uint32_t>> ef;
    edge_incident_faces(fv, ef);
    // caching mesh type; edge manifold, closed
    for (uint32_t e = 0; e < ef.size(); ++e) {
        if (ef[e].size() < 2) {
            m_is_input_closed = false;
        }
        if (ef[e].size() > 2) {
            m_is_input_edge_manifold = false;
        }
    }
    //===============================


    //=========== 4)
    // copy fv
    std::vector<std::vector<uint32_t>> rep(fv);
    rep.swap(m_fvn);
    // extend m_fvn by adding the face neighbors
    for (uint32_t e = 0; e < ef.size(); ++e) {
        assert(ef[e].size() != 0);  // we don't handle dangling edges

        for (uint32_t f = 0; f < ef[e].size(); ++f) {
            uint32_t f0 = ef[e][f];
            for (uint32_t s = f + 1; s < ef[e].size(); ++s) {
                uint32_t f1 = ef[e][s];
                m_fvn[f0].push_back(f1);
                m_fvn[f1].push_back(f0);
            }
        }
    }
    //===============================


    //=========== 5)
    // create an instance of Patcher and execute it and then move the
    // ownership to m_patcher
    std::unique_ptr<PATCHER::Patcher> pp = std::make_unique<PATCHER::Patcher>(
        patchSize, m_fvn, m_num_vertices, m_num_edges, true, m_quite);
    pp->execute(
        [this](uint32_t v0, uint32_t v1) { return this->get_edge_id(v0, v1); },
        ef);

    m_patcher = std::move(pp);
    m_num_patches = m_patcher->get_num_patches();
    // m_patcher->export_patches(Verts);
    //===============================

    //=========== 5.5)
    // sort indices based on patches
    if (m_is_sort) {
        sort(fv, coordinates);
    }
    //===============================

    //=========== 6)
    m_max_size.x = m_max_size.y = 0;
    m_h_owned_size.resize(m_num_patches);
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        build_patch_locally(p);
        m_max_size.x = static_cast<unsigned int>(
            std::max(size_t(m_max_size.x), m_h_patches_edges[p].size()));
        m_max_size.y = static_cast<unsigned int>(
            std::max(size_t(m_max_size.y), m_h_patches_faces[p].size()));
    }

    m_max_size.x = round_up_multiple(m_max_size.x, 32u);
    m_max_size.y = round_up_multiple(m_max_size.y, 32u);

    m_max_vertices_per_patch = 0;
    m_max_edges_per_patch = 0;
    m_max_faces_per_patch = 0;
    m_max_owned_vertices_per_patch = 0;
    m_max_owned_edges_per_patch = 0;
    m_max_owned_faces_per_patch = 0;
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        m_max_vertices_per_patch = std::max(
            m_max_vertices_per_patch, uint32_t(m_h_patches_ltog_v[p].size()));
        m_max_edges_per_patch = std::max(
            m_max_edges_per_patch, uint32_t(m_h_patches_ltog_e[p].size()));
        m_max_faces_per_patch = std::max(
            m_max_faces_per_patch, uint32_t(m_h_patches_ltog_f[p].size()));

        m_max_owned_faces_per_patch =
            std::max(m_max_owned_faces_per_patch, m_h_owned_size[p].x);
        m_max_owned_edges_per_patch =
            std::max(m_max_owned_edges_per_patch, m_h_owned_size[p].y);
        m_max_owned_vertices_per_patch =
            std::max(m_max_owned_vertices_per_patch, m_h_owned_size[p].z);
    }

    // scanned histogram of element count in patches
    m_h_patch_distribution_v.resize(m_num_patches + 1, 0);
    m_h_patch_distribution_e.resize(m_num_patches + 1, 0);
    m_h_patch_distribution_f.resize(m_num_patches + 1, 0);

    for (uint32_t v = 0; v < m_num_vertices; ++v) {
        uint32_t patch = m_patcher->get_vertex_patch_id(v);
        if (patch != INVALID32) {
            m_h_patch_distribution_v[patch]++;
        }
    }
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        uint32_t patch = m_patcher->get_face_patch_id(f);
        if (patch != INVALID32) {
            m_h_patch_distribution_f[patch]++;
        }
    }
    for (uint32_t e = 0; e < m_num_edges; ++e) {
        uint32_t patch = m_patcher->get_edge_patch_id(e);
        if (patch != INVALID32) {
            m_h_patch_distribution_e[patch]++;
        }
    }
    auto ex_scan = [](std::vector<uint32_t>& vv) {
        uint32_t dd = 0;
        for (uint32_t i = 1; i < vv.size(); ++i) {
            uint32_t temp = vv[i];
            vv[i] = dd + vv[i - 1];
            dd = temp;
        }
        vv[0] = 0;
    };

#ifdef NDEBUG
    {
        uint32_t edges(0), faces(0), vertices(0);
        for (uint32_t p = 0; p < m_num_patches; ++p) {
            edges += m_h_patch_distribution_e[p];
            faces += m_h_patch_distribution_f[p];
            vertices += m_h_patch_distribution_v[p];
        }

        assert(edges == m_num_edges);
        assert(faces == m_num_faces);
        assert(vertices == m_num_vertices);
    }
#endif  // NDEBUG
    ex_scan(m_h_patch_distribution_v);
    ex_scan(m_h_patch_distribution_e);
    ex_scan(m_h_patch_distribution_f);

    if (!m_quite) {
        RXMESH_TRACE("#Vertices = {}, #Faces= {}, #Edges= {}", m_num_vertices,
                     m_num_faces, m_num_edges);
        RXMESH_TRACE("Input is {} edge manifold",
                     ((m_is_input_edge_manifold) ? "" : " Not"));
        RXMESH_TRACE("Input is {} closed", ((m_is_input_closed) ? "" : " Not"));
        RXMESH_TRACE("max valence = {}", m_max_valence);
        RXMESH_TRACE("max edge incident faces = {}", m_max_edge_incident_faces);
        RXMESH_TRACE("max face adjacent faces = {}", m_max_face_adjacent_faces);
        RXMESH_TRACE("per-patch maximum edges references= {}", m_max_size.x);
        RXMESH_TRACE("per-patch maximum  faces references= {}", m_max_size.y);
        RXMESH_TRACE("per-patch maximum face count (owned)= {} ({})",
                     m_max_faces_per_patch, m_max_owned_faces_per_patch);
        RXMESH_TRACE("per-patch maximum edge count (owned) = {} ({})",
                     m_max_edges_per_patch, m_max_owned_edges_per_patch);
        RXMESH_TRACE("per-patch maximum vertex count (owned)= {} ({})",
                     m_max_vertices_per_patch, m_max_owned_vertices_per_patch);
    }
    //===============================

    m_max_ele_count = std::max(m_num_edges, m_num_faces);
    m_max_ele_count = std::max(m_num_vertices, m_max_ele_count);
}

template <uint32_t patchSize>
void RXMesh<patchSize>::build_patch_locally(const uint32_t patch_id)
{
    // Build the patch in local index space
    // This is the two small matrices defining incident relation between
    // edge-vertices and faces-edges along with the mapping from local to
    // global space for vertices, edge, and faces

    // We we create a new patch, we make sure that the elements owned by the
    // patch will have local indices lower than any elements (of the same type)
    // that is not owned by the patch
    const uint32_t *p_val(m_patcher->get_patches_val()),
        *p_off(m_patcher->get_patches_offset());


    // patch start and end
    const uint32_t p_start = (patch_id == 0) ? 0 : p_off[patch_id - 1];
    const uint32_t p_end = p_off[patch_id];
    const uint32_t r_start =
        (patch_id == 0) ? 0 :
                          m_patcher->get_external_ribbon_offset()[patch_id - 1];
    const uint32_t r_end = m_patcher->get_external_ribbon_offset()[patch_id];

    const uint32_t total_patch_num_faces =
        (p_end - p_start) + (r_end - r_start);
    uint16_t total_patch_num_edges(0), total_patch_num_vertices(0);

    assert(total_patch_num_faces <= m_num_faces);

    //** faces
    // container for this patch local faces i.e., face incident edges
    std::vector<uint16_t> fp(m_face_degree * total_patch_num_faces);
    // std::vector<flag_t>   fr_flag(total_patch_num_faces, 0);
    // the mapping from this patch local space (uint16_t) to global one
    std::vector<uint32_t> f_ltog(total_patch_num_faces);

    //** edges
    // container for this patch local edges i.e., edge incident vertices
    std::vector<uint16_t> ep;

    // the mapping from this patch local space to global one
    std::vector<uint32_t> e_ltog;

    //** vertices
    // the mapping from this patch local space to global one
    std::vector<uint32_t> v_ltog;

    // count the number of elements owned and not owned by the patch
    uint16_t              num_edges_owned(0), num_vertices_owned(0);
    std::vector<uint32_t> tmp_e, tmp_v;
    tmp_e.reserve(patchSize * 3);
    tmp_v.reserve(patchSize);
    auto insert_if_not_found = [](uint32_t               index,
                                  std::vector<uint32_t>& tmp) -> uint32_t {
        for (uint32_t i = 0; i < tmp.size(); ++i) {
            if (tmp[i] == index) {
                return INVALID32;
            }
        }
        tmp.push_back(index);
        return static_cast<uint32_t>(tmp.size() - 1);
    };
    auto count_num_elements = [&](uint32_t global_f) {
        for (uint32_t j = 0; j < 3; j++) {
            // find the edge global id
            uint32_t global_v0 = m_fvn[global_f][j];
            uint32_t global_v1 = m_fvn[global_f][(j + 1) % 3];

            // find the edge in m_edge_map with v0,v1
            std::pair<uint32_t, uint32_t> my_edge =
                edge_key(global_v0, global_v1);
            uint32_t global_e = get_edge_id(my_edge);

            uint32_t v_index = insert_if_not_found(global_v0, tmp_v);
            if (v_index != INVALID32) {
                total_patch_num_vertices++;
                if (m_patcher->get_vertex_patch_id(global_v0) == patch_id) {
                    num_vertices_owned++;
                }
            }

            uint32_t e_index = insert_if_not_found(global_e, tmp_e);
            if (e_index != INVALID32) {
                total_patch_num_edges++;
                if (m_patcher->get_edge_patch_id(global_e) == patch_id) {
                    num_edges_owned++;
                }
            }
        }
    };
    for (uint32_t s = p_start; s < p_end; ++s) {
        uint32_t global_f = p_val[s];
        count_num_elements(global_f);
    }
    for (uint32_t s = r_start; s < r_end; ++s) {
        uint32_t global_f = m_patcher->get_external_ribbon_val()[s];
        count_num_elements(global_f);
    }

    // 1) loop over patch faces
    e_ltog.resize(total_patch_num_edges);
    v_ltog.resize(total_patch_num_vertices);
    ep.resize(total_patch_num_edges * 2);

    // to track how many faces/edges/vertices we have locally created so far
    uint16_t faces_count(0), edges_owned_count(0), edges_not_owned_count(0),
        vertices_owned_count(0), vertices_not_owned_count(0);
    for (uint32_t s = p_start; s < p_end; ++s) {
        uint32_t global_f = p_val[s];
        create_new_local_face(patch_id, global_f, m_fvn[global_f], faces_count,
                              edges_owned_count, edges_not_owned_count,
                              vertices_owned_count, vertices_not_owned_count,
                              num_edges_owned, num_vertices_owned, f_ltog,
                              e_ltog, v_ltog, fp, ep);
        // fr_flag[local_f] = (p_flag[s]) ? 1 : 0;
    }


    // 2) loop over ribbon faces
    for (uint32_t s = r_start; s < r_end; ++s) {
        uint32_t global_f = m_patcher->get_external_ribbon_val()[s];
        create_new_local_face(patch_id, global_f, m_fvn[global_f], faces_count,
                              edges_owned_count, edges_not_owned_count,
                              vertices_owned_count, vertices_not_owned_count,
                              num_edges_owned, num_vertices_owned, f_ltog,
                              e_ltog, v_ltog, fp, ep);
        // fr_flag[local_f] = 2;
    }

    if (vertices_owned_count != num_vertices_owned ||
        edges_owned_count != num_edges_owned ||
        edges_owned_count + edges_not_owned_count != total_patch_num_edges ||
        vertices_owned_count + vertices_not_owned_count !=
            total_patch_num_vertices) {
        RXMESH_ERROR("RXMesh::build_patch_locally() patch is " +
                     std::to_string(patch_id) + " not built correctly!!");
    }


    m_h_owned_size[patch_id].x = (p_end - p_start);
    m_h_owned_size[patch_id].y = num_edges_owned;
    m_h_owned_size[patch_id].z = num_vertices_owned;

    // faces
    m_h_patches_faces.push_back(fp);
    m_h_patches_ltog_f.push_back(f_ltog);
    // m_h_patches_face_ribbon_flag.push_back(fr_flag);

    // edges
    m_h_patches_edges.push_back(ep);
    m_h_patches_ltog_e.push_back(e_ltog);

    // vertices
    m_h_patches_ltog_v.push_back(v_ltog);
}

template <uint32_t patchSize>
uint16_t RXMesh<patchSize>::create_new_local_face(
    const uint32_t               patch_id,
    const uint32_t               global_f,
    const std::vector<uint32_t>& fv,
    uint16_t&                    faces_count,
    uint16_t&                    edges_owned_count,
    uint16_t&                    edges_not_owned_count,
    uint16_t&                    vertices_owned_count,
    uint16_t&                    vertices_not_owned_count,
    const uint16_t               num_edges_owned,
    const uint16_t               num_vertices_owned,
    std::vector<uint32_t>&       f_ltog,
    std::vector<uint32_t>&       e_ltog,
    std::vector<uint32_t>&       v_ltog,
    std::vector<uint16_t>&       fp,
    std::vector<uint16_t>&       ep)
{

    uint16_t local_f = faces_count++;
    f_ltog[local_f] = global_f;

    // shift to left and set first bit to 1 if global_f's patch is this patch
    f_ltog[local_f] = f_ltog[local_f] << 1;
    f_ltog[local_f] =
        f_ltog[local_f] | (m_patcher->get_face_patch_id(global_f) == patch_id);

    auto find_increment_index =
        [&patch_id](const uint32_t& global, std::vector<uint32_t>& vect,
                    uint16_t& owned_count, uint16_t& not_owned_count,
                    const uint16_t num_owned, bool& incremented,
                    const uint32_t ele_patch) -> uint16_t {
        incremented = true;

        for (uint16_t id = 0; id < owned_count; ++id) {
            if (global == (vect[id] >> 1)) {
                incremented = false;
                return id;
            }
        }

        for (uint16_t id = num_owned; id < num_owned + not_owned_count; ++id) {
            if (global == (vect[id] >> 1)) {
                incremented = false;
                return id;
            }
        }
        uint32_t to_store = (global << 1);
        uint16_t ret_id;
        if (ele_patch == patch_id) {
            to_store = to_store | 1;
            ret_id = owned_count++;
        } else {
            ret_id = num_owned + (not_owned_count++);
        }
        vect[ret_id] = to_store;
        return ret_id;
    };

    for (uint32_t j = 0; j < m_face_degree; j++) {

        // find the edge global id
        uint32_t global_v0 = fv[j];
        uint32_t global_v1 = fv[(j + 1) % m_face_degree];

        // find the edge in m_edge_map with v0,v1
        std::pair<uint32_t, uint32_t> my_edge = edge_key(global_v0, global_v1);

        assert(my_edge.first == global_v0 || my_edge.first == global_v1);
        assert(my_edge.second == global_v0 || my_edge.second == global_v1);

        int dir = 1;
        if (my_edge.first == global_v0 && my_edge.second == global_v1) {
            dir = 0;
        }

        uint32_t global_e = get_edge_id(my_edge);

        // convert edge to local index by searching for it. if not
        // found, then increment the number of local edges
        bool     new_e(false);
        uint16_t local_e = find_increment_index(
            global_e, e_ltog, edges_owned_count, edges_not_owned_count,
            num_edges_owned, new_e, m_patcher->get_edge_patch_id(global_e));

        if (new_e) {
            // if it is new edges, then we need to either look for
            // its vertices. if there were inserted before in the
            // patch, then retrieve their local id. otherwise, we
            // new vertices to the patch
            assert(my_edge.first != my_edge.second);

            bool     new_v(false);
            uint16_t local_v0 = find_increment_index(
                my_edge.first, v_ltog, vertices_owned_count,
                vertices_not_owned_count, num_vertices_owned, new_v,
                m_patcher->get_vertex_patch_id(my_edge.first));

            uint16_t local_v1 = find_increment_index(
                my_edge.second, v_ltog, vertices_owned_count,
                vertices_not_owned_count, num_vertices_owned, new_v,
                m_patcher->get_vertex_patch_id(my_edge.second));

            assert(local_v0 != local_v1);

            // new edges are appended in the end of e_ltog
            // and so as their vertices in ep
            ep[2 * local_e] = local_v0;
            ep[2 * local_e + 1] = local_v1;
        }
        // shift local_e to left
        // set the first bit to 1 if (dir ==1)
        local_e = local_e << 1;
        local_e = local_e | (dir & 1);
        fp[local_f * m_face_degree + j] = local_e;
    }

    return local_f;
}

template <uint32_t patchSize>
void RXMesh<patchSize>::set_num_vertices(
    const std::vector<std::vector<uint32_t>>& fv)
{
    m_num_vertices = 0;
    for (uint32_t i = 0; i < fv.size(); ++i) {
        if (fv[i].size() != 3) {
            RXMESH_ERROR("RXMesh::count_vertices() Face" + std::to_string(i) +
                         " is not triangles. Non-triangular faces are not "
                         "supported yet");
        }
        for (uint32_t j = 0; j < fv[i].size(); ++j) {
            m_num_vertices = std::max(m_num_vertices, fv[i][j]);
        }
    }
    ++m_num_vertices;
}


template <uint32_t patchSize>
void RXMesh<patchSize>::populate_edge_map(
    const std::vector<std::vector<uint32_t>>& fv)
{

    // create edges and populate edge_map
    // and also compute max valence

    m_edges_map.reserve(m_num_faces * 3);  // upper bound

    std::vector<uint32_t> vv_count(m_num_vertices, 0);
    m_max_valence = 0;

    for (uint32_t f = 0; f < m_num_faces; ++f) {

        if (fv[f].size() < 3) {
            RXMESH_ERROR(
                "RXMesh::populate_edge_map() Face {} has less than three "
                "vertices",
                f);
        }
        for (uint32_t j = 0; j < fv[f].size(); ++j) {

            uint32_t v0 = fv[f][j];
            uint32_t v1 = (j != fv[f].size() - 1) ? fv[f][j + 1] : fv[f][0];

            std::pair<uint32_t, uint32_t> my_edge = edge_key(v0, v1);

            typename std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t,
                                        edge_key_hash>::const_iterator e_it =
                m_edges_map.find(my_edge);

            if (e_it == m_edges_map.end()) {
                m_edges_map.insert(std::make_pair(my_edge, m_num_edges++));

                vv_count[v0]++;
                vv_count[v1]++;

                // also set max valence
                if (m_max_valence < vv_count[v0]) {
                    m_max_valence = vv_count[v0];
                    m_max_valence_vertex_id = v0;
                }
                if (m_max_valence < vv_count[v1]) {
                    m_max_valence = vv_count[v1];
                    m_max_valence_vertex_id = v1;
                }
            }
        }
    }
}

template <uint32_t patchSize>
void RXMesh<patchSize>::edge_incident_faces(
    const std::vector<std::vector<uint32_t>>& fv,
    std::vector<std::vector<uint32_t>>&       ef)
{
    // populate ef by the faces incident to each edge
    // must call populate_edge_map before call it

    assert(m_edges_map.size() > 0);

    uint32_t num_edges = static_cast<uint32_t>(m_edges_map.size());

    // reserve space assuming mesh is mostly manifold (edge is shared by
    // two faces)
    ef.clear();
    ef.resize(num_edges, std::vector<uint32_t>(0));
    for (uint32_t e = 0; e < num_edges; ++e) {
        ef[e].reserve(2);
    }

    m_max_edge_incident_faces = 0;
    for (uint32_t f = 0; f < m_num_faces; ++f) {

        for (uint32_t j = 0; j < fv[f].size(); ++j) {

            uint32_t v0 = fv[f][j];
            uint32_t v1 = (j != fv[f].size() - 1) ? fv[f][j + 1] : fv[f][0];

            uint32_t edge_num = get_edge_id(v0, v1);
            ef[edge_num].push_back(f);
            m_max_edge_incident_faces = std::max(m_max_edge_incident_faces,
                                                 uint32_t(ef[edge_num].size()));
        }
    }

    // calc m_max_face_adjacent_faces
    m_max_face_adjacent_faces = 0;
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        uint32_t ff_count = 0;
        for (uint32_t j = 0; j < fv[f].size(); ++j) {
            uint32_t v0 = fv[f][j];
            uint32_t v1 = (j != fv[f].size() - 1) ? fv[f][j + 1] : fv[f][0];
            uint32_t edge_num = get_edge_id(v0, v1);
            ff_count += ef[edge_num].size() - 1;
        }
        m_max_face_adjacent_faces =
            std::max(ff_count, m_max_face_adjacent_faces);
    }
}

template <uint32_t patchSize>
uint32_t RXMesh<patchSize>::get_edge_id(const uint32_t v0,
                                        const uint32_t v1) const
{
    // v0 and v1 are two vertices in global space. we return the edge
    // id in global space also (by querying m_edges_map)
    assert(m_edges_map.size() != 0);

    std::pair<uint32_t, uint32_t> edge = edge_key(v0, v1);

    assert(edge.first == v0 || edge.first == v1);
    assert(edge.second == v0 || edge.second == v1);

    return get_edge_id(edge);
}

template <uint32_t patchSize>
uint32_t RXMesh<patchSize>::get_edge_id(
    const std::pair<uint32_t, uint32_t>& edge) const
{
    uint32_t edge_id = -1;
    try {
        edge_id = m_edges_map.at(edge);
    } catch (const std::out_of_range&) {
        RXMESH_ERROR(
            "RXMesh::get_edge_id() mapping edges went wrong."
            " Can not find an edge connecting vertices {} and {}",
            edge.first, edge.second);
    }

    return edge_id;
}
//**************************************************************************

//********************** sort
template <uint32_t patchSize>
void RXMesh<patchSize>::sort(std::vector<std::vector<uint32_t>>& fv,
                             std::vector<std::vector<coordT>>&   coordinates)
{
    if (m_num_patches == 1) {
        return;
    }
    std::vector<uint32_t> new_face_id(m_num_faces, INVALID32);
    std::vector<uint32_t> new_vertex_id(m_num_vertices, INVALID32);
    std::vector<uint32_t> new_edge_id(m_num_edges, INVALID32);

    const uint32_t* patches_offset = m_patcher->get_patches_offset();
    const uint32_t* patches_val = m_patcher->get_patches_val();

    // patch status:
    // 1) 0: has not been processed/seen before
    // 2) 1: currently in the queue
    // 3) 2: has been processed (assigned new id)
    std::vector<uint32_t> patch_status(m_num_patches, 0);

    std::queue<uint32_t> patch_queue;
    patch_queue.push(0);
    uint32_t face_counter = 0;
    uint32_t vertex_counter = 0;
    uint32_t edge_counter = 0;

    //*****Compute new ID for faces, edges, and vertices
    while (true) {

        std::queue<uint32_t> patch_queue;

        for (uint32_t p = 0; p < m_num_patches; ++p) {
            if (patch_status[p] == 0) {
                patch_queue.push(p);
                patch_status[p] = 1;
                break;
            }
        }

        // this happens when all patches has been processed
        if (patch_queue.empty()) {
            break;
        }


        while (patch_queue.size() > 0) {
            uint32_t p = patch_queue.front();
            patch_queue.pop();
            patch_status[p] = 2;

            uint32_t p_start = (p == 0) ? 0 : patches_offset[p - 1];
            uint32_t p_end = patches_offset[p];
            // first loop over p's faces and assigned its faces new id
            for (uint32_t f = p_start; f < p_end; ++f) {
                uint32_t face = patches_val[f];
                new_face_id[face] = face_counter++;

                // assign face's vertices new id
                for (uint32_t v = 0; v < 3; ++v) {
                    uint32_t vertex = m_fvn[face][v];
                    // if the vertex is owned by this patch
                    if (m_patcher->get_vertex_patch_id(vertex) == p &&
                        new_vertex_id[vertex] == INVALID32) {
                        new_vertex_id[vertex] = vertex_counter++;
                    }
                }


                // assign face's edge new id
                uint32_t v1 = 2;
                for (uint32_t v0 = 0; v0 < 3; ++v0) {
                    uint32_t vertex0 = m_fvn[face][v0];
                    uint32_t vertex1 = m_fvn[face][v1];
                    uint32_t edge = get_edge_id(vertex0, vertex1);

                    // if the edge is owned by this patch
                    if (m_patcher->get_edge_patch_id(edge) == p &&
                        new_edge_id[edge] == INVALID32) {
                        new_edge_id[edge] = edge_counter++;
                    }
                    v1 = v0;
                }
            }

            // second loop over p's ribbon and push new patches into the queue
            // only if there are not in the queue and the have not been
            // processed yet.
            uint32_t ribbon_start =
                (p == 0) ? 0 : m_patcher->get_external_ribbon_offset()[p - 1];
            uint32_t ribbon_end = m_patcher->get_external_ribbon_offset()[p];
            for (uint32_t f = ribbon_start; f < ribbon_end; ++f) {
                // this is a face in the ribbon
                uint32_t face = m_patcher->get_external_ribbon_val()[f];
                // get the face actual patch
                uint32_t face_patch = m_patcher->get_face_patch_id(face);
                assert(face_patch != p);
                if (patch_status[face_patch] == 0) {
                    patch_queue.push(face_patch);
                    patch_status[face_patch] = 1;
                }
            }
        }
    }
    if (edge_counter != m_num_edges || vertex_counter != m_num_vertices ||
        face_counter != m_num_faces) {
        RXMESH_ERROR("RXMesh::sort Error in assigning new IDs");
    }
    //**** Apply changes
    m_max_valence_vertex_id = new_vertex_id[m_max_valence_vertex_id];
    // coordinates
    {
        std::vector<std::vector<coordT>> coord_ordered(coordinates);
        for (uint32_t v = 0; v < m_num_vertices; ++v) {
            uint32_t new_v_id = new_vertex_id[v];
            coord_ordered[new_v_id][0] = coordinates[v][0];
            coord_ordered[new_v_id][1] = coordinates[v][1];
            coord_ordered[new_v_id][2] = coordinates[v][2];
        }
        coordinates.swap(coord_ordered);
    }

    // edge map
    {
        std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t,
                           edge_key_hash>
            edges_map;
        edges_map.reserve(m_num_faces * 3);
        for (auto& it : m_edges_map) {
            uint32_t v0 = new_vertex_id[it.first.first];
            uint32_t v1 = new_vertex_id[it.first.second];
            uint32_t edge_id = new_edge_id[it.second];

            std::pair<uint32_t, uint32_t> my_edge = edge_key(v0, v1);

            typename std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t,
                                        edge_key_hash>::const_iterator e_it =
                edges_map.find(my_edge);

            if (e_it == edges_map.end()) {
                edges_map.insert(std::make_pair(my_edge, edge_id));
            } else {
                RXMESH_ERROR("RXMesh::sort Unknown error");
            }
        }
        m_edges_map.swap(edges_map);
    }

    // m_fvn
    {
        std::vector<std::vector<uint32_t>> fvn(m_fvn);
        for (uint32_t f = 0; f < m_fvn.size(); ++f) {
            uint32_t new_f_id = new_face_id[f];
            fvn[new_f_id].resize(3);
            // v
            fvn[new_f_id][0] = new_vertex_id[m_fvn[f][0]];
            fvn[new_f_id][1] = new_vertex_id[m_fvn[f][1]];
            fvn[new_f_id][2] = new_vertex_id[m_fvn[f][2]];

            fv[new_f_id][0] = fvn[new_f_id][0];
            fv[new_f_id][1] = fvn[new_f_id][1];
            fv[new_f_id][2] = fvn[new_f_id][2];

            // n
            for (uint32_t n = 3; n < m_fvn[f].size(); ++n) {
                fvn[new_f_id].push_back(new_face_id[m_fvn[f][n]]);
            }
        }
        m_fvn.swap(fvn);
    }

    // patcher
    {
        uint32_t* patch_val = m_patcher->get_patches_val();
        for (uint32_t i = 0; i < m_num_faces; ++i) {
            patch_val[i] = new_face_id[patch_val[i]];
        }

        uint32_t num_ext_ribbon_faces =
            m_patcher->get_external_ribbon_offset()[m_num_patches - 1];
        for (uint32_t i = 0; i < num_ext_ribbon_faces; ++i) {
            m_patcher->get_external_ribbon_val()[i] =
                new_face_id[m_patcher->get_external_ribbon_val()[i]];
        }

        {
            std::vector<uint32_t> face_patch(m_num_faces);
            for (uint32_t f = 0; f < m_num_faces; ++f) {
                uint32_t new_f_id = new_face_id[f];
                face_patch[new_f_id] = m_patcher->get_face_patch_id(f);
            }
            std::memcpy(m_patcher->get_face_patch().data(), face_patch.data(),
                        m_num_faces * sizeof(uint32_t));
        }

        {

            std::vector<uint32_t> vertex_patch(m_num_vertices);
            for (uint32_t v = 0; v < m_num_vertices; ++v) {
                uint32_t new_v_id = new_vertex_id[v];
                vertex_patch[new_v_id] = m_patcher->get_vertex_patch_id(v);
            }
            std::memcpy(m_patcher->get_vertex_patch().data(),
                        vertex_patch.data(), m_num_vertices * sizeof(uint32_t));
        }

        {
            std::vector<uint32_t> edge_patch(m_num_edges);
            for (uint32_t e = 0; e < m_num_edges; ++e) {
                uint32_t new_e_id = new_edge_id[e];
                edge_patch[new_e_id] = m_patcher->get_edge_patch_id(e);
            }
            std::memcpy(m_patcher->get_edge_patch().data(), edge_patch.data(),
                        m_num_edges * sizeof(uint32_t));
        }
    }

    /*m_patcher->export_patches(coordinates);

    std::vector<uint32_t> vert_id(m_num_vertices);
    std::vector<uint32_t> face_id(m_num_faces);
    fill_with_sequential_numbers(vert_id.data(), vert_id.size());
    fill_with_sequential_numbers(face_id.data(), face_id.size());
    export_attribute_VTK("sort_faces.vtk", m_fvn, coordinates,
                         true, face_id.data(), vert_id.data(), false);
    export_attribute_VTK("sort_vertices.vtk", m_fvn, coordinates,
                         false, face_id.data(), vert_id.data(), false);*/
}
//**************************************************************************

//********************** Move to Device
template <uint32_t patchSize>
template <typename Tin, typename Tst>
void RXMesh<patchSize>::get_starting_ids(
    const std::vector<std::vector<Tin>>& input,
    std::vector<Tst>&                    starting_id)
{
    // get the starting ids for the mesh elements in input and store it
    // in the first (x) component of starting_id

    // uint32_t prv = 0;
    assert(starting_id.size() > 0);
    assert(starting_id.size() > input.size());
    starting_id[0].x = 0;
    for (uint32_t p = 1; p <= input.size(); ++p) {
        starting_id[p].x = starting_id[p - 1].x + input[p - 1].size();
        // starting_id[p].x = input[p].size() + prv;
        // prv = starting_id[p].x;
    }
}

template <uint32_t patchSize>
template <typename Tin, typename Tad>
void RXMesh<patchSize>::get_size(const std::vector<std::vector<Tin>>& input,
                                 std::vector<Tad>&                    ad)
{
    // get the size of each element of input and store it as the second(y)
    // component in ad
    assert(ad.size() >= input.size());

    for (uint32_t p = 0; p < input.size(); ++p) {
        ad[p].y = input[p].size();
    }
}

template <uint32_t patchSize>
template <typename T>
void RXMesh<patchSize>::padding_to_multiple(std::vector<std::vector<T>>& input,
                                            const uint32_t multiple,
                                            const T        init_val)
{
    // resize each element on input to be mulitple of multiple by add
    // init_val to the end

    for (uint32_t p = 0; p < input.size(); ++p) {
        const uint32_t new_size =
            round_up_multiple(uint32_t(input[p].size()), multiple);
        assert(new_size >= input[p].size());
        input[p].resize(new_size, static_cast<T>(init_val));
    }
}

template <uint32_t patchSize>
void RXMesh<patchSize>::device_alloc_local()
{

    // allocate and transfer patch information to device
    // make sure to build_local first before calling this

    // storing the start id(x) and element count(y)
    m_h_ad_size_ltog_v.resize(m_num_patches + 1);
    m_h_ad_size_ltog_e.resize(m_num_patches + 1);
    m_h_ad_size_ltog_f.resize(m_num_patches + 1);
    m_h_ad_size.resize(m_num_patches + 1);

    // get mesh element count per patch
    get_size(m_h_patches_ltog_v, m_h_ad_size_ltog_v);
    get_size(m_h_patches_ltog_e, m_h_ad_size_ltog_e);
    get_size(m_h_patches_ltog_f, m_h_ad_size_ltog_f);

    // how many edges and faces we have in each patch
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        m_h_ad_size[p].y = m_h_ad_size_ltog_e[p].y * 2;  // edges size
        m_h_ad_size[p].w =
            m_h_ad_size_ltog_f[p].y * m_face_degree;  // faces size
    }


    // increase to multiple so that each vector size is multiple of 32
    // so that when we copy it to the device, read will be coalesced
    padding_to_multiple(m_h_patches_edges, WARPSIZE,
                        static_cast<uint16_t>(INVALID16));
    padding_to_multiple(m_h_patches_faces, WARPSIZE,
                        static_cast<uint16_t>(INVALID16));
    padding_to_multiple(m_h_patches_ltog_v, WARPSIZE,
                        static_cast<uint32_t>(INVALID32));
    padding_to_multiple(m_h_patches_ltog_e, WARPSIZE,
                        static_cast<uint32_t>(INVALID32));
    padding_to_multiple(m_h_patches_ltog_f, WARPSIZE,
                        static_cast<uint32_t>(INVALID32));

    // uint32_t invalid_flag = 0;
    // if (sizeof(flag_t) == 1) {
    //     invalid_flag = INVALID8;
    // } else if (sizeof(flag_t) == 2) {
    //     invalid_flag = INVALID16;
    // } else if (sizeof(flag_t) == 4) {
    //     invalid_flag = INVALID32;
    // }
    // padding_to_multiple(m_h_patches_face_ribbon_flag, WARPSIZE,
    //                    static_cast<flag_t>(invalid_flag));

    // get the starting id of each patch
    std::vector<uint1> h_edges_ad(m_num_patches + 1),
        h_faces_ad(m_num_patches + 1);

    get_starting_ids(m_h_patches_ltog_v, m_h_ad_size_ltog_v);
    get_starting_ids(m_h_patches_ltog_e, m_h_ad_size_ltog_e);
    get_starting_ids(m_h_patches_ltog_f, m_h_ad_size_ltog_f);
    get_starting_ids(m_h_patches_edges, h_edges_ad);
    get_starting_ids(m_h_patches_faces, h_faces_ad);

    // m_h_ad_size[0].x = m_h_ad_size[0].z = 0;
    for (uint32_t p = 0; p <= m_num_patches; ++p) {
        m_h_ad_size[p].x = h_edges_ad[p].x;  // edges address
        m_h_ad_size[p].z = h_faces_ad[p].x;  // faces address
    }


    // alloc mesh data
    CUDA_ERROR(cudaMalloc((void**)&m_d_patches_ltog_v,
                          sizeof(uint32_t) * m_h_ad_size_ltog_v.back().x));
    CUDA_ERROR(cudaMalloc((void**)&m_d_patches_ltog_e,
                          sizeof(uint32_t) * m_h_ad_size_ltog_e.back().x));
    CUDA_ERROR(cudaMalloc((void**)&m_d_patches_ltog_f,
                          sizeof(uint32_t) * m_h_ad_size_ltog_f.back().x));
    CUDA_ERROR(cudaMalloc((void**)&m_d_patches_edges,
                          sizeof(uint16_t) * m_h_ad_size.back().x));
    CUDA_ERROR(cudaMalloc((void**)&m_d_patches_faces,
                          sizeof(uint16_t) * m_h_ad_size.back().z));
    if (!m_quite) {
        uint32_t patch_local_storage =
            sizeof(uint16_t) * (m_h_ad_size.back().x + m_h_ad_size.back().z) +
            sizeof(uint32_t) *
                (m_h_ad_size_ltog_v.back().x + m_h_ad_size_ltog_e.back().x +
                 m_h_ad_size_ltog_f.back().x);
        uint32_t patch_membership_storage =
            (m_num_faces + m_num_edges + m_num_vertices) * sizeof(uint32_t);
        m_total_gpu_storage_mb =
            double(patch_local_storage + patch_membership_storage) /
            double(1024 * 1024);
        RXMESH_TRACE("Total storage = {0:f} Mb", m_total_gpu_storage_mb);
    }

    // alloc flags
    CUDA_ERROR(cudaMalloc((void**)&m_d_patches_face_ribbon_flag,
                          sizeof(flag_t) * h_faces_ad.back().x));

    // alloc ad_size_ltog and edges_/faces_ad
    CUDA_ERROR(cudaMalloc((void**)&m_d_ad_size_ltog_v,
                          sizeof(uint2) * (m_num_patches + 1)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_ad_size_ltog_e,
                          sizeof(uint2) * (m_num_patches + 1)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_ad_size_ltog_f,
                          sizeof(uint2) * (m_num_patches + 1)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_ad_size, sizeof(uint4) * (m_num_patches + 1)));

    CUDA_ERROR(cudaMalloc((void**)&m_d_owned_size,
                          sizeof(uint4) * (m_num_patches + 1)));


    // copy the mesh data for each patch
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        // m_d_ pointer are linear. The host containers are not but we can
        // take advantage of pointer arthematic (w/ word offsetting) to get
        // things work without copyt the host containers in a linear array

        uint32_t start_v = m_h_ad_size_ltog_v[p].x;
        uint32_t start_e = m_h_ad_size_ltog_e[p].x;
        uint32_t start_f = m_h_ad_size_ltog_f[p].x;
        uint32_t start_edges = m_h_ad_size[p].x;
        uint32_t start_faces = m_h_ad_size[p].z;

        // ltog
        CUDA_ERROR(cudaMemcpy(m_d_patches_ltog_v + start_v,
                              m_h_patches_ltog_v[p].data(),
                              m_h_ad_size_ltog_v[p].y * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        CUDA_ERROR(cudaMemcpy(m_d_patches_ltog_e + start_e,
                              m_h_patches_ltog_e[p].data(),
                              m_h_ad_size_ltog_e[p].y * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        CUDA_ERROR(cudaMemcpy(m_d_patches_ltog_f + start_f,
                              m_h_patches_ltog_f[p].data(),
                              m_h_ad_size_ltog_f[p].y * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        // patches
        CUDA_ERROR(cudaMemcpy(m_d_patches_edges + start_edges,
                              m_h_patches_edges[p].data(),
                              m_h_ad_size_ltog_e[p].y * 2 * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));

        CUDA_ERROR(cudaMemcpy(
            m_d_patches_faces + start_faces, m_h_patches_faces[p].data(),
            m_h_ad_size_ltog_f[p].y * m_face_degree * sizeof(uint16_t),
            cudaMemcpyHostToDevice));

        // flags
        /*CUDA_ERROR(cudaMemcpy(m_d_patches_face_ribbon_flag + start_faces,
            m_h_patches_face_ribbon_flag[p].data(), m_h_ad_size_ltog_f[p].y *
            m_face_degree *	sizeof(flag_t), cudaMemcpyHostToDevice));*/
    }


    // copy ad_size
    CUDA_ERROR(cudaMemcpy(m_d_ad_size_ltog_v, m_h_ad_size_ltog_v.data(),
                          sizeof(uint2) * (m_num_patches + 1),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_ad_size_ltog_e, m_h_ad_size_ltog_e.data(),
                          sizeof(uint2) * (m_num_patches + 1),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_ad_size_ltog_f, m_h_ad_size_ltog_f.data(),
                          sizeof(uint2) * (m_num_patches + 1),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_ad_size, m_h_ad_size.data(),
                          sizeof(uint4) * (m_num_patches + 1),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_owned_size, m_h_owned_size.data(),
                          sizeof(uint4) * (m_num_patches),
                          cudaMemcpyHostToDevice));


    // allocate and copy face/vertex/edge patch
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_face_patch, sizeof(uint32_t) * (m_num_faces)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_edge_patch, sizeof(uint32_t) * (m_num_edges)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_vertex_patch,
                          sizeof(uint32_t) * (m_num_vertices)));

    CUDA_ERROR(
        cudaMemcpy(m_d_face_patch, this->m_patcher->get_face_patch().data(),
                   sizeof(uint32_t) * (m_num_faces), cudaMemcpyHostToDevice));
    CUDA_ERROR(
        cudaMemcpy(m_d_edge_patch, this->m_patcher->get_edge_patch().data(),
                   sizeof(uint32_t) * (m_num_edges), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(
        m_d_vertex_patch, this->m_patcher->get_vertex_patch().data(),
        sizeof(uint32_t) * (m_num_vertices), cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMalloc((void**)&m_d_patch_distribution_v,
                          (m_num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_patch_distribution_e,
                          (m_num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_patch_distribution_f,
                          (m_num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(
        m_d_patch_distribution_v, m_h_patch_distribution_v.data(),
        (m_num_patches + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(
        m_d_patch_distribution_e, m_h_patch_distribution_e.data(),
        (m_num_patches + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(
        m_d_patch_distribution_f, m_h_patch_distribution_f.data(),
        (m_num_patches + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate and copy the context to the gpu
    m_rxmesh_context.init(
        m_num_edges, m_num_faces, m_num_vertices, m_face_degree, m_max_valence,
        m_max_edge_incident_faces, m_max_face_adjacent_faces, m_num_patches,
        m_d_face_patch, m_d_edge_patch, m_d_vertex_patch, m_d_patches_ltog_v,
        m_d_patches_ltog_e, m_d_patches_ltog_f, m_d_ad_size_ltog_v,
        m_d_ad_size_ltog_e, m_d_ad_size_ltog_f, m_d_patches_edges,
        m_d_patches_faces, m_d_ad_size, m_d_owned_size, m_max_size,
        m_d_patches_face_ribbon_flag, m_d_patch_distribution_v,
        m_d_patch_distribution_e, m_d_patch_distribution_f);
}


//**************************************************************************


//********************** Export
template <uint32_t patchSize>
void RXMesh<patchSize>::write_connectivity(std::fstream& file) const
{
    for (uint32_t p = 0; p < m_num_patches; ++p) {  // for every patch
        assert(m_h_ad_size[p].w % 3 == 0);
        uint16_t patch_num_faces = m_h_ad_size[p].w / 3;
        for (uint32_t f = 0; f < patch_num_faces; ++f) {
            uint32_t f_global = m_h_patches_ltog_f[p][f] >> 1;
            if (m_patcher->get_face_patch_id(f_global) != p) {
                // if it is a ribbon
                continue;
            }

            file << "f ";
            for (uint32_t e = 0; e < 3; ++e) {
                uint16_t edge = m_h_patches_faces[p][3 * f + e];
                flag_t   dir(0);
                RXMeshContext::unpack_edge_dir(edge, edge, dir);
                uint16_t e_id = (2 * edge) + dir;
                uint16_t v = m_h_patches_edges[p][e_id];
                file << (m_h_patches_ltog_v[p][v] >> 1) + 1 << " ";
            }
            file << std::endl;
        }
    }
}

//**************************************************************************

template class RXMesh<PATCH_SIZE>;
}  // namespace RXMESH
