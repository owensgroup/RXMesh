#include "rxmesh.h"

#include <assert.h>
#include <exception>
#include <memory>
#include <queue>
#include "patcher/patcher.h"
#include "rxmesh/rxmesh_context.h"
#include "rxmesh/util/export_tools.h"
#include "rxmesh/util/math.h"
#include "rxmesh/util/util.h"

namespace rxmesh {
RXMesh::RXMesh(std::vector<std::vector<uint32_t>>& fv,
               const bool                          quite /*= true*/)
    : m_num_edges(0),
      m_num_faces(0),
      m_num_vertices(0),
      m_max_ele_count(0),
      m_max_valence(0),
      m_max_edge_incident_faces(0),
      m_max_face_adjacent_faces(0),
      m_face_degree(3),
      m_num_patches(0),
      m_patch_size(512),
      m_is_input_edge_manifold(true),
      m_is_input_closed(true),
      m_quite(quite),
      m_max_vertices_per_patch(0),
      m_max_edges_per_patch(0),
      m_max_faces_per_patch(0),
      m_d_patches_ltog_v(nullptr),
      m_d_patches_ltog_e(nullptr),
      m_d_patches_ltog_f(nullptr),
      m_d_ad_size_ltog_v(nullptr),
      m_d_ad_size_ltog_e(nullptr),
      m_d_ad_size_ltog_f(nullptr),
      m_d_patches_edges(nullptr),
      m_d_patches_faces(nullptr),
      m_d_patch_distribution_v(nullptr),
      m_d_patch_distribution_e(nullptr),
      m_d_patch_distribution_f(nullptr),
      m_d_ad_size(nullptr),
      m_d_neighbour_patches(nullptr),
      m_d_neighbour_patches_offset(nullptr)
{
    // Build everything from scratch including patches
    build_local(fv);
    device_alloc_local();
}

RXMesh::~RXMesh()
{
    GPU_FREE(m_d_patches_ltog_v);
    GPU_FREE(m_d_patches_ltog_e);
    GPU_FREE(m_d_patches_ltog_f);
    GPU_FREE(m_d_patches_edges);
    GPU_FREE(m_d_patches_faces);
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
    GPU_FREE(m_d_neighbour_patches);
    GPU_FREE(m_d_neighbour_patches_offset);
};

void RXMesh::build_local(std::vector<std::vector<uint32_t>>& fv)
{

    // 5) patch the mesh
    // 6) populate the local mesh

    std::vector<std::vector<uint32_t>> ff;
    std::vector<std::vector<uint32_t>> ef;
    build_supporting_structures(fv, ef, ff);
    calc_statistics(fv, ef);


    //=========== 4)
    // copy fv

    std::vector<std::vector<uint32_t>> rep(fv);
    rep.swap(m_fvn);
    // extend m_fvn by adding the face neighbors
    for (uint32_t e = 0; e < ef.size(); ++e) {
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
    std::unique_ptr<patcher::Patcher> pp =
        std::make_unique<patcher::Patcher>(m_patch_size,
                                           m_fvn,
                                           ff,
                                           fv,
                                           m_edges_map,
                                           m_num_vertices,
                                           m_num_edges,
                                           m_quite);
    pp->execute(
        [this](uint32_t v0, uint32_t v1) { return this->get_edge_id(v0, v1); },
        ef);

    m_patcher     = std::move(pp);
    m_num_patches = m_patcher->get_num_patches();
    // m_patcher->export_patches(Verts);
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

    m_max_vertices_per_patch       = 0;
    m_max_edges_per_patch          = 0;
    m_max_faces_per_patch          = 0;
    m_max_owned_vertices_per_patch = 0;
    m_max_owned_edges_per_patch    = 0;
    m_max_owned_faces_per_patch    = 0;
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
            vv[i]         = dd + vv[i - 1];
            dd            = temp;
        }
        vv[0] = 0;
    };

    ex_scan(m_h_patch_distribution_v);
    ex_scan(m_h_patch_distribution_e);
    ex_scan(m_h_patch_distribution_f);

    if (!m_quite) {
        RXMESH_TRACE("#Vertices = {}, #Faces= {}, #Edges= {}",
                     m_num_vertices,
                     m_num_faces,
                     m_num_edges);
        RXMESH_TRACE("Input is {} edge manifold",
                     ((m_is_input_edge_manifold) ? "" : " Not"));
        RXMESH_TRACE("Input is {} closed", ((m_is_input_closed) ? "" : " Not"));
        RXMESH_TRACE("max valence = {}", m_max_valence);
        RXMESH_TRACE("max edge incident faces = {}", m_max_edge_incident_faces);
        RXMESH_TRACE("max face adjacent faces = {}", m_max_face_adjacent_faces);
        RXMESH_TRACE("per-patch maximum edges references= {}", m_max_size.x);
        RXMESH_TRACE("per-patch maximum  faces references= {}", m_max_size.y);
        RXMESH_TRACE("per-patch maximum face count (owned)= {} ({})",
                     m_max_faces_per_patch,
                     m_max_owned_faces_per_patch);
        RXMESH_TRACE("per-patch maximum edge count (owned) = {} ({})",
                     m_max_edges_per_patch,
                     m_max_owned_edges_per_patch);
        RXMESH_TRACE("per-patch maximum vertex count (owned)= {} ({})",
                     m_max_vertices_per_patch,
                     m_max_owned_vertices_per_patch);
    }
    //===============================

    m_max_ele_count = std::max(m_num_edges, m_num_faces);
    m_max_ele_count = std::max(m_num_vertices, m_max_ele_count);
}

void RXMesh::build_supporting_structures(
    const std::vector<std::vector<uint32_t>>& fv,
    std::vector<std::vector<uint32_t>>&       ef,
    std::vector<std::vector<uint32_t>>&       ff)
{
    m_num_faces    = static_cast<uint32_t>(fv.size());
    m_num_vertices = 0;
    m_num_edges    = 0;
    m_edges_map.clear();

    // assuming manifold mesh i.e., #E = 1.5#F
    ef.clear();
    uint32_t reserve_size =
        static_cast<size_t>(1.5f * static_cast<float>(m_num_faces));
    ef.reserve(reserve_size);
    m_edges_map.reserve(reserve_size);


    ff.clear();
    ff.resize(m_num_faces, std::vector<uint32_t>(0));

    for (uint32_t f = 0; f < fv.size(); ++f) {
        if (fv[f].size() != 3) {
            RXMESH_ERROR(
                "rxmesh::build_supporting_structures() Face {} is not "
                "triangle. Non-triangular faces are not supported",
                f);
            exit(EXIT_FAILURE);
        }

        for (uint32_t v = 0; v < fv[f].size(); ++v) {
            uint32_t v0 = fv[f][v];
            uint32_t v1 = fv[f][(v + 1) % 3];

            m_num_vertices = std::max(m_num_vertices, v0);

            std::pair<uint32_t, uint32_t> edge   = detail::edge_key(v0, v1);
            auto                          e_iter = m_edges_map.find(edge);
            if (e_iter == m_edges_map.end()) {
                uint32_t edge_id = m_num_edges++;
                m_edges_map.insert(std::make_pair(edge, edge_id));
                std::vector<uint32_t> tmp(1, f);
                ef.push_back(tmp);
            } else {
                uint32_t edge_id = (*e_iter).second;

                for (uint32_t f0 = 0; f0 < ef[edge_id].size(); ++f0) {
                    uint32_t other_face = ef[edge_id][f0];
                    ff[other_face].push_back(f);
                    ff[f].push_back(other_face);
                }

                ef[edge_id].push_back(f);
            }
        }
    }
    ++m_num_vertices;

    if (m_num_edges != static_cast<uint32_t>(m_edges_map.size())) {
        RXMESH_ERROR(
            "rxmesh::build_supporting_structures() m_num_edges ({}) should "
            "match the size of edge_map ({})",
            m_num_edges,
            m_edges_map.size());
        exit(EXIT_FAILURE);
    }
}

void RXMesh::calc_statistics(const std::vector<std::vector<uint32_t>>& fv,
                             const std::vector<std::vector<uint32_t>>& ef)
{
    if (m_num_vertices == 0 || m_num_faces == 0 || m_num_edges == 0 ||
        fv.size() == 0 || ef.size() == 0) {
        RXMESH_ERROR(
            "RXMesh::calc_statistics() input mesh has not been initialized");
        exit(EXIT_FAILURE);
    }

    // calc max valence, max ef, is input closed, and is input manifold
    m_max_edge_incident_faces = 0;
    m_max_valence             = 0;
    std::vector<uint32_t> vv_count(m_num_vertices, 0);
    m_is_input_closed        = true;
    m_is_input_edge_manifold = true;
    for (auto& e_iter : m_edges_map) {
        uint32_t v0 = e_iter.first.first;
        uint32_t v1 = e_iter.first.second;

        vv_count[v0]++;
        vv_count[v1]++;

        m_max_valence = std::max(m_max_valence, vv_count[v0]);
        m_max_valence = std::max(m_max_valence, vv_count[v1]);

        uint32_t edge_id = e_iter.second;
        m_max_edge_incident_faces =
            std::max(m_max_edge_incident_faces, uint32_t(ef[edge_id].size()));

        if (ef[edge_id].size() < 2) {
            m_is_input_closed = false;
        }
        if (ef[edge_id].size() > 2) {
            m_is_input_edge_manifold = false;
        }
    }

    // calc max ff
    m_max_face_adjacent_faces = 0;
    for (uint32_t f = 0; f < fv.size(); ++f) {
        uint32_t ff_count = 0;
        for (uint32_t v = 0; v < fv[f].size(); ++v) {
            uint32_t v0       = fv[f][v];
            uint32_t v1       = fv[f][(v + 1) % 3];
            uint32_t edge_num = get_edge_id(v0, v1);
            ff_count += ef[edge_num].size() - 1;
        }
        m_max_face_adjacent_faces =
            std::max(ff_count, m_max_face_adjacent_faces);
    }
}

void RXMesh::build_patch_locally(const uint32_t patch_id)
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
    const uint32_t p_end   = p_off[patch_id];
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
    tmp_e.reserve(m_patch_size * 3);
    tmp_v.reserve(m_patch_size);
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
                detail::edge_key(global_v0, global_v1);
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
        create_new_local_face(patch_id,
                              global_f,
                              m_fvn[global_f],
                              faces_count,
                              edges_owned_count,
                              edges_not_owned_count,
                              vertices_owned_count,
                              vertices_not_owned_count,
                              num_edges_owned,
                              num_vertices_owned,
                              f_ltog,
                              e_ltog,
                              v_ltog,
                              fp,
                              ep);
    }


    // 2) loop over ribbon faces
    for (uint32_t s = r_start; s < r_end; ++s) {
        uint32_t global_f = m_patcher->get_external_ribbon_val()[s];
        create_new_local_face(patch_id,
                              global_f,
                              m_fvn[global_f],
                              faces_count,
                              edges_owned_count,
                              edges_not_owned_count,
                              vertices_owned_count,
                              vertices_not_owned_count,
                              num_edges_owned,
                              num_vertices_owned,
                              f_ltog,
                              e_ltog,
                              v_ltog,
                              fp,
                              ep);
    }

    if (vertices_owned_count != num_vertices_owned ||
        edges_owned_count != num_edges_owned ||
        edges_owned_count + edges_not_owned_count != total_patch_num_edges ||
        vertices_owned_count + vertices_not_owned_count !=
            total_patch_num_vertices) {
        RXMESH_ERROR("rxmesh::build_patch_locally() patch is " +
                     std::to_string(patch_id) + " not built correctly!!");
    }


    m_h_owned_size[patch_id].x = (p_end - p_start);
    m_h_owned_size[patch_id].y = num_edges_owned;
    m_h_owned_size[patch_id].z = num_vertices_owned;

    // faces
    m_h_patches_faces.push_back(fp);
    m_h_patches_ltog_f.push_back(f_ltog);


    // edges
    m_h_patches_edges.push_back(ep);
    m_h_patches_ltog_e.push_back(e_ltog);

    // vertices
    m_h_patches_ltog_v.push_back(v_ltog);
}

uint16_t RXMesh::create_new_local_face(const uint32_t               patch_id,
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
                                       std::vector<uint16_t>& ep)
{

    uint16_t local_f = faces_count++;
    f_ltog[local_f]  = global_f;

    // shift to left and set first bit to 1 if global_f's patch is this patch
    f_ltog[local_f] = f_ltog[local_f] << 1;
    f_ltog[local_f] =
        f_ltog[local_f] | (m_patcher->get_face_patch_id(global_f) == patch_id);

    auto find_increment_index = [&patch_id](
                                    const uint32_t&        global,
                                    std::vector<uint32_t>& vect,
                                    uint16_t&              owned_count,
                                    uint16_t&              not_owned_count,
                                    const uint16_t         num_owned,
                                    bool&                  incremented,
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
            ret_id   = owned_count++;
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
        std::pair<uint32_t, uint32_t> my_edge = detail::edge_key(global_v0, global_v1);

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
        uint16_t local_e =
            find_increment_index(global_e,
                                 e_ltog,
                                 edges_owned_count,
                                 edges_not_owned_count,
                                 num_edges_owned,
                                 new_e,
                                 m_patcher->get_edge_patch_id(global_e));

        if (new_e) {
            // if it is new edges, then we need to either look for
            // its vertices. if there were inserted before in the
            // patch, then retrieve their local id. otherwise, we
            // new vertices to the patch
            assert(my_edge.first != my_edge.second);

            bool     new_v(false);
            uint16_t local_v0 = find_increment_index(
                my_edge.first,
                v_ltog,
                vertices_owned_count,
                vertices_not_owned_count,
                num_vertices_owned,
                new_v,
                m_patcher->get_vertex_patch_id(my_edge.first));

            uint16_t local_v1 = find_increment_index(
                my_edge.second,
                v_ltog,
                vertices_owned_count,
                vertices_not_owned_count,
                num_vertices_owned,
                new_v,
                m_patcher->get_vertex_patch_id(my_edge.second));

            assert(local_v0 != local_v1);

            // new edges are appended in the end of e_ltog
            // and so as their vertices in ep
            ep[2 * local_e]     = local_v0;
            ep[2 * local_e + 1] = local_v1;
        }
        // shift local_e to left
        // set the first bit to 1 if (dir ==1)
        local_e                         = local_e << 1;
        local_e                         = local_e | (dir & 1);
        fp[local_f * m_face_degree + j] = local_e;
    }

    return local_f;
}

uint32_t RXMesh::get_edge_id(const uint32_t v0, const uint32_t v1) const
{
    // v0 and v1 are two vertices in global space. we return the edge
    // id in global space also (by querying m_edges_map)
    assert(m_edges_map.size() != 0);

    std::pair<uint32_t, uint32_t> edge = detail::edge_key(v0, v1);

    assert(edge.first == v0 || edge.first == v1);
    assert(edge.second == v0 || edge.second == v1);

    return get_edge_id(edge);
}

uint32_t RXMesh::get_edge_id(const std::pair<uint32_t, uint32_t>& edge) const
{
    uint32_t edge_id = -1;
    try {
        edge_id = m_edges_map.at(edge);
    } catch (const std::out_of_range&) {
        RXMESH_ERROR(
            "rxmesh::get_edge_id() mapping edges went wrong."
            " Can not find an edge connecting vertices {} and {}",
            edge.first,
            edge.second);
        exit(EXIT_FAILURE);
    }

    return edge_id;
}

template <typename Tin, typename Tst>
void RXMesh::get_starting_ids(const std::vector<std::vector<Tin>>& input,
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


template <typename Tin, typename Tad>
void RXMesh::get_size(const std::vector<std::vector<Tin>>& input,
                      std::vector<Tad>&                    ad)
{
    // get the size of each element of input and store it as the second(y)
    // component in ad
    assert(ad.size() >= input.size());

    for (uint32_t p = 0; p < input.size(); ++p) {
        ad[p].y = input[p].size();
    }
}

template <typename T>
void RXMesh::padding_to_multiple(std::vector<std::vector<T>>& input,
                                 const uint32_t               multiple,
                                 const T                      init_val)
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

void RXMesh::device_alloc_local()
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
    padding_to_multiple(
        m_h_patches_edges, WARPSIZE, static_cast<uint16_t>(INVALID16));
    padding_to_multiple(
        m_h_patches_faces, WARPSIZE, static_cast<uint16_t>(INVALID16));
    padding_to_multiple(
        m_h_patches_ltog_v, WARPSIZE, static_cast<uint32_t>(INVALID32));
    padding_to_multiple(
        m_h_patches_ltog_e, WARPSIZE, static_cast<uint32_t>(INVALID32));
    padding_to_multiple(
        m_h_patches_ltog_f, WARPSIZE, static_cast<uint32_t>(INVALID32));

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

        uint32_t start_v     = m_h_ad_size_ltog_v[p].x;
        uint32_t start_e     = m_h_ad_size_ltog_e[p].x;
        uint32_t start_f     = m_h_ad_size_ltog_f[p].x;
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
            m_d_patches_faces + start_faces,
            m_h_patches_faces[p].data(),
            m_h_ad_size_ltog_f[p].y * m_face_degree * sizeof(uint16_t),
            cudaMemcpyHostToDevice));
    }


    // copy ad_size
    CUDA_ERROR(cudaMemcpy(m_d_ad_size_ltog_v,
                          m_h_ad_size_ltog_v.data(),
                          sizeof(uint2) * (m_num_patches + 1),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_ad_size_ltog_e,
                          m_h_ad_size_ltog_e.data(),
                          sizeof(uint2) * (m_num_patches + 1),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_ad_size_ltog_f,
                          m_h_ad_size_ltog_f.data(),
                          sizeof(uint2) * (m_num_patches + 1),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_ad_size,
                          m_h_ad_size.data(),
                          sizeof(uint4) * (m_num_patches + 1),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_owned_size,
                          m_h_owned_size.data(),
                          sizeof(uint4) * (m_num_patches),
                          cudaMemcpyHostToDevice));


    // allocate and copy face/vertex/edge patch
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_face_patch, sizeof(uint32_t) * (m_num_faces)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_edge_patch, sizeof(uint32_t) * (m_num_edges)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_vertex_patch,
                          sizeof(uint32_t) * (m_num_vertices)));

    CUDA_ERROR(cudaMemcpy(m_d_face_patch,
                          this->m_patcher->get_face_patch().data(),
                          sizeof(uint32_t) * (m_num_faces),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_edge_patch,
                          this->m_patcher->get_edge_patch().data(),
                          sizeof(uint32_t) * (m_num_edges),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_vertex_patch,
                          this->m_patcher->get_vertex_patch().data(),
                          sizeof(uint32_t) * (m_num_vertices),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMalloc((void**)&m_d_patch_distribution_v,
                          (m_num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_patch_distribution_e,
                          (m_num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_patch_distribution_f,
                          (m_num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(m_d_patch_distribution_v,
                          m_h_patch_distribution_v.data(),
                          (m_num_patches + 1) * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_patch_distribution_e,
                          m_h_patch_distribution_e.data(),
                          (m_num_patches + 1) * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_patch_distribution_f,
                          m_h_patch_distribution_f.data(),
                          (m_num_patches + 1) * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));


    uint32_t* n_patches        = m_patcher->get_neighbour_patches();
    uint32_t* n_patches_offset = m_patcher->get_neighbour_patches_offset();

    CUDA_ERROR(cudaMalloc((void**)&m_d_neighbour_patches_offset,
                          m_num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(m_d_neighbour_patches_offset,
                          n_patches_offset,
                          m_num_patches * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    if (n_patches) {
        CUDA_ERROR(
            cudaMalloc((void**)&m_d_neighbour_patches,
                       n_patches_offset[m_num_patches - 1] * sizeof(uint32_t)));
        CUDA_ERROR(
            cudaMemcpy(m_d_neighbour_patches,
                       n_patches,
                       n_patches_offset[m_num_patches - 1] * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
    }


    // Allocate and copy the context to the gpu
    m_rxmesh_context.init(m_num_edges,
                          m_num_faces,
                          m_num_vertices,
                          m_face_degree,
                          m_max_valence,
                          m_max_edge_incident_faces,
                          m_max_face_adjacent_faces,
                          m_num_patches,
                          m_d_face_patch,
                          m_d_edge_patch,
                          m_d_vertex_patch,
                          m_d_patches_ltog_v,
                          m_d_patches_ltog_e,
                          m_d_patches_ltog_f,
                          m_d_ad_size_ltog_v,
                          m_d_ad_size_ltog_e,
                          m_d_ad_size_ltog_f,
                          m_d_patches_edges,
                          m_d_patches_faces,
                          m_d_ad_size,
                          m_d_owned_size,
                          m_max_size,
                          m_d_patch_distribution_v,
                          m_d_patch_distribution_e,
                          m_d_patch_distribution_f,
                          m_d_neighbour_patches,
                          m_d_neighbour_patches_offset);
}

void RXMesh::write_connectivity(std::fstream& file) const
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
                uint16_t v    = m_h_patches_edges[p][e_id];
                file << (m_h_patches_ltog_v[p][v] >> 1) + 1 << " ";
            }
            file << std::endl;
        }
    }
}

}  // namespace rxmesh
