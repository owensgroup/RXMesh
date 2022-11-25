#include <assert.h>
#include <omp.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>

#include "patcher/patcher.h"
#include "rxmesh/context.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/util.h"

namespace rxmesh {
RXMesh::RXMesh()
    : m_num_edges(0),
      m_num_faces(0),
      m_num_vertices(0),
      m_input_max_valence(0),
      m_input_max_edge_incident_faces(0),
      m_input_max_face_adjacent_faces(0),
      m_num_patches(0),
      m_patch_size(512),
      m_is_input_edge_manifold(true),
      m_is_input_closed(true),
      m_quite(false),
      m_d_patches_info(nullptr),
      m_h_patches_info(nullptr),
      m_capacity_factor(0),
      m_lp_hashtable_load_factor(0)
{
}

void RXMesh::init(const std::vector<std::vector<uint32_t>>& fv,
                  const std::string                         patcher_file,
                  const bool                                quite,
                  const float                               capacity_factor,
                  const float lp_hashtable_load_factor)
{
    m_quite                    = quite;
    m_capacity_factor          = capacity_factor;
    m_lp_hashtable_load_factor = lp_hashtable_load_factor;

    // Build everything from scratch including patches
    if (fv.empty()) {
        RXMESH_ERROR(
            "RXMesh::init input fv is empty. Can not build RXMesh properly");
    }
    if (m_capacity_factor < 1.0) {
        RXMESH_ERROR("RXMesh::init capacity factor should be at least one");
    }
    if (m_lp_hashtable_load_factor > 1.0) {
        RXMESH_ERROR(
            "RXMesh::init hashtable load factor should be less than 1");
    }

    build(fv, patcher_file);
    build_device();

    calc_max_elements();

    // Allocate and copy the context to the gpu
    m_rxmesh_context.init(m_num_vertices,
                          m_num_edges,
                          m_num_faces,
                          m_max_vertices_per_patch,
                          m_max_edges_per_patch,
                          m_max_faces_per_patch,
                          m_num_patches,
                          m_d_patches_info);


    if (!m_quite) {
        RXMESH_TRACE("#Vertices = {}, #Faces= {}, #Edges= {}",
                     m_num_vertices,
                     m_num_faces,
                     m_num_edges);
        RXMESH_TRACE("Input is {} edge manifold",
                     ((m_is_input_edge_manifold) ? "" : " Not"));
        RXMESH_TRACE("Input is {} closed", ((m_is_input_closed) ? "" : " Not"));
        RXMESH_TRACE("Input max valence = {}", m_input_max_valence);
        RXMESH_TRACE("max edge incident faces = {}",
                     m_input_max_edge_incident_faces);
        RXMESH_TRACE("max face adjacent faces = {}",
                     m_input_max_face_adjacent_faces);
        RXMESH_TRACE("per-patch maximum face count = {}",
                     m_max_faces_per_patch);
        RXMESH_TRACE("per-patch maximum edge count = {}",
                     m_max_edges_per_patch);
        RXMESH_TRACE("per-patch maximum vertex count = {}",
                     m_max_vertices_per_patch);
        RXMESH_TRACE("per-patch maximum not-owned face count = {}",
                     m_max_not_owned_faces);
        RXMESH_TRACE("per-patch maximum not-owned edge count = {}",
                     m_max_not_owned_edges);
        RXMESH_TRACE("per-patch maximum not-owned vertex count = {}",
                     m_max_not_owned_vertices);
    }
}

RXMesh::~RXMesh()
{
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        free(m_h_patches_info[p].active_mask_v);
        free(m_h_patches_info[p].active_mask_e);
        free(m_h_patches_info[p].active_mask_f);
        free(m_h_patches_info[p].owned_mask_v);
        free(m_h_patches_info[p].owned_mask_e);
        free(m_h_patches_info[p].owned_mask_f);
        free(m_h_patches_info[p].num_faces);
        m_h_patches_info[p].lp_v.free();
        m_h_patches_info[p].lp_e.free();
        m_h_patches_info[p].lp_f.free();
        m_h_patches_info[p].patch_stash.free();
    }

    // m_d_patches_info is a pointer to pointer(s) which we can not dereference
    // on the host so we copy these pointers to the host by re-using
    // m_h_patches_info and then free the memory these pointers are pointing to.
    // Finally, we free the parent pointer memory

    CUDA_ERROR(cudaMemcpy(m_h_patches_info,
                          m_d_patches_info,
                          m_num_patches * sizeof(PatchInfo),
                          cudaMemcpyDeviceToHost));

    for (uint32_t p = 0; p < m_num_patches; ++p) {
        GPU_FREE(m_h_patches_info[p].active_mask_v);
        GPU_FREE(m_h_patches_info[p].active_mask_e);
        GPU_FREE(m_h_patches_info[p].active_mask_f);
        GPU_FREE(m_h_patches_info[p].owned_mask_v);
        GPU_FREE(m_h_patches_info[p].owned_mask_e);
        GPU_FREE(m_h_patches_info[p].owned_mask_f);
        GPU_FREE(m_h_patches_info[p].ev);
        GPU_FREE(m_h_patches_info[p].fe);
        GPU_FREE(m_h_patches_info[p].num_faces);
        m_h_patches_info[p].lp_v.free();
        m_h_patches_info[p].lp_e.free();
        m_h_patches_info[p].lp_f.free();
        m_h_patches_info[p].patch_stash.free();
    }
    GPU_FREE(m_d_patches_info);
    free(m_h_patches_info);
    m_rxmesh_context.release();
}

void RXMesh::build(const std::vector<std::vector<uint32_t>>& fv,
                   const std::string                         patcher_file)
{
    std::vector<uint32_t>              ff_values;
    std::vector<uint32_t>              ff_offset;
    std::vector<std::vector<uint32_t>> ef;

    build_supporting_structures(fv, ef, ff_offset, ff_values);

    if (!patcher_file.empty()) {
        m_patcher = std::make_unique<patcher::Patcher>(patcher_file);
    } else {
        m_patcher = std::make_unique<patcher::Patcher>(m_patch_size,
                                                       ff_offset,
                                                       ff_values,
                                                       fv,
                                                       m_edges_map,
                                                       m_num_vertices,
                                                       m_num_edges,
                                                       m_quite);
    }


    m_num_patches    = m_patcher->get_num_patches();
    m_h_patches_info = (PatchInfo*)malloc(m_num_patches * sizeof(PatchInfo));
    m_h_patches_ltog_f.resize(m_num_patches);
    m_h_patches_ltog_e.resize(m_num_patches);
    m_h_patches_ltog_v.resize(m_num_patches);
    m_h_num_owned_f.resize(m_num_patches);
    m_h_num_owned_v.resize(m_num_patches);
    m_h_num_owned_e.resize(m_num_patches);
    m_h_vertex_prefix.resize(m_num_patches + 1, 0);
    m_h_edge_prefix.resize(m_num_patches + 1, 0);
    m_h_face_prefix.resize(m_num_patches + 1, 0);
#pragma omp parallel for
    for (int p = 0; p < static_cast<int>(m_num_patches); ++p) {
        build_single_patch(fv, p);
    }

    for (uint32_t p = 0; p < m_num_patches; ++p) {
        m_h_vertex_prefix[p + 1] = m_h_vertex_prefix[p] + m_h_num_owned_v[p];
        m_h_edge_prefix[p + 1]   = m_h_edge_prefix[p] + m_h_num_owned_e[p];
        m_h_face_prefix[p + 1]   = m_h_face_prefix[p] + m_h_num_owned_f[p];
    }

    calc_input_statistics(fv, ef);
}

const std::pair<uint32_t, uint16_t> RXMesh::map_to_local(
    const uint32_t               i,
    const std::vector<uint32_t>& element_prefix) const
{
    const auto end = element_prefix.end();

    auto p = std::lower_bound(
        element_prefix.begin(), end, i, [](int a, int b) { return a <= b; });
    if (p == end) {
        RXMESH_ERROR(
            "RXMeshStatic::map_to_local can not its patch. Input is out of "
            "pound!");
    }
    p -= 1;
    return {std::distance(element_prefix.begin(), p), i - *p};
}

void RXMesh::build_supporting_structures(
    const std::vector<std::vector<uint32_t>>& fv,
    std::vector<std::vector<uint32_t>>&       ef,
    std::vector<uint32_t>&                    ff_offset,
    std::vector<uint32_t>&                    ff_values)
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

    std::vector<uint32_t> ff_size(m_num_faces, 0);

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
                    ++ff_size[other_face];
                }
                ff_size[f] += ef[edge_id].size();

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

    ff_offset.resize(m_num_faces);
    std::inclusive_scan(ff_size.begin(), ff_size.end(), ff_offset.begin());
    ff_values.clear();
    ff_values.resize(ff_offset.back());
    std::fill(ff_size.begin(), ff_size.end(), 0);

    for (uint32_t e = 0; e < m_num_edges; ++e) {
        for (uint32_t i = 0; i < ef[e].size(); ++i) {
            uint32_t f0 = ef[e][i];
            for (uint32_t j = i + 1; j < ef[e].size(); ++j) {
                uint32_t f1 = ef[e][j];

                uint32_t f0_offset = ff_size[f0]++;
                uint32_t f1_offset = ff_size[f1]++;
                f0_offset += (f0 == 0) ? 0 : ff_offset[f0 - 1];
                f1_offset += (f1 == 0) ? 0 : ff_offset[f1 - 1];

                ff_values[f0_offset] = f1;
                ff_values[f1_offset] = f0;
            }
        }
    }
}

void RXMesh::calc_input_statistics(const std::vector<std::vector<uint32_t>>& fv,
                                   const std::vector<std::vector<uint32_t>>& ef)
{
    if (m_num_vertices == 0 || m_num_faces == 0 || m_num_edges == 0 ||
        fv.size() == 0 || ef.size() == 0) {
        RXMESH_ERROR(
            "RXMesh::calc_statistics() input mesh has not been initialized");
        exit(EXIT_FAILURE);
    }

    // calc max valence, max ef, is input closed, and is input manifold
    m_input_max_edge_incident_faces = 0;
    m_input_max_valence             = 0;
    std::vector<uint32_t> vv_count(m_num_vertices, 0);
    m_is_input_closed        = true;
    m_is_input_edge_manifold = true;
    for (const auto& e_iter : m_edges_map) {
        uint32_t v0 = e_iter.first.first;
        uint32_t v1 = e_iter.first.second;

        vv_count[v0]++;
        vv_count[v1]++;

        m_input_max_valence = std::max(m_input_max_valence, vv_count[v0]);
        m_input_max_valence = std::max(m_input_max_valence, vv_count[v1]);

        uint32_t edge_id                = e_iter.second;
        m_input_max_edge_incident_faces = std::max(
            m_input_max_edge_incident_faces, uint32_t(ef[edge_id].size()));

        if (ef[edge_id].size() < 2) {
            m_is_input_closed = false;
        }
        if (ef[edge_id].size() > 2) {
            m_is_input_edge_manifold = false;
        }
    }

    // calc max ff
    m_input_max_face_adjacent_faces = 0;
    for (uint32_t f = 0; f < fv.size(); ++f) {
        uint32_t ff_count = 0;
        for (uint32_t v = 0; v < fv[f].size(); ++v) {
            uint32_t v0       = fv[f][v];
            uint32_t v1       = fv[f][(v + 1) % 3];
            uint32_t edge_num = get_edge_id(v0, v1);
            ff_count += ef[edge_num].size() - 1;
        }
        m_input_max_face_adjacent_faces =
            std::max(ff_count, m_input_max_face_adjacent_faces);
    }
}

void RXMesh::calc_max_elements()
{
    m_max_not_owned_vertices = 0;
    m_max_not_owned_edges    = 0;
    m_max_not_owned_faces    = 0;
    m_max_vertices_per_patch = 0;
    m_max_edges_per_patch    = 0;
    m_max_faces_per_patch    = 0;


    for (uint32_t p = 0; p < this->m_num_patches; ++p) {
        m_max_vertices_per_patch = std::max(
            m_max_vertices_per_patch, m_h_patches_info[p].num_vertices[0]);

        m_max_edges_per_patch =
            std::max(m_max_edges_per_patch, m_h_patches_info[p].num_edges[0]);

        m_max_faces_per_patch =
            std::max(m_max_faces_per_patch, m_h_patches_info[p].num_faces[0]);


        m_max_not_owned_vertices =
            std::max(m_max_not_owned_vertices,
                     detail::count_set_bits(m_h_patches_info[p].num_vertices[0],
                                            m_h_patches_info[p].owned_mask_v));

        m_max_not_owned_edges =
            std::max(m_max_not_owned_edges,
                     detail::count_set_bits(m_h_patches_info[p].num_edges[0],
                                            m_h_patches_info[p].owned_mask_e));

        m_max_not_owned_faces =
            std::max(m_max_not_owned_faces,
                     detail::count_set_bits(m_h_patches_info[p].num_faces[0],
                                            m_h_patches_info[p].owned_mask_f));
    }
}

uint32_t RXMesh::max_bitmask_size(ELEMENT ele) const
{
    switch (ele) {
        case rxmesh::ELEMENT::VERTEX:
            return detail::mask_num_bytes(this->m_max_vertices_per_patch);
        case rxmesh::ELEMENT::EDGE:
            return detail::mask_num_bytes(this->m_max_edges_per_patch);
        case rxmesh::ELEMENT::FACE:
            return detail::mask_num_bytes(this->m_max_faces_per_patch);
        default:
            RXMESH_ERROR(
                "RXMesh::max_bitmask_size() unknown mesh element type");
            return 0;
    }
}

uint32_t RXMesh::max_lp_hashtable_size(ELEMENT ele) const
{
    switch (ele) {
        case rxmesh::ELEMENT::VERTEX:
            return static_cast<uint16_t>(
                static_cast<float>(m_max_not_owned_vertices) /
                m_lp_hashtable_load_factor);
        case rxmesh::ELEMENT::EDGE:
            return static_cast<uint16_t>(
                static_cast<float>(m_max_not_owned_edges) /
                m_lp_hashtable_load_factor);
        case rxmesh::ELEMENT::FACE:
            return static_cast<uint16_t>(
                static_cast<float>(m_max_not_owned_faces) /
                m_lp_hashtable_load_factor);
        default:
            RXMESH_ERROR(
                "RXMesh::max_lp_hashtable_size() unknown mesh element type");
            return 0;
    }
}

void RXMesh::build_single_patch(const std::vector<std::vector<uint32_t>>& fv,
                                const uint32_t patch_id)
{
    // Build the patch local index space
    // This is the two small matrices defining incident relation between
    // edge-vertices and faces-edges (i.e., the topology) along with the mapping
    // from local to global space for vertices, edge, and faces

    // When we create a new patch, we make sure that the elements owned by the
    // patch will have local indices lower than any other elements (of the same
    // type) that is not owned by the patch.

    build_single_patch_ltog(fv, patch_id);

    build_single_patch_topology(fv, patch_id);
}

void RXMesh::build_single_patch_ltog(
    const std::vector<std::vector<uint32_t>>& fv,
    const uint32_t                            patch_id)
{
    // patch start and end
    const uint32_t p_start =
        (patch_id == 0) ? 0 : m_patcher->get_patches_offset()[patch_id - 1];
    const uint32_t p_end = m_patcher->get_patches_offset()[patch_id];

    // ribbon start and end
    const uint32_t r_start =
        (patch_id == 0) ? 0 :
                          m_patcher->get_external_ribbon_offset()[patch_id - 1];
    const uint32_t r_end = m_patcher->get_external_ribbon_offset()[patch_id];


    const uint32_t total_patch_num_faces =
        (p_end - p_start) + (r_end - r_start);
    m_h_patches_ltog_f[patch_id].resize(total_patch_num_faces);
    m_h_patches_ltog_v[patch_id].resize(3 * total_patch_num_faces);
    m_h_patches_ltog_e[patch_id].resize(3 * total_patch_num_faces);

    auto add_new_face = [&](uint32_t global_face_id, uint16_t local_face_id) {
        m_h_patches_ltog_f[patch_id][local_face_id] = global_face_id;

        for (uint32_t v = 0; v < 3; ++v) {
            uint32_t v0 = fv[global_face_id][v];
            uint32_t v1 = fv[global_face_id][(v + 1) % 3];

            uint32_t edge_id = get_edge_id(v0, v1);

            m_h_patches_ltog_v[patch_id][local_face_id * 3 + v] = v0;

            m_h_patches_ltog_e[patch_id][local_face_id * 3 + v] = edge_id;
        }
    };

    uint16_t local_face_id = 0;
    for (uint32_t f = p_start; f < p_end; ++f) {
        uint32_t face_id = m_patcher->get_patches_val()[f];
        add_new_face(face_id, local_face_id++);
    }

    for (uint32_t f = r_start; f < r_end; ++f) {
        uint32_t face_id = m_patcher->get_external_ribbon_val()[f];
        add_new_face(face_id, local_face_id++);
    }


    auto create_unique_mapping = [&](std::vector<uint32_t>&       ltog_map,
                                     const std::vector<uint32_t>& patch) {
        std::sort(ltog_map.begin(), ltog_map.end());
        auto unique_end = std::unique(ltog_map.begin(), ltog_map.end());
        ltog_map.resize(unique_end - ltog_map.begin());

        // we use stable partition since we want ltog to be sorted so we can
        // use binary search on it when we populate the topology
        auto part_end = std::stable_partition(
            ltog_map.begin(), ltog_map.end(), [&patch, patch_id](uint32_t i) {
                return patch[i] == patch_id;
            });
        return static_cast<uint16_t>(part_end - ltog_map.begin());
    };

    m_h_num_owned_f[patch_id] = create_unique_mapping(
        m_h_patches_ltog_f[patch_id], m_patcher->get_face_patch());

    m_h_num_owned_e[patch_id] = create_unique_mapping(
        m_h_patches_ltog_e[patch_id], m_patcher->get_edge_patch());

    m_h_num_owned_v[patch_id] = create_unique_mapping(
        m_h_patches_ltog_v[patch_id], m_patcher->get_vertex_patch());
}

void RXMesh::build_single_patch_topology(
    const std::vector<std::vector<uint32_t>>& fv,
    const uint32_t                            patch_id)
{
    // patch start and end
    const uint32_t p_start =
        (patch_id == 0) ? 0 : m_patcher->get_patches_offset()[patch_id - 1];
    const uint32_t p_end = m_patcher->get_patches_offset()[patch_id];

    // ribbon start and end
    const uint32_t r_start =
        (patch_id == 0) ? 0 :
                          m_patcher->get_external_ribbon_offset()[patch_id - 1];
    const uint32_t r_end = m_patcher->get_external_ribbon_offset()[patch_id];

    const uint16_t patch_num_edges = m_h_patches_ltog_e[patch_id].size();
    const uint16_t patch_num_faces = m_h_patches_ltog_f[patch_id].size();

    m_h_patches_info[patch_id].ev =
        (LocalVertexT*)malloc(patch_num_edges * 2 * sizeof(LocalVertexT));
    m_h_patches_info[patch_id].fe =
        (LocalEdgeT*)malloc(patch_num_faces * 3 * sizeof(LocalEdgeT));

    m_h_patches_info[patch_id].ev =
        (LocalVertexT*)malloc(patch_num_edges * 2 * sizeof(LocalVertexT));
    m_h_patches_info[patch_id].fe =
        (LocalEdgeT*)malloc(patch_num_faces * 3 * sizeof(LocalEdgeT));

    std::vector<bool> is_added_edge(patch_num_edges, false);

    auto find_local_index = [&patch_id](
                                const uint32_t               global_id,
                                const uint32_t               element_patch,
                                const uint16_t               num_owned_elements,
                                const std::vector<uint32_t>& ltog) -> uint16_t {
        uint32_t start = 0;
        uint32_t end   = num_owned_elements;
        if (element_patch != patch_id) {
            start = num_owned_elements;
            end   = ltog.size();
        }
        auto it = std::lower_bound(
            ltog.begin() + start, ltog.begin() + end, global_id);
        if (it == ltog.begin() + end) {
            return INVALID16;
        } else {
            return static_cast<uint16_t>(it - ltog.begin());
        }
    };


    auto add_new_face = [&](const uint32_t global_face_id) {
        const uint16_t local_face_id =
            find_local_index(global_face_id,
                             m_patcher->get_face_patch_id(global_face_id),
                             m_h_num_owned_f[patch_id],
                             m_h_patches_ltog_f[patch_id]);

        for (uint32_t v = 0; v < 3; ++v) {


            const uint32_t global_v0 = fv[global_face_id][v];
            const uint32_t global_v1 = fv[global_face_id][(v + 1) % 3];

            std::pair<uint32_t, uint32_t> edge_key =
                detail::edge_key(global_v0, global_v1);

            assert(edge_key.first == global_v0 || edge_key.first == global_v1);
            assert(edge_key.second == global_v0 ||
                   edge_key.second == global_v1);

            int dir = 1;
            if (edge_key.first == global_v0 && edge_key.second == global_v1) {
                dir = 0;
            }

            const uint32_t global_edge_id = get_edge_id(edge_key);

            uint16_t local_edge_id =
                find_local_index(global_edge_id,
                                 m_patcher->get_edge_patch_id(global_edge_id),
                                 m_h_num_owned_e[patch_id],
                                 m_h_patches_ltog_e[patch_id]);

            assert(local_edge_id != INVALID16);
            if (!is_added_edge[local_edge_id]) {

                is_added_edge[local_edge_id] = true;

                const uint16_t local_v0 = find_local_index(
                    edge_key.first,
                    m_patcher->get_vertex_patch_id(edge_key.first),
                    m_h_num_owned_v[patch_id],
                    m_h_patches_ltog_v[patch_id]);

                const uint16_t local_v1 = find_local_index(
                    edge_key.second,
                    m_patcher->get_vertex_patch_id(edge_key.second),
                    m_h_num_owned_v[patch_id],
                    m_h_patches_ltog_v[patch_id]);

                assert(local_v0 != INVALID16 && local_v1 != INVALID16);

                m_h_patches_info[patch_id].ev[local_edge_id * 2].id = local_v0;
                m_h_patches_info[patch_id].ev[local_edge_id * 2 + 1].id =
                    local_v1;
            }

            // shift local_e to left
            // set the first bit to 1 if (dir ==1)
            local_edge_id = local_edge_id << 1;
            local_edge_id = local_edge_id | (dir & 1);
            m_h_patches_info[patch_id].fe[local_face_id * 3 + v].id =
                local_edge_id;
        }
    };


    for (uint32_t f = p_start; f < p_end; ++f) {
        uint32_t face_id = m_patcher->get_patches_val()[f];
        add_new_face(face_id);
    }

    for (uint32_t f = r_start; f < r_end; ++f) {
        uint32_t face_id = m_patcher->get_external_ribbon_val()[f];
        add_new_face(face_id);
    }
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
    uint32_t edge_id = INVALID32;
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

void RXMesh::build_device()
{
    CUDA_ERROR(cudaMalloc((void**)&m_d_patches_info,
                          m_num_patches * sizeof(PatchInfo)));

#pragma omp parallel for
    for (int p = 0; p < static_cast<int>(m_num_patches); ++p) {
        uint16_t* h_counts = (uint16_t*)malloc(6 * sizeof(uint16_t));

        m_h_patches_info[p].num_faces         = h_counts;
        m_h_patches_info[p].num_faces[0]      = m_h_patches_ltog_f[p].size();
        m_h_patches_info[p].num_edges         = h_counts + 1;
        m_h_patches_info[p].num_edges[0]      = m_h_patches_ltog_e[p].size();
        m_h_patches_info[p].num_vertices      = h_counts + 2;
        m_h_patches_info[p].num_vertices[0]   = m_h_patches_ltog_v[p].size();
        m_h_patches_info[p].faces_capacity    = h_counts + 3;
        m_h_patches_info[p].faces_capacity[0] = static_cast<uint16_t>(
            m_capacity_factor *
            static_cast<float>(m_h_patches_info[p].num_faces[0]));
        m_h_patches_info[p].edges_capacity    = h_counts + 4;
        m_h_patches_info[p].edges_capacity[0] = static_cast<uint16_t>(
            m_capacity_factor *
            static_cast<float>(m_h_patches_info[p].num_edges[0]));
        m_h_patches_info[p].vertices_capacity    = h_counts + 5;
        m_h_patches_info[p].vertices_capacity[0] = static_cast<uint16_t>(
            m_capacity_factor *
            static_cast<float>(m_h_patches_info[p].num_vertices[0]));
        m_h_patches_info[p].patch_id    = p;
        m_h_patches_info[p].patch_stash = PatchStash(false);


        uint16_t* d_counts;
        CUDA_ERROR(cudaMalloc((void**)&d_counts, 6 * sizeof(uint16_t)));


        PatchInfo d_patch;
        d_patch.num_faces         = d_counts;
        d_patch.num_edges         = d_counts + 1;
        d_patch.num_vertices      = d_counts + 2;
        d_patch.faces_capacity    = d_counts + 3;
        d_patch.edges_capacity    = d_counts + 4;
        d_patch.vertices_capacity = d_counts + 5;
        d_patch.patch_id          = p;
        d_patch.patch_stash       = PatchStash(true);

        // copy count and capacities
        CUDA_ERROR(cudaMemcpy(d_patch.num_faces,
                              m_h_patches_info[p].num_faces,
                              sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_patch.num_edges,
                              m_h_patches_info[p].num_edges,
                              sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_patch.num_vertices,
                              m_h_patches_info[p].num_vertices,
                              sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_patch.faces_capacity,
                              m_h_patches_info[p].faces_capacity,
                              sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_patch.edges_capacity,
                              m_h_patches_info[p].edges_capacity,
                              sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_patch.vertices_capacity,
                              m_h_patches_info[p].vertices_capacity,
                              sizeof(uint16_t),
                              cudaMemcpyHostToDevice));

        // allocate and copy patch topology to the device
        CUDA_ERROR(cudaMalloc(
            (void**)&d_patch.ev,
            m_h_patches_info[p].edges_capacity[0] * 2 * sizeof(LocalVertexT)));
        CUDA_ERROR(cudaMemcpy(
            d_patch.ev,
            m_h_patches_info[p].ev,
            m_h_patches_info[p].num_edges[0] * 2 * sizeof(LocalVertexT),
            cudaMemcpyHostToDevice));

        CUDA_ERROR(cudaMalloc(
            (void**)&d_patch.fe,
            m_h_patches_info[p].faces_capacity[0] * 3 * sizeof(LocalEdgeT)));
        CUDA_ERROR(cudaMemcpy(
            d_patch.fe,
            m_h_patches_info[p].fe,
            m_h_patches_info[p].num_faces[0] * 3 * sizeof(LocalEdgeT),
            cudaMemcpyHostToDevice));

        // allocate and set bitmask
        auto bitmask = [&](uint32_t*& d_mask,
                           uint32_t*& h_mask,
                           uint32_t   size,
                           auto       predicate) {
            size_t num_bytes = detail::mask_num_bytes(size);
            h_mask           = (uint32_t*)malloc(num_bytes);
            CUDA_ERROR(cudaMalloc((void**)&d_mask, num_bytes));

            for (uint16_t i = 0; i < size; ++i) {
                if (predicate(i)) {
                    detail::bitmask_set_bit(i, h_mask);
                } else {
                    detail::bitmask_clear_bit(i, h_mask);
                }
            }

            CUDA_ERROR(
                cudaMemcpy(d_mask, h_mask, num_bytes, cudaMemcpyHostToDevice));
        };


        // vertices active mask
        bitmask(d_patch.active_mask_v,
                m_h_patches_info[p].active_mask_v,
                m_h_patches_info[p].vertices_capacity[0],
                [&](uint16_t v) {
                    return v < m_h_patches_info[p].num_vertices[0];
                });

        // edges active mask
        bitmask(
            d_patch.active_mask_e,
            m_h_patches_info[p].active_mask_e,
            m_h_patches_info[p].edges_capacity[0],
            [&](uint16_t e) { return e < m_h_patches_info[p].num_edges[0]; });

        // faces active mask
        bitmask(
            d_patch.active_mask_f,
            m_h_patches_info[p].active_mask_f,
            m_h_patches_info[p].faces_capacity[0],
            [&](uint16_t f) { return f < m_h_patches_info[p].num_faces[0]; });

        // vertices owned mask
        bitmask(d_patch.owned_mask_v,
                m_h_patches_info[p].owned_mask_v,
                m_h_patches_info[p].vertices_capacity[0],
                [&](uint16_t v) { return v < m_h_num_owned_v[p]; });

        // edges owned mask
        bitmask(d_patch.owned_mask_e,
                m_h_patches_info[p].owned_mask_e,
                m_h_patches_info[p].edges_capacity[0],
                [&](uint16_t e) { return e < m_h_num_owned_e[p]; });

        // faces owned mask
        bitmask(d_patch.owned_mask_f,
                m_h_patches_info[p].owned_mask_f,
                m_h_patches_info[p].faces_capacity[0],
                [&](uint16_t f) { return f < m_h_num_owned_f[p]; });

        // Populate PatchStash
        auto populate_patch_stash =
            [&](const std::vector<std::vector<uint32_t>>& ltog,
                const std::vector<uint32_t>&              element_patch,
                const std::vector<uint16_t>&              num_owned) {
                const uint16_t num_not_owned = ltog[p].size() - num_owned[p];

                // loop over all not-owned elements to populate PatchStash
                for (uint16_t i = 0; i < num_not_owned; ++i) {
                    uint16_t local_id    = i + num_owned[p];
                    uint32_t global_id   = ltog[p][local_id];
                    uint32_t owner_patch = element_patch[global_id];

                    m_h_patches_info[p].patch_stash.insert_patch(owner_patch);
                }
            };

        populate_patch_stash(
            m_h_patches_ltog_f, m_patcher->get_face_patch(), m_h_num_owned_f);
        populate_patch_stash(
            m_h_patches_ltog_e, m_patcher->get_edge_patch(), m_h_num_owned_e);
        populate_patch_stash(
            m_h_patches_ltog_v, m_patcher->get_vertex_patch(), m_h_num_owned_v);

        CUDA_ERROR(cudaMemcpy(d_patch.patch_stash.m_stash,
                              m_h_patches_info[p].patch_stash.m_stash,
                              PatchStash::stash_size * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));


        // build LPHashtable
        auto build_ht = [&](const std::vector<std::vector<uint32_t>>& ltog,
                            const std::vector<uint32_t>& element_patch,
                            const std::vector<uint16_t>& num_owned,
                            PatchStash&                  stash,
                            LPHashTable&                 h_hashtable,
                            LPHashTable&                 d_hashtable) {
            const uint16_t num_not_owned = ltog[p].size() - num_owned[p];

            const uint16_t capacity = static_cast<uint16_t>(
                static_cast<float>(num_not_owned) / m_lp_hashtable_load_factor);

            h_hashtable = LPHashTable(capacity, false);
            d_hashtable = LPHashTable(capacity, true);

            for (uint16_t i = 0; i < num_not_owned; ++i) {
                uint16_t local_id    = i + num_owned[p];
                uint32_t global_id   = ltog[p][local_id];
                uint32_t owner_patch = element_patch[global_id];

                auto it = std::lower_bound(
                    ltog[owner_patch].begin(),
                    ltog[owner_patch].begin() + num_owned[owner_patch],
                    global_id);

                if (it == ltog[owner_patch].begin() + num_owned[owner_patch]) {
                    RXMESH_ERROR(
                        "rxmesh::build_device can not find the local id of "
                        "{} in patch {}. Maybe this patch does not own "
                        "this mesh element.",
                        global_id,
                        owner_patch);
                } else {
                    uint16_t local_id_in_owner_patch =
                        static_cast<uint16_t>(it - ltog[owner_patch].begin());

                    uint8_t patch_id = stash.find_patch_index(owner_patch);

                    LPPair pair(local_id, local_id_in_owner_patch, patch_id);
                    if (!h_hashtable.insert(pair)) {
                        RXMESH_ERROR(
                            "rxmesh::build_device failed to insert in the "
                            "hashtable. Retry with smaller load factor. Load "
                            "factor used = {}",
                            m_lp_hashtable_load_factor);
                    }
                }
            }

            CUDA_ERROR(cudaMemcpy(d_hashtable.get_table(),
                                  h_hashtable.get_table(),
                                  h_hashtable.num_bytes(),
                                  cudaMemcpyHostToDevice));

            CUDA_ERROR(cudaMemcpy(d_hashtable.get_stash(),
                                  h_hashtable.get_stash(),
                                  LPHashTable::stash_size * sizeof(LPPair),
                                  cudaMemcpyHostToDevice));
        };


        build_ht(m_h_patches_ltog_f,
                 m_patcher->get_face_patch(),
                 m_h_num_owned_f,
                 m_h_patches_info[p].patch_stash,
                 m_h_patches_info[p].lp_f,
                 d_patch.lp_f);

        build_ht(m_h_patches_ltog_e,
                 m_patcher->get_edge_patch(),
                 m_h_num_owned_e,
                 m_h_patches_info[p].patch_stash,
                 m_h_patches_info[p].lp_e,
                 d_patch.lp_e);

        build_ht(m_h_patches_ltog_v,
                 m_patcher->get_vertex_patch(),
                 m_h_num_owned_v,
                 m_h_patches_info[p].patch_stash,
                 m_h_patches_info[p].lp_v,
                 d_patch.lp_v);

        CUDA_ERROR(cudaMemcpy(m_d_patches_info + p,
                              &d_patch,
                              sizeof(PatchInfo),
                              cudaMemcpyHostToDevice));
    }
}

}  // namespace rxmesh
