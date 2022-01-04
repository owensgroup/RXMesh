#include <assert.h>
#include <omp.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>

#include "patcher/patcher.h"
#include "rxmesh/context.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/util/util.h"

namespace rxmesh {
RXMesh::RXMesh(const std::vector<std::vector<uint32_t>>& fv, const bool quite)
    : m_num_edges(0),
      m_num_faces(0),
      m_num_vertices(0),
      m_max_valence(0),
      m_max_edge_incident_faces(0),
      m_max_face_adjacent_faces(0),
      m_max_vertices_per_patch(0),
      m_max_edges_per_patch(0),
      m_max_faces_per_patch(0),
      m_max_not_owned_vertices(0),
      m_max_not_owned_edges(0),
      m_max_not_owned_faces(0),
      m_num_patches(0),
      m_patch_size(512),
      m_is_input_edge_manifold(true),
      m_is_input_closed(true),
      m_quite(quite),
      m_d_patches_info(nullptr),
      m_h_patches_info(nullptr)
{
    // Build everything from scratch including patches
    if (fv.empty()) {
        RXMESH_ERROR(
            "RXMesh::RXMesh input fv is empty. Can not be build RXMesh "
            "properly");
    }
    build(fv);
    build_device();
    calc_max_not_owned_elements();

    // Allocate and copy the context to the gpu
    m_rxmesh_context.init(m_num_edges,
                          m_num_faces,
                          m_num_vertices,
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
        RXMESH_TRACE("max valence = {}", m_max_valence);
        RXMESH_TRACE("max edge incident faces = {}", m_max_edge_incident_faces);
        RXMESH_TRACE("max face adjacent faces = {}", m_max_face_adjacent_faces);
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
        free(m_h_patches_info[p].not_owned_patch_v);
        free(m_h_patches_info[p].not_owned_patch_e);
        free(m_h_patches_info[p].not_owned_patch_f);
        free(m_h_patches_info[p].not_owned_id_v);
        free(m_h_patches_info[p].not_owned_id_e);
        free(m_h_patches_info[p].not_owned_id_f);
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
        GPU_FREE(m_h_patches_info[p].not_owned_patch_v);
        GPU_FREE(m_h_patches_info[p].not_owned_patch_e);
        GPU_FREE(m_h_patches_info[p].not_owned_patch_f);
        GPU_FREE(m_h_patches_info[p].not_owned_id_v);
        GPU_FREE(m_h_patches_info[p].not_owned_id_e);
        GPU_FREE(m_h_patches_info[p].not_owned_id_f);
    }
    GPU_FREE(m_d_patches_info);
    free(m_h_patches_info);
}

void RXMesh::build(const std::vector<std::vector<uint32_t>>& fv)
{
    std::vector<uint32_t>              ff_values;
    std::vector<uint32_t>              ff_offset;
    std::vector<std::vector<uint32_t>> ef;
    build_supporting_structures(fv, ef, ff_offset, ff_values);

    m_patcher = std::make_unique<patcher::Patcher>(m_patch_size,
                                                   ff_offset,
                                                   ff_values,
                                                   fv,
                                                   m_edges_map,
                                                   m_num_vertices,
                                                   m_num_edges,
                                                   m_quite);

    m_num_patches = m_patcher->get_num_patches();

    m_h_patches_ltog_f.resize(m_num_patches);
    m_h_patches_ltog_e.resize(m_num_patches);
    m_h_patches_ltog_v.resize(m_num_patches);
    m_h_num_owned_f.resize(m_num_patches);
    m_h_num_owned_v.resize(m_num_patches);
    m_h_num_owned_e.resize(m_num_patches);
    m_h_patches_fe.resize(m_num_patches);
    m_h_patches_ev.resize(m_num_patches);

#pragma omp parallel for
    for (int p = 0; p < static_cast<int>(m_num_patches); ++p) {
        build_single_patch(fv, p);
    }

    calc_statistics(fv, ef);
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

    // max number of vertices/edges/faces per patch
    m_max_vertices_per_patch = 0;
    m_max_edges_per_patch    = 0;
    m_max_faces_per_patch    = 0;
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        m_max_vertices_per_patch = std::max(
            m_max_vertices_per_patch, uint32_t(m_h_patches_ltog_v[p].size()));
        m_max_edges_per_patch = std::max(
            m_max_edges_per_patch, uint32_t(m_h_patches_ltog_e[p].size()));
        m_max_faces_per_patch = std::max(
            m_max_faces_per_patch, uint32_t(m_h_patches_ltog_f[p].size()));
    }
}

void RXMesh::calc_max_not_owned_elements()
{
    m_max_not_owned_vertices = 0;
    m_max_not_owned_edges    = 0;
    m_max_not_owned_faces    = 0;

    for (int p = 0; p < static_cast<int>(m_num_patches); ++p) {
        m_max_not_owned_vertices =
            std::max(m_max_not_owned_vertices,
                     uint32_t(m_h_patches_info[p].num_vertices -
                              m_h_patches_info[p].num_owned_vertices));

        m_max_not_owned_edges =
            std::max(m_max_not_owned_edges,
                     uint32_t(m_h_patches_info[p].num_edges -
                              m_h_patches_info[p].num_owned_edges));

        m_max_not_owned_faces =
            std::max(m_max_not_owned_faces,
                     uint32_t(m_h_patches_info[p].num_faces -
                              m_h_patches_info[p].num_owned_faces));
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

    m_h_patches_ev[patch_id].resize(patch_num_edges * 2);
    m_h_patches_fe[patch_id].resize(patch_num_faces * 3);

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

                m_h_patches_ev[patch_id][local_edge_id * 2]     = local_v0;
                m_h_patches_ev[patch_id][local_edge_id * 2 + 1] = local_v1;
            }

            // shift local_e to left
            // set the first bit to 1 if (dir ==1)
            local_edge_id = local_edge_id << 1;
            local_edge_id = local_edge_id | (dir & 1);
            m_h_patches_fe[patch_id][local_face_id * 3 + v] = local_edge_id;
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

    m_h_patches_info = (PatchInfo*)malloc(m_num_patches * sizeof(PatchInfo));

#pragma omp parallel for
    for (int p = 0; p < static_cast<int>(m_num_patches); ++p) {
        PatchInfo d_patch;
        d_patch.num_faces          = m_h_patches_ltog_f[p].size();
        d_patch.num_edges          = m_h_patches_ltog_e[p].size();
        d_patch.num_vertices       = m_h_patches_ltog_v[p].size();
        d_patch.num_owned_faces    = m_h_num_owned_f[p];
        d_patch.num_owned_edges    = m_h_num_owned_e[p];
        d_patch.num_owned_vertices = m_h_num_owned_v[p];
        d_patch.patch_id           = p;

        m_h_patches_info[p].num_faces          = m_h_patches_ltog_f[p].size();
        m_h_patches_info[p].num_edges          = m_h_patches_ltog_e[p].size();
        m_h_patches_info[p].num_vertices       = m_h_patches_ltog_v[p].size();
        m_h_patches_info[p].num_owned_faces    = m_h_num_owned_f[p];
        m_h_patches_info[p].num_owned_edges    = m_h_num_owned_e[p];
        m_h_patches_info[p].num_owned_vertices = m_h_num_owned_v[p];
        m_h_patches_info[p].patch_id           = p;


        // allocate and copy patch topology to the device
        CUDA_ERROR(cudaMalloc((void**)&d_patch.ev,
                              d_patch.num_edges * 2 * sizeof(LocalVertexT)));
        CUDA_ERROR(cudaMemcpy(d_patch.ev,
                              m_h_patches_ev[p].data(),
                              d_patch.num_edges * 2 * sizeof(LocalVertexT),
                              cudaMemcpyHostToDevice));
        m_h_patches_info[p].ev =
            reinterpret_cast<LocalVertexT*>(m_h_patches_ev[p].data());

        CUDA_ERROR(cudaMalloc((void**)&d_patch.fe,
                              d_patch.num_faces * 3 * sizeof(LocalEdgeT)));
        CUDA_ERROR(cudaMemcpy(d_patch.fe,
                              m_h_patches_fe[p].data(),
                              d_patch.num_faces * 3 * sizeof(LocalEdgeT),
                              cudaMemcpyHostToDevice));
        m_h_patches_info[p].fe =
            reinterpret_cast<LocalEdgeT*>(m_h_patches_fe[p].data());

        // copy not-owned mesh elements to device

        auto populate_not_owned =
            [p](const std::vector<std::vector<uint32_t>>& ltog,
                const std::vector<uint32_t>&              element_patch,
                const std::vector<uint16_t>&              num_owned,
                auto*&                                    d_not_owned_id,
                uint32_t*&                                d_not_owned_patch,
                auto*&                                    h_not_owned_id,
                uint32_t*&                                h_not_owned_patch) {
                using LocalT = typename std::remove_reference<decltype(
                    *d_not_owned_id)>::type;

                const uint16_t num_not_owned = ltog[p].size() - num_owned[p];

                h_not_owned_id =
                    (LocalT*)malloc(num_not_owned * sizeof(LocalT));
                h_not_owned_patch =
                    (uint32_t*)malloc(num_not_owned * sizeof(uint32_t));

                for (uint16_t i = 0; i < num_not_owned; ++i) {
                    uint16_t local_id     = i + num_owned[p];
                    uint32_t global_id    = ltog[p][local_id];
                    uint32_t owning_patch = element_patch[global_id];
                    h_not_owned_patch[i]  = owning_patch;

                    auto it = std::lower_bound(
                        ltog[owning_patch].begin(),
                        ltog[owning_patch].begin() + num_owned[owning_patch],
                        global_id);

                    if (it ==
                        ltog[owning_patch].begin() + num_owned[owning_patch]) {
                        RXMESH_ERROR(
                            "rxmesh::build_device can not find the local id of "
                            "{} in patch {}. Maybe this patch does not own "
                            "this mesh element.",
                            global_id,
                            owning_patch);
                    } else {
                        h_not_owned_id[i].id = static_cast<uint16_t>(
                            it - ltog[owning_patch].begin());
                    }
                }

                // Copy to device
                CUDA_ERROR(cudaMalloc((void**)&d_not_owned_id,
                                      sizeof(LocalT) * num_not_owned));
                CUDA_ERROR(cudaMemcpy(d_not_owned_id,
                                      h_not_owned_id,
                                      sizeof(LocalT) * num_not_owned,
                                      cudaMemcpyHostToDevice));

                CUDA_ERROR(cudaMalloc((void**)&d_not_owned_patch,
                                      sizeof(uint32_t) * num_not_owned));
                CUDA_ERROR(cudaMemcpy(d_not_owned_patch,
                                      h_not_owned_patch,
                                      sizeof(uint32_t) * num_not_owned,
                                      cudaMemcpyHostToDevice));
            };


        populate_not_owned(m_h_patches_ltog_f,
                           m_patcher->get_face_patch(),
                           m_h_num_owned_f,
                           d_patch.not_owned_id_f,
                           d_patch.not_owned_patch_f,
                           m_h_patches_info[p].not_owned_id_f,
                           m_h_patches_info[p].not_owned_patch_f);

        populate_not_owned(m_h_patches_ltog_e,
                           m_patcher->get_edge_patch(),
                           m_h_num_owned_e,
                           d_patch.not_owned_id_e,
                           d_patch.not_owned_patch_e,
                           m_h_patches_info[p].not_owned_id_e,
                           m_h_patches_info[p].not_owned_patch_e);

        populate_not_owned(m_h_patches_ltog_v,
                           m_patcher->get_vertex_patch(),
                           m_h_num_owned_v,
                           d_patch.not_owned_id_v,
                           d_patch.not_owned_patch_v,
                           m_h_patches_info[p].not_owned_id_v,
                           m_h_patches_info[p].not_owned_patch_v);

        CUDA_ERROR(cudaMemcpy(m_d_patches_info + p,
                              &d_patch,
                              sizeof(PatchInfo),
                              cudaMemcpyHostToDevice));
    }
}

}  // namespace rxmesh
