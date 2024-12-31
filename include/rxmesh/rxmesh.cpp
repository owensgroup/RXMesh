#include <assert.h>
#include <omp.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <set>

#include "patcher/patcher.h"
#include "rxmesh/context.h"
#include "rxmesh/patch_scheduler.cuh"
#include "rxmesh/rxmesh.h"
#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/util.h"

namespace rxmesh {
RXMesh::RXMesh(uint32_t patch_size)
    : m_num_edges(0),
      m_num_faces(0),
      m_num_vertices(0),
      m_max_edge_capacity(0),
      m_max_face_capacity(0),
      m_max_vertex_capacity(0),
      m_input_max_valence(0),
      m_input_max_edge_incident_faces(0),
      m_input_max_face_adjacent_faces(0),
      m_is_input_edge_manifold(true),
      m_is_input_closed(true),
      m_num_patches(0),
      m_max_num_patches(0),
      m_patch_size(patch_size),
      m_max_capacity_lp_v(0),
      m_max_capacity_lp_e(0),
      m_max_capacity_lp_f(0),
      m_max_vertices_per_patch(0),
      m_max_edges_per_patch(0),
      m_max_faces_per_patch(0),
      m_h_vertex_prefix(nullptr),
      m_h_edge_prefix(nullptr),
      m_h_face_prefix(nullptr),
      m_d_vertex_prefix(nullptr),
      m_d_edge_prefix(nullptr),
      m_d_face_prefix(nullptr),
      m_d_patches_info(nullptr),
      m_h_patches_info(nullptr),
      m_capacity_factor(0.f),
      m_lp_hashtable_load_factor(0.f),
      m_patch_alloc_factor(0.f),
      m_topo_memory_mega_bytes(0.0),
      m_num_colors(0)
{
}

void RXMesh::init(const std::vector<std::vector<uint32_t>>& fv,
                  const std::string                         patcher_file,
                  const float                               capacity_factor,
                  const float                               patch_alloc_factor,
                  const float lp_hashtable_load_factor)
{
    m_topo_memory_mega_bytes   = 0;
    m_capacity_factor          = capacity_factor;
    m_lp_hashtable_load_factor = lp_hashtable_load_factor;
    m_patch_alloc_factor       = patch_alloc_factor;

    // Build everything from scratch including patches
    if (fv.empty()) {
        RXMESH_ERROR(
            "RXMesh::init input fv is empty. Can not build RXMesh properly");
    }
    if (m_capacity_factor < 1.0) {
        RXMESH_ERROR("RXMesh::init capacity factor should be at least one");
    }
    if (m_patch_alloc_factor < 1.0) {
        RXMESH_ERROR(
            "RXMesh::init patch allocation factor should be at least one");
    }
    if (m_lp_hashtable_load_factor > 1.0) {
        RXMESH_ERROR(
            "RXMesh::init hashtable load factor should be less than 1");
    }

    m_timers.add("LPHashTable");
    m_timers.add("ht.insert");
    m_timers.add("lower_bound");
    m_timers.add("bitmask");
    m_timers.add("buildHT");
    m_timers.add("cudaMalloc");
    m_timers.add("malloc");

    m_timers.add("build");
    m_timers.start("build");
    build(fv, patcher_file);
    m_timers.stop("build");
    RXMESH_INFO("build time = {} (ms)", m_timers.elapsed_millis("build"));

    m_timers.add("populate_patch_stash");
    m_timers.start("populate_patch_stash");
    populate_patch_stash();
    m_timers.stop("populate_patch_stash");
    RXMESH_INFO("populate_patch_stash time = {} (ms)",
                m_timers.elapsed_millis("populate_patch_stash"));

    m_timers.add("coloring");
    m_timers.start("coloring");
    patch_graph_coloring();
    m_timers.stop("coloring");
    RXMESH_INFO("Num colors = {}", m_num_colors);
    RXMESH_INFO("patch graph coloring time = {} (ms)",
                m_timers.elapsed_millis("coloring"));


    m_timers.add("build_device");
    m_timers.start("build_device");
    build_device();
    m_timers.stop("build_device");
    RXMESH_INFO("build_device time = {} (ms)",
                m_timers.elapsed_millis("build_device"));


    m_timers.add("PatchScheduler");
    m_timers.start("PatchScheduler");
    PatchScheduler sch;
    sch.init(get_max_num_patches());
    m_topo_memory_mega_bytes +=
        BYTES_TO_MEGABYTES(sizeof(uint32_t) * get_max_num_patches());
    sch.refill(get_num_patches());
    m_timers.stop("PatchScheduler");
    RXMESH_INFO("PatchScheduler time = {} (ms)",
                m_timers.elapsed_millis("PatchScheduler"));


    m_timers.add("allocate_extra_patches");
    m_timers.start("allocate_extra_patches");
    // Allocate  extra patches
    allocate_extra_patches();
    m_timers.stop("allocate_extra_patches");
    RXMESH_INFO("allocate_extra_patches time = {} (ms)",
                m_timers.elapsed_millis("allocate_extra_patches"));

    m_timers.add("context.init");
    m_timers.start("context.init");
    // Allocate and copy the context to the gpu
    m_rxmesh_context.init(m_num_vertices,
                          m_num_edges,
                          m_num_faces,
                          m_max_vertices_per_patch,
                          m_max_edges_per_patch,
                          m_max_faces_per_patch,
                          get_num_patches(),
                          get_max_num_patches(),
                          m_capacity_factor,
                          m_d_vertex_prefix,
                          m_d_edge_prefix,
                          m_d_face_prefix,
                          m_h_vertex_prefix,
                          m_h_edge_prefix,
                          m_h_face_prefix,
                          max_lp_hashtable_capacity<LocalVertexT>(),
                          max_lp_hashtable_capacity<LocalEdgeT>(),
                          max_lp_hashtable_capacity<LocalFaceT>(),
                          m_d_patches_info,
                          sch);
    m_timers.stop("context.init");
    RXMESH_INFO("context.init time = {} (ms)",
                m_timers.elapsed_millis("context.init"));


    RXMESH_TRACE("#Vertices = {}, #Faces= {}, #Edges= {}",
                 m_num_vertices,
                 m_num_faces,
                 m_num_edges);
    RXMESH_TRACE("Input is{} edge manifold",
                 ((m_is_input_edge_manifold) ? "" : " Not"));
    RXMESH_TRACE("Input is{} closed", ((m_is_input_closed) ? "" : " Not"));
    RXMESH_TRACE("Input max valence = {}", m_input_max_valence);
    RXMESH_TRACE("max edge incident faces = {}",
                 m_input_max_edge_incident_faces);
    RXMESH_TRACE("max face adjacent faces = {}",
                 m_input_max_face_adjacent_faces);
    RXMESH_TRACE("per-patch maximum face count = {}", m_max_faces_per_patch);
    RXMESH_TRACE("per-patch maximum edge count = {}", m_max_edges_per_patch);
    RXMESH_TRACE("per-patch maximum vertex count = {}",
                 m_max_vertices_per_patch);

    RXMESH_INFO("cudaMalloc time = {} (ms)",
                m_timers.elapsed_millis("cudaMalloc"));

    RXMESH_INFO("malloc time = {} (ms)", m_timers.elapsed_millis("malloc"));

    RXMESH_INFO("buildHT time = {} (ms)", m_timers.elapsed_millis("buildHT"));
    RXMESH_INFO("bitmask time = {} (ms)", m_timers.elapsed_millis("bitmask"));
    RXMESH_INFO("lower_bound time = {} (ms)",
                m_timers.elapsed_millis("lower_bound"));
    RXMESH_INFO("ht.insert time = {} (ms)",
                m_timers.elapsed_millis("ht.insert"));
    RXMESH_INFO("LPHashTable time = {} (ms)",
                m_timers.elapsed_millis("LPHashTable"));
}

RXMesh::~RXMesh()
{
    m_rxmesh_context.m_patch_scheduler.free();

    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        free(m_h_patches_info[p].active_mask_v);
        free(m_h_patches_info[p].active_mask_e);
        free(m_h_patches_info[p].active_mask_f);
        free(m_h_patches_info[p].owned_mask_v);
        free(m_h_patches_info[p].owned_mask_e);
        free(m_h_patches_info[p].owned_mask_f);
        free(m_h_patches_info[p].num_faces);
        free(m_h_patches_info[p].dirty);
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
                          get_num_patches() * sizeof(PatchInfo),
                          cudaMemcpyDeviceToHost));

    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        GPU_FREE(m_h_patches_info[p].active_mask_v);
        GPU_FREE(m_h_patches_info[p].active_mask_e);
        GPU_FREE(m_h_patches_info[p].active_mask_f);
        GPU_FREE(m_h_patches_info[p].owned_mask_v);
        GPU_FREE(m_h_patches_info[p].owned_mask_e);
        GPU_FREE(m_h_patches_info[p].owned_mask_f);
        GPU_FREE(m_h_patches_info[p].ev);
        GPU_FREE(m_h_patches_info[p].fe);
        GPU_FREE(m_h_patches_info[p].num_faces);
        GPU_FREE(m_h_patches_info[p].dirty);
        m_h_patches_info[p].lp_v.free();
        m_h_patches_info[p].lp_e.free();
        m_h_patches_info[p].lp_f.free();
        m_h_patches_info[p].patch_stash.free();
        m_h_patches_info[p].lock.free();
    }
    GPU_FREE(m_d_patches_info);
    free(m_h_patches_info);
    m_rxmesh_context.release();

    GPU_FREE(m_d_vertex_prefix);
    GPU_FREE(m_d_edge_prefix);
    GPU_FREE(m_d_face_prefix);

    free(m_h_vertex_prefix);
    free(m_h_edge_prefix);
    free(m_h_face_prefix);
}

void RXMesh::build(const std::vector<std::vector<uint32_t>>& fv,
                   const std::string                         patcher_file)
{
    std::vector<uint32_t>              ff_values;
    std::vector<uint32_t>              ff_offset;
    std::vector<std::vector<uint32_t>> ef;
    std::vector<std::vector<uint32_t>> ev;

    m_max_capacity_lp_v = 0;
    m_max_capacity_lp_e = 0;
    m_max_capacity_lp_f = 0;

    build_supporting_structures(fv, ev, ef, ff_offset, ff_values);

    if (!patcher_file.empty()) {
        if (!std::filesystem::exists(patcher_file)) {
            RXMESH_ERROR(
                "RXMesh::build patch file {} does not exit. Building unique "
                "patches.",
                patcher_file);
            m_patcher = std::make_unique<patcher::Patcher>(m_patch_size,
                                                           ff_offset,
                                                           ff_values,
                                                           fv,
                                                           m_edges_map,
                                                           m_num_vertices,
                                                           m_num_edges,
                                                           false);
        } else {
            m_patcher = std::make_unique<patcher::Patcher>(patcher_file);
        }
    } else {
        m_patcher = std::make_unique<patcher::Patcher>(m_patch_size,
                                                       ff_offset,
                                                       ff_values,
                                                       fv,
                                                       m_edges_map,
                                                       m_num_vertices,
                                                       m_num_edges,
                                                       false);
    }


    m_num_patches     = m_patcher->get_num_patches();
    m_max_num_patches = static_cast<uint32_t>(
        std::ceil(m_patch_alloc_factor * static_cast<float>(m_num_patches)));

    m_h_patches_info =
        (PatchInfo*)malloc(get_max_num_patches() * sizeof(PatchInfo));
    m_h_patches_ltog_f.resize(get_num_patches());
    m_h_patches_ltog_e.resize(get_num_patches());
    m_h_patches_ltog_v.resize(get_num_patches());
    m_h_num_owned_f.resize(get_max_num_patches(), 0);
    m_h_num_owned_v.resize(get_max_num_patches(), 0);
    m_h_num_owned_e.resize(get_max_num_patches(), 0);

#pragma omp parallel for
    for (int p = 0; p < static_cast<int>(get_num_patches()); ++p) {
        build_single_patch_ltog(fv, ev, p);
    }

    // calc max elements for use in build_device (which populates
    // m_h_patches_info and thus we can not use calc_max_elements now)
    m_max_vertices_per_patch = 0;
    m_max_edges_per_patch    = 0;
    m_max_faces_per_patch    = 0;
    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        m_max_vertices_per_patch =
            std::max(m_max_vertices_per_patch,
                     static_cast<uint32_t>(m_h_patches_ltog_v[p].size()));
        m_max_edges_per_patch =
            std::max(m_max_edges_per_patch,
                     static_cast<uint32_t>(m_h_patches_ltog_e[p].size()));
        m_max_faces_per_patch =
            std::max(m_max_faces_per_patch,
                     static_cast<uint32_t>(m_h_patches_ltog_f[p].size()));
    }

    m_max_vertex_capacity = static_cast<uint16_t>(std::ceil(
        m_capacity_factor * static_cast<float>(m_max_vertices_per_patch)));

    m_max_edge_capacity = static_cast<uint16_t>(std::ceil(
        m_capacity_factor * static_cast<float>(m_max_edges_per_patch)));

    m_max_face_capacity = static_cast<uint16_t>(std::ceil(
        m_capacity_factor * static_cast<float>(m_max_faces_per_patch)));

#pragma omp parallel for
    for (int p = 0; p < static_cast<int>(get_num_patches()); ++p) {
        build_single_patch_topology(fv, p);
    }

    const uint32_t patches_1_bytes =
        (get_max_num_patches() + 1) * sizeof(uint32_t);

    m_h_vertex_prefix = (uint32_t*)malloc(patches_1_bytes);
    m_h_edge_prefix   = (uint32_t*)malloc(patches_1_bytes);
    m_h_face_prefix   = (uint32_t*)malloc(patches_1_bytes);

    memset(m_h_vertex_prefix, 0, patches_1_bytes);
    memset(m_h_edge_prefix, 0, patches_1_bytes);
    memset(m_h_face_prefix, 0, patches_1_bytes);

    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        m_h_vertex_prefix[p + 1] = m_h_vertex_prefix[p] + m_h_num_owned_v[p];
        m_h_edge_prefix[p + 1]   = m_h_edge_prefix[p] + m_h_num_owned_e[p];
        m_h_face_prefix[p + 1]   = m_h_face_prefix[p] + m_h_num_owned_f[p];
    }

    m_timers.start("cudaMalloc");
    CUDA_ERROR(cudaMalloc((void**)&m_d_vertex_prefix, patches_1_bytes));
    // m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(patches_1_bytes);
    CUDA_ERROR(cudaMalloc((void**)&m_d_edge_prefix, patches_1_bytes));
    // m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(patches_1_bytes);
    CUDA_ERROR(cudaMalloc((void**)&m_d_face_prefix, patches_1_bytes));
    // m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(patches_1_bytes);
    m_timers.stop("cudaMalloc");


    CUDA_ERROR(cudaMemcpy(m_d_vertex_prefix,
                          m_h_vertex_prefix,
                          patches_1_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_edge_prefix,
                          m_h_edge_prefix,
                          patches_1_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_face_prefix,
                          m_h_face_prefix,
                          patches_1_bytes,
                          cudaMemcpyHostToDevice));

    calc_input_statistics(fv, ef);
}

void RXMesh::build_supporting_structures(
    const std::vector<std::vector<uint32_t>>& fv,
    std::vector<std::vector<uint32_t>>&       ev,
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
    ev.reserve(2 * reserve_size);

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

                std::vector<uint32_t> evv = {v0, v1};
                ev.push_back(evv);

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

    ff_offset.resize(m_num_faces + 1);
    std::exclusive_scan(ff_size.begin(), ff_size.end(), ff_offset.begin(), 0);
    ff_offset[m_num_faces] =
        ff_offset[m_num_faces - 1] + ff_size[m_num_faces - 1];
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
                f0_offset += ff_offset[f0];
                f1_offset += ff_offset[f1];

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
    m_max_vertices_per_patch = 0;
    m_max_edges_per_patch    = 0;
    m_max_faces_per_patch    = 0;


    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        m_max_vertices_per_patch =
            std::max(m_max_vertices_per_patch,
                     uint32_t(m_h_patches_info[p].num_vertices[0]));
        m_max_edges_per_patch = std::max(
            m_max_edges_per_patch, uint32_t(m_h_patches_info[p].num_edges[0]));
        m_max_faces_per_patch = std::max(
            m_max_faces_per_patch, uint32_t(m_h_patches_info[p].num_faces[0]));
    }
}

void RXMesh::build_single_patch_ltog(
    const std::vector<std::vector<uint32_t>>& fv,
    const std::vector<std::vector<uint32_t>>& ev,
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
    m_h_patches_ltog_v[patch_id].reserve(3 * total_patch_num_faces);
    m_h_patches_ltog_e[patch_id].reserve(3 * total_patch_num_faces);

    std::vector<bool> is_vertex_added(m_num_vertices, false);
    std::vector<bool> is_edge_added(m_num_edges, false);

    // add faces owned by this patch
    auto add_new_face = [&](uint32_t global_face_id, uint16_t local_face_id) {
        m_h_patches_ltog_f[patch_id][local_face_id] = global_face_id;

        for (uint32_t v = 0; v < 3; ++v) {
            uint32_t v0 = fv[global_face_id][v];
            uint32_t v1 = fv[global_face_id][(v + 1) % 3];

            uint32_t edge_id = get_edge_id(v0, v1);

            if (!is_vertex_added[v0]) {
                is_vertex_added[v0] = true;

                m_h_patches_ltog_v[patch_id].push_back(v0);
            }

            if (!is_edge_added[edge_id]) {
                is_edge_added[edge_id] = true;

                m_h_patches_ltog_e[patch_id].push_back(edge_id);
            }
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


    // add edges owned by this patch
    for (uint32_t e = 0; e < m_num_edges; ++e) {
        // if the edge is owned by this patch but it was not added yet
        if (m_patcher->get_edge_patch_id(e) == patch_id && !is_edge_added[e]) {
            m_h_patches_ltog_e[patch_id].push_back(e);
            for (uint32_t i = 0; i < 2; ++i) {
                uint32_t v = ev[e][i];
                if (!is_vertex_added[v]) {
                    m_h_patches_ltog_v[patch_id].push_back(v);
                }
            }
        }
    }

    // add vertices owned by this patch
    for (uint32_t v = 0; v < m_num_vertices; ++v) {
        // if the edge is owned by this patch but it was not added yet
        if (m_patcher->get_vertex_patch_id(v) == patch_id &&
            !is_vertex_added[v]) {
            m_h_patches_ltog_v[patch_id].push_back(v);
        }
    }

    auto create_unique_mapping = [&](std::vector<uint32_t>&       ltog_map,
                                     const std::vector<uint32_t>& patch) {
        std::sort(ltog_map.begin(), ltog_map.end());
#ifndef NDEBUG
        auto unique_end = std::unique(ltog_map.begin(), ltog_map.end());
        assert(unique_end == ltog_map.end());
#endif

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
    
    const uint32_t edges_cap = m_max_edge_capacity;

    const uint32_t faces_cap = m_max_face_capacity;

    m_h_patches_info[patch_id].ev =
        (LocalVertexT*)malloc(edges_cap * 2 * sizeof(LocalVertexT));
    m_h_patches_info[patch_id].fe =
        (LocalEdgeT*)malloc(faces_cap * 3 * sizeof(LocalEdgeT));

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

const VertexHandle RXMesh::map_to_local_vertex(uint32_t i) const
{
    auto pl = map_to_local<VertexHandle>(i, m_h_vertex_prefix);
    return {pl.first, pl.second};
}

const EdgeHandle RXMesh::map_to_local_edge(uint32_t i) const
{
    auto pl = map_to_local<EdgeHandle>(i, m_h_edge_prefix);
    return {pl.first, pl.second};
}

const FaceHandle RXMesh::map_to_local_face(uint32_t i) const
{
    auto pl = map_to_local<FaceHandle>(i, m_h_face_prefix);
    return {pl.first, pl.second};
}


template <typename HandleT>
const std::pair<uint32_t, uint16_t> RXMesh::map_to_local(
    const uint32_t  i,
    const uint32_t* element_prefix) const
{
    const auto end = element_prefix + get_num_patches() + 1;

    auto p = std::lower_bound(
        element_prefix, end, i, [](int a, int b) { return a <= b; });
    if (p == end) {
        RXMESH_ERROR(
            "RXMeshStatic::map_to_local can not its patch. Input is out of "
            "range!");
    }
    p -= 1;
    uint32_t patch_id = std::distance(element_prefix, p);
    uint32_t prefix   = i - *p;
    uint16_t local_id = 0;
    uint16_t num_elements =
        *(m_h_patches_info[patch_id].template get_num_elements<HandleT>());
    for (uint16_t l = 0; l < num_elements; ++l) {
        if (m_h_patches_info[patch_id].is_owned(typename HandleT::LocalT(l)) &&
            !m_h_patches_info[patch_id].is_deleted(
                typename HandleT::LocalT(l))) {
            if (local_id == prefix) {
                local_id = l;
                break;
            }
            local_id++;
        }
    }
    return {patch_id, local_id};
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

uint16_t RXMesh::get_per_patch_max_vertex_capacity() const
{
    return m_max_vertex_capacity;
}
uint16_t RXMesh::get_per_patch_max_edge_capacity() const
{
    return m_max_edge_capacity;
}
uint16_t RXMesh::get_per_patch_max_face_capacity() const
{
    return m_max_face_capacity;
}

void RXMesh::populate_patch_stash()
{
    auto populate_patch_stash = [&](uint32_t                     p,
                                    const std::vector<uint32_t>& ltog,
                                    const std::vector<uint32_t>& element_patch,
                                    const uint16_t&              num_owned) {
        const uint16_t num_not_owned = ltog.size() - num_owned;

        // loop over all not-owned elements to populate PatchStash
        for (uint16_t i = 0; i < num_not_owned; ++i) {
            uint16_t local_id    = i + num_owned;
            uint32_t global_id   = ltog[local_id];
            uint32_t owner_patch = element_patch[global_id];

            m_h_patches_info[p].patch_stash.insert_patch(owner_patch);
        }
    };

    // #pragma omp parallel for
    for (int p = 0; p < static_cast<int>(get_num_patches()); ++p) {
        m_h_patches_info[p].patch_stash = PatchStash(false);

        populate_patch_stash(p,
                             m_h_patches_ltog_v[p],
                             m_patcher->get_vertex_patch(),
                             m_h_num_owned_v[p]);
        populate_patch_stash(p,
                             m_h_patches_ltog_e[p],
                             m_patcher->get_edge_patch(),
                             m_h_num_owned_e[p]);
        populate_patch_stash(p,
                             m_h_patches_ltog_f[p],
                             m_patcher->get_face_patch(),
                             m_h_num_owned_f[p]);
    }

    // #pragma omp parallel for
    for (int p = get_num_patches(); p < static_cast<int>(get_max_num_patches());
         ++p) {
        m_h_patches_info[p].patch_stash = PatchStash(false);
    }
}

void RXMesh::build_device()
{
    m_timers.start("cudaMalloc");
    CUDA_ERROR(cudaMalloc((void**)&m_d_patches_info,
                          get_max_num_patches() * sizeof(PatchInfo)));
    m_timers.stop("cudaMalloc");

    m_topo_memory_mega_bytes +=
        BYTES_TO_MEGABYTES(get_max_num_patches() * sizeof(PatchInfo));


    // #pragma omp parallel for
    for (int p = 0; p < static_cast<int>(get_num_patches()); ++p) {

        const uint16_t p_num_vertices =
            static_cast<uint16_t>(m_h_patches_ltog_v[p].size());
        const uint16_t p_num_edges =
            static_cast<uint16_t>(m_h_patches_ltog_e[p].size());
        const uint16_t p_num_faces =
            static_cast<uint16_t>(m_h_patches_ltog_f[p].size());

        build_device_single_patch(p,
                                  p_num_vertices,
                                  p_num_edges,
                                  p_num_faces,
                                  get_per_patch_max_vertex_capacity(),
                                  get_per_patch_max_edge_capacity(),
                                  get_per_patch_max_face_capacity(),
                                  m_h_num_owned_v[p],
                                  m_h_num_owned_e[p],
                                  m_h_num_owned_f[p],
                                  m_h_patches_ltog_v[p],
                                  m_h_patches_ltog_e[p],
                                  m_h_patches_ltog_f[p],
                                  m_h_patches_info[p],
                                  m_d_patches_info[p]);
    }


    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        m_max_capacity_lp_v = std::max(m_max_capacity_lp_v,
                                       m_h_patches_info[p].lp_v.get_capacity());

        m_max_capacity_lp_e = std::max(m_max_capacity_lp_e,
                                       m_h_patches_info[p].lp_e.get_capacity());

        m_max_capacity_lp_f = std::max(m_max_capacity_lp_f,
                                       m_h_patches_info[p].lp_f.get_capacity());
    }

    // make sure that if a patch stash of patch p has patch q, then q's patch
    // stash should have p in it
    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        for (uint8_t p_sh = 0; p_sh < PatchStash::stash_size; ++p_sh) {
            uint32_t q = m_h_patches_info[p].patch_stash.get_patch(p_sh);
            if (q != INVALID32) {
                bool found = false;
                for (uint8_t q_sh = 0; q_sh < PatchStash::stash_size; ++q_sh) {
                    if (m_h_patches_info[q].patch_stash.get_patch(q_sh) == p) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    m_h_patches_info[q].patch_stash.insert_patch(p);
                }
            }
        }
    }
}

void RXMesh::build_device_single_patch(const uint32_t patch_id,
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
                                       PatchInfo& h_patch_info,
                                       PatchInfo& d_patch_info)
{


    m_timers.start("malloc");
    uint16_t* h_counts = (uint16_t*)malloc(6 * sizeof(uint16_t));
    m_timers.stop("malloc");

    h_patch_info.num_faces            = h_counts;
    h_patch_info.num_faces[0]         = p_num_faces;
    h_patch_info.num_edges            = h_counts + 1;
    h_patch_info.num_edges[0]         = p_num_edges;
    h_patch_info.num_vertices         = h_counts + 2;
    h_patch_info.num_vertices[0]      = p_num_vertices;
    h_patch_info.faces_capacity       = h_counts + 3;
    h_patch_info.faces_capacity[0]    = p_faces_capacity;
    h_patch_info.edges_capacity       = h_counts + 4;
    h_patch_info.edges_capacity[0]    = p_edges_capacity;
    h_patch_info.vertices_capacity    = h_counts + 5;
    h_patch_info.vertices_capacity[0] = p_vertices_capacity;
    h_patch_info.patch_id             = patch_id;
    h_patch_info.dirty                = (int*)malloc(sizeof(int));
    h_patch_info.dirty[0]             = 0;
    h_patch_info.child_id             = INVALID32;
    h_patch_info.should_slice         = false;


    uint16_t* d_counts;

    m_timers.start("cudaMalloc");
    CUDA_ERROR(cudaMalloc((void**)&d_counts, 6 * sizeof(uint16_t)));
    m_timers.stop("cudaMalloc");


    m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(6 * sizeof(uint16_t));

    PatchInfo d_patch;
    d_patch.num_faces         = d_counts;
    d_patch.num_edges         = d_counts + 1;
    d_patch.num_vertices      = d_counts + 2;
    d_patch.faces_capacity    = d_counts + 3;
    d_patch.edges_capacity    = d_counts + 4;
    d_patch.vertices_capacity = d_counts + 5;
    d_patch.patch_id          = patch_id;
    d_patch.color             = h_patch_info.color;
    d_patch.patch_stash       = PatchStash(true);
    d_patch.lock.init();
    d_patch.child_id     = INVALID32;
    d_patch.should_slice = false;

    m_topo_memory_mega_bytes +=
        BYTES_TO_MEGABYTES(PatchStash::stash_size * sizeof(uint32_t));

    // copy count and capacities
    CUDA_ERROR(cudaMemcpy(d_patch.num_faces,
                          h_patch_info.num_faces,
                          sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_patch.num_edges,
                          h_patch_info.num_edges,
                          sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_patch.num_vertices,
                          h_patch_info.num_vertices,
                          sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_patch.faces_capacity,
                          h_patch_info.faces_capacity,
                          sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_patch.edges_capacity,
                          h_patch_info.edges_capacity,
                          sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_patch.vertices_capacity,
                          h_patch_info.vertices_capacity,
                          sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    // allocate and copy patch topology to the device
    // we realloc the host h_patch_info EV and FE to ensure that both host and
    // device has the same capacity
    m_timers.start("cudaMalloc");
    CUDA_ERROR(cudaMalloc((void**)&d_patch.ev,
                          p_edges_capacity * 2 * sizeof(LocalVertexT)));
    m_timers.stop("cudaMalloc");


    m_topo_memory_mega_bytes +=
        BYTES_TO_MEGABYTES(p_edges_capacity * 2 * sizeof(LocalVertexT));
    h_patch_info.ev = (LocalVertexT*)realloc(
        h_patch_info.ev, p_edges_capacity * 2 * sizeof(LocalVertexT));

    if (p_num_edges > 0) {
        CUDA_ERROR(cudaMemcpy(d_patch.ev,
                              h_patch_info.ev,
                              p_num_edges * 2 * sizeof(LocalVertexT),
                              cudaMemcpyHostToDevice));
    }

    m_timers.start("cudaMalloc");
    CUDA_ERROR(cudaMalloc((void**)&d_patch.fe,
                          p_faces_capacity * 3 * sizeof(LocalEdgeT)));
    m_timers.stop("cudaMalloc");


    m_topo_memory_mega_bytes +=
        BYTES_TO_MEGABYTES(p_faces_capacity * 3 * sizeof(LocalEdgeT));
    h_patch_info.fe = (LocalEdgeT*)realloc(
        h_patch_info.fe, p_faces_capacity * 3 * sizeof(LocalEdgeT));

    if (p_num_faces > 0) {
        CUDA_ERROR(cudaMemcpy(d_patch.fe,
                              h_patch_info.fe,
                              p_num_faces * 3 * sizeof(LocalEdgeT),
                              cudaMemcpyHostToDevice));
    }

    m_timers.start("cudaMalloc");
    CUDA_ERROR(cudaMalloc((void**)&d_patch.dirty, sizeof(int)));
    m_timers.stop("cudaMalloc");


    m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(sizeof(int));
    CUDA_ERROR(cudaMemset(d_patch.dirty, 0, sizeof(int)));


    // allocate and set bitmask
    auto bitmask = [&](uint32_t*& d_mask,
                       uint32_t*& h_mask,
                       uint32_t   capacity,
                       auto       predicate) {
        m_timers.start("bitmask");

        size_t num_bytes = detail::mask_num_bytes(capacity);

        m_timers.start("malloc");
        h_mask = (uint32_t*)malloc(num_bytes);
        m_timers.stop("malloc");

        m_timers.start("cudaMalloc");
        CUDA_ERROR(cudaMalloc((void**)&d_mask, num_bytes));
        m_timers.stop("cudaMalloc");


        m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(num_bytes);

        for (uint16_t i = 0; i < capacity; ++i) {
            if (predicate(i)) {
                detail::bitmask_set_bit(i, h_mask);
            } else {
                detail::bitmask_clear_bit(i, h_mask);
            }
        }

        CUDA_ERROR(
            cudaMemcpy(d_mask, h_mask, num_bytes, cudaMemcpyHostToDevice));

        m_timers.stop("bitmask");
    };


    // vertices active mask
    bitmask(d_patch.active_mask_v,
            h_patch_info.active_mask_v,
            p_vertices_capacity,
            [&](uint16_t v) { return v < p_num_vertices; });

    // edges active mask
    bitmask(d_patch.active_mask_e,
            h_patch_info.active_mask_e,
            p_edges_capacity,
            [&](uint16_t e) { return e < p_num_edges; });

    // faces active mask
    bitmask(d_patch.active_mask_f,
            h_patch_info.active_mask_f,
            p_faces_capacity,
            [&](uint16_t f) { return f < p_num_faces; });

    // vertices owned mask
    bitmask(d_patch.owned_mask_v,
            h_patch_info.owned_mask_v,
            p_vertices_capacity,
            [&](uint16_t v) { return v < p_num_owned_vertices; });

    // edges owned mask
    bitmask(d_patch.owned_mask_e,
            h_patch_info.owned_mask_e,
            p_edges_capacity,
            [&](uint16_t e) { return e < p_num_owned_edges; });

    // faces owned mask
    bitmask(d_patch.owned_mask_f,
            h_patch_info.owned_mask_f,
            p_faces_capacity,
            [&](uint16_t f) { return f < p_num_owned_faces; });


    // Copy PatchStash
    if (patch_id != INVALID32) {
        CUDA_ERROR(cudaMemcpy(d_patch.patch_stash.m_stash,
                              h_patch_info.patch_stash.m_stash,
                              PatchStash::stash_size * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }


    // build LPHashtable
    auto build_ht = [&](const std::vector<std::vector<uint32_t>>& ltog,
                        const std::vector<uint32_t>&              p_ltog,
                        const std::vector<uint32_t>&              element_patch,
                        const std::vector<uint16_t>&              num_owned,
                        const uint16_t                            num_elements,
                        const uint16_t num_owned_elements,
                        const uint16_t cap,
                        PatchStash&    stash,
                        LPHashTable&   h_hashtable,
                        LPHashTable&   d_hashtable) {
        m_timers.start("buildHT");

        const uint16_t num_not_owned = num_elements - num_owned_elements;

        uint16_t capacity = cap;

        if (patch_id != INVALID32) {
            capacity = static_cast<uint16_t>(std::ceil(
                m_capacity_factor * static_cast<float>(num_not_owned) /
                m_lp_hashtable_load_factor));
        }

        m_timers.start("LPHashTable");
        h_hashtable = LPHashTable(capacity, false);
        d_hashtable = LPHashTable(capacity, true);
        m_timers.stop("LPHashTable");

        m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(d_hashtable.num_bytes());
        m_topo_memory_mega_bytes +=
            BYTES_TO_MEGABYTES(LPHashTable::stash_size * sizeof(LPPair));

        for (uint16_t i = 0; i < num_not_owned; ++i) {
            uint16_t local_id    = i + num_owned_elements;
            uint32_t global_id   = p_ltog[local_id];
            uint32_t owner_patch = element_patch[global_id];

            m_timers.start("lower_bound");
            auto it = std::lower_bound(
                ltog[owner_patch].begin(),
                ltog[owner_patch].begin() + num_owned[owner_patch],
                global_id);
            m_timers.stop("lower_bound");

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

                uint8_t owner_st = stash.find_patch_index(owner_patch);

                m_timers.start("ht.insert");
                LPPair pair(local_id, local_id_in_owner_patch, owner_st);
                if (!h_hashtable.insert(pair, nullptr, nullptr)) {
                    RXMESH_ERROR(
                        "rxmesh::build_device failed to insert in the "
                        "hashtable. Retry with smaller load factor. Load "
                        "factor used = {}",
                        m_lp_hashtable_load_factor);
                }
                m_timers.stop("ht.insert");
            }
        }

        d_hashtable.move(h_hashtable);

        m_timers.stop("buildHT");
    };

    const uint16_t lp_cap_v = max_lp_hashtable_capacity<LocalVertexT>();
    build_ht(m_h_patches_ltog_v,
             ltog_v,
             m_patcher->get_vertex_patch(),
             m_h_num_owned_v,
             p_num_vertices,
             p_num_owned_vertices,
             lp_cap_v,
             h_patch_info.patch_stash,
             h_patch_info.lp_v,
             d_patch.lp_v);

    const uint16_t lp_cap_e = max_lp_hashtable_capacity<LocalEdgeT>();
    build_ht(m_h_patches_ltog_e,
             ltog_e,
             m_patcher->get_edge_patch(),
             m_h_num_owned_e,
             p_num_edges,
             p_num_owned_edges,
             lp_cap_e,
             h_patch_info.patch_stash,
             h_patch_info.lp_e,
             d_patch.lp_e);

    const uint16_t lp_cap_f = max_lp_hashtable_capacity<LocalFaceT>();
    build_ht(m_h_patches_ltog_f,
             ltog_f,
             m_patcher->get_face_patch(),
             m_h_num_owned_f,
             p_num_faces,
             p_num_owned_faces,
             lp_cap_f,
             h_patch_info.patch_stash,
             h_patch_info.lp_f,
             d_patch.lp_f);


    CUDA_ERROR(cudaMemcpy(
        &d_patch_info, &d_patch, sizeof(PatchInfo), cudaMemcpyHostToDevice));
}

void RXMesh::allocate_extra_patches()
{

    const uint16_t p_vertices_capacity = get_per_patch_max_vertex_capacity();
    const uint16_t p_edges_capacity    = get_per_patch_max_edge_capacity();
    const uint16_t p_faces_capacity    = get_per_patch_max_face_capacity();

    // #pragma omp parallel for
    for (int p = get_num_patches(); p < static_cast<int>(get_max_num_patches());
         ++p) {

        const uint16_t p_num_vertices = 0;
        const uint16_t p_num_edges    = 0;
        const uint16_t p_num_faces    = 0;

        m_timers.start("malloc");
        m_h_patches_info[p].ev =
            (LocalVertexT*)malloc(2 * p_edges_capacity * sizeof(LocalVertexT));
        m_h_patches_info[p].fe =
            (LocalEdgeT*)malloc(3 * p_faces_capacity * sizeof(LocalEdgeT));
        m_timers.stop("malloc");

        build_device_single_patch(INVALID32,
                                  p_num_vertices,
                                  p_num_edges,
                                  p_num_faces,
                                  p_vertices_capacity,
                                  p_edges_capacity,
                                  p_faces_capacity,
                                  m_h_num_owned_v[p],
                                  m_h_num_owned_e[p],
                                  m_h_num_owned_f[p],
                                  m_h_patches_ltog_v[0],
                                  m_h_patches_ltog_e[0],
                                  m_h_patches_ltog_f[0],
                                  m_h_patches_info[p],
                                  m_d_patches_info[p]);
    }


    for (uint32_t p = get_num_patches(); p < get_max_num_patches(); ++p) {
        m_max_capacity_lp_v = std::max(m_max_capacity_lp_v,
                                       m_h_patches_info[p].lp_v.get_capacity());

        m_max_capacity_lp_e = std::max(m_max_capacity_lp_e,
                                       m_h_patches_info[p].lp_e.get_capacity());

        m_max_capacity_lp_f = std::max(m_max_capacity_lp_f,
                                       m_h_patches_info[p].lp_f.get_capacity());
    }
}

void RXMesh::patch_graph_coloring()
{
    std::vector<uint32_t> ids(m_num_patches);
    fill_with_random_numbers(ids.data(), ids.size());

    m_num_colors = 0;

    // init all colors
    for (uint32_t p_id : ids) {
        m_h_patches_info[p_id].color = INVALID32;
    }

    // assign colors
    for (uint32_t p_id : ids) {
        PatchInfo&         patch = m_h_patches_info[p_id];
        std::set<uint32_t> neighbours_color;

        // One Ring
        // put neighbour colors in a set
        // for (uint32_t i = 0; i < patch.patch_stash.stash_size; ++i) {
        //    uint32_t n = patch.patch_stash.get_patch(i);
        //    if (n != INVALID32) {
        //        uint32_t c = m_h_patches_info[n].color;
        //        if (c != INVALID32) {
        //            neighbours_color.insert(c);
        //        }
        //    }
        //}

        // Two Ring
        for (uint32_t i = 0; i < patch.patch_stash.stash_size; ++i) {
            uint32_t n = patch.patch_stash.get_patch(i);
            if (n != INVALID32) {
                uint32_t c = m_h_patches_info[n].color;
                if (c != INVALID32) {
                    neighbours_color.insert(c);
                }


                for (uint32_t j = 0; j < patch.patch_stash.stash_size; ++j) {
                    uint32_t nn = m_h_patches_info[n].patch_stash.get_patch(j);
                    if (nn != INVALID32 && nn != patch.patch_id) {
                        uint32_t cc = m_h_patches_info[nn].color;
                        if (cc != INVALID32) {
                            neighbours_color.insert(cc);
                        }
                    }
                }
            }
        }

        // find the min color id that is not in the list/set
        for (uint32_t i = 0; i < m_num_patches; ++i) {
            if (neighbours_color.find(i) == neighbours_color.end()) {
                patch.color  = i;
                m_num_colors = std::max(m_num_colors, patch.color);
                break;
            }
        }
    }

    m_num_colors++;
}
}  // namespace rxmesh
