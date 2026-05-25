#include <assert.h>
#include <omp.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <new>
#include <numeric>
#include <queue>
#include <set>

#include "patcher/patcher.h"
#include "rxmesh/context.h"
#include "rxmesh/patch_scheduler.h"
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
      m_num_colors(0),
      m_h_v_handles(nullptr),
      m_h_e_handles(nullptr),
      m_h_f_handles(nullptr),
      m_d_v_handles(nullptr),
      m_d_e_handles(nullptr),
      m_d_f_handles(nullptr),
      m_d_evs_all(nullptr),
      m_d_fes_all(nullptr),
      m_d_active_mask_v_all(nullptr),
      m_d_active_mask_e_all(nullptr),
      m_d_active_mask_f_all(nullptr),
      m_d_owned_mask_v_all(nullptr),
      m_d_owned_mask_e_all(nullptr),
      m_d_owned_mask_f_all(nullptr),
      m_d_counts_all(nullptr),
      m_d_dirty_all(nullptr),
      m_d_patch_stashes_all(nullptr),
      m_d_lp_v_tables_all(nullptr),
      m_d_lp_e_tables_all(nullptr),
      m_d_lp_f_tables_all(nullptr),
      m_d_lp_v_stashes_all(nullptr),
      m_d_lp_e_stashes_all(nullptr),
      m_d_lp_f_stashes_all(nullptr),
      m_d_patch_locks_all(nullptr),
      m_d_patch_spins_all(nullptr),
      m_ev_stride_elems(0),
      m_fe_stride_elems(0),
      m_mask_v_stride_words(0),
      m_mask_e_stride_words(0),
      m_mask_f_stride_words(0),
      m_counts_stride_elems(0),
      m_dirty_stride_elems(0)


{
    m_profile_ltog_setup_ms       = 0.0;
    m_profile_ltog_faces_ms       = 0.0;
    m_profile_ltog_edge_scan_ms   = 0.0;
    m_profile_ltog_vertex_scan_ms = 0.0;
    m_profile_ltog_partition_ms   = 0.0;
    m_profile_topology_alloc_ms   = 0.0;
    m_profile_topology_faces_ms   = 0.0;
}

void RXMesh::init(const std::vector<std::vector<uint32_t>>& fv,
                  const std::string                         patcher_file,
                  const float                               capacity_factor,
                  const float                               patch_alloc_factor,
                  const float lp_hashtable_load_factor)
{
    m_capacity_factor             = capacity_factor;
    m_lp_hashtable_load_factor    = lp_hashtable_load_factor;
    m_patch_alloc_factor          = patch_alloc_factor;
    m_profile_ltog_setup_ms       = 0.0;
    m_profile_ltog_faces_ms       = 0.0;
    m_profile_ltog_edge_scan_ms   = 0.0;
    m_profile_ltog_vertex_scan_ms = 0.0;
    m_profile_ltog_partition_ms   = 0.0;
    m_profile_topology_alloc_ms   = 0.0;
    m_profile_topology_faces_ms   = 0.0;

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
    m_timers.add("hashtable.move");
    m_timers.add("cudaMemcpy");
    m_timers.add("bitmask.cudaMemcpy");
    m_timers.add("build.supporting_structures");
    m_timers.add("build.patcher");
    m_timers.add("build.patch_arrays");
    m_timers.add("build.ltog");
    m_timers.add("build.max_capacity");
    m_timers.add("build.topology");
    m_timers.add("build.prefix");
    m_timers.add("build.lp_capacity");
    m_timers.add("build.prefix.cudaMemcpy");
    m_timers.add("build.input_stats");
    m_timers.add("support.first_pass");
    m_timers.add("support.ff_values");
    m_timers.add("stats.edge_loop");
    m_timers.add("stats.face_loop");
    m_timers.add("build_device.cudaMalloc");
    m_timers.add("build_device.real_patches");
    m_timers.add("build_device.sym_stash");
    m_timers.add("patch.counts.cudaMemcpy");
    m_timers.add("patch.topology.cudaMemcpy");
    m_timers.add("patch.dirty.cudaMemset");
    m_timers.add("patch.patch_stash.init");
    m_timers.add("patch.lock.init");
    m_timers.add("patch.patch_stash.cudaMemcpy");
    m_timers.add("patch.patch_info.cudaMemcpy");
    m_timers.add("buildHT.v");
    m_timers.add("buildHT.e");
    m_timers.add("buildHT.f");
    m_timers.add("bitmask.active_v");
    m_timers.add("bitmask.active_e");
    m_timers.add("bitmask.active_f");
    m_timers.add("bitmask.owned_v");
    m_timers.add("bitmask.owned_e");
    m_timers.add("bitmask.owned_f");
    m_timers.add("create_handles.malloc");
    m_timers.add("create_handles.cudaMalloc");
    m_timers.add("create_handles.populate");
    m_timers.add("create_handles.cudaMemcpy");

    // 1)
    m_timers.add("build");
    m_timers.start("build");
    build(fv, patcher_file);
    m_timers.stop("build");

    // 2)
    m_timers.add("populate_patch_stash");
    m_timers.start("populate_patch_stash");
    populate_patch_stash();
    m_timers.stop("populate_patch_stash");

    // 3)
    m_timers.add("coloring");
    m_timers.start("coloring");
    patch_graph_coloring();
    m_timers.stop("coloring");
    RXMESH_INFO("Num colors = {}", m_num_colors);

    // 4)
    m_timers.add("build_device");
    m_timers.start("build_device");
    build_device();
    m_timers.stop("build_device");


    // 5)
    m_timers.add("PatchScheduler");
    m_timers.start("PatchScheduler");
    PatchScheduler sch;
    sch.init(get_max_num_patches());
    sch.refill(get_num_patches());
    m_timers.stop("PatchScheduler");


    // 6)
    m_timers.add("compute_max_lp_capacity");
    m_timers.start("compute_max_lp_capacity");
    // Allocate  extra patches
    compute_max_lp_capacity();
    m_timers.stop("compute_max_lp_capacity");

    // 7)
    m_timers.add("create_handles");
    m_timers.start("create_handles");
    create_handles();
    m_timers.stop("create_handles");

    // 8)
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
                          m_d_v_handles,
                          m_d_e_handles,
                          m_d_f_handles,
                          m_d_patches_info,
                          sch);
    m_timers.stop("context.init");


    RXMESH_INFO("#Vertices = {}, #Faces= {}, #Edges= {}, #Patches = {}",
                m_num_vertices,
                m_num_faces,
                m_num_edges,
                m_num_patches);
    RXMESH_INFO("Input is{} edge manifold",
                ((m_is_input_edge_manifold) ? "" : " Not"));
    RXMESH_INFO("Input is{} closed", ((m_is_input_closed) ? "" : " Not"));
    RXMESH_INFO("Input max valence = {}", m_input_max_valence);
    RXMESH_INFO("max edge incident faces = {}",
                m_input_max_edge_incident_faces);
    RXMESH_INFO("max face adjacent faces = {}",
                m_input_max_face_adjacent_faces);
    RXMESH_INFO("per-patch maximum face count = {}", m_max_faces_per_patch);
    RXMESH_INFO("per-patch maximum edge count = {}", m_max_edges_per_patch);
    RXMESH_INFO("per-patch maximum vertex count = {}",
                m_max_vertices_per_patch);

    ////
    RXMESH_INFO("1) build time = {} (ms)", m_timers.elapsed_millis("build"));
    RXMESH_INFO("   --supporting_structures time = {} (ms)",
                m_timers.elapsed_millis("build.supporting_structures"));
    RXMESH_INFO("      ---support.first_pass time = {} (ms)",
                m_timers.elapsed_millis("support.first_pass"));
    RXMESH_INFO("      ---support.ff_values time = {} (ms)",
                m_timers.elapsed_millis("support.ff_values"));
    RXMESH_INFO("   --patcher time = {} (ms)",
                m_timers.elapsed_millis("build.patcher"));
    RXMESH_INFO("   --patch arrays time = {} (ms)",
                m_timers.elapsed_millis("build.patch_arrays"));
    RXMESH_INFO("   --ltog loop time = {} (ms)",
                m_timers.elapsed_millis("build.ltog"));
    RXMESH_INFO(
        "      ---ltog accumulated setup/faces/edge_scan/"
        "vertex_scan/partition = {}/{}/{}/{}/{} (ms)",
        m_profile_ltog_setup_ms,
        m_profile_ltog_faces_ms,
        m_profile_ltog_edge_scan_ms,
        m_profile_ltog_vertex_scan_ms,
        m_profile_ltog_partition_ms);
    RXMESH_INFO("   --max/capacity time = {} (ms)",
                m_timers.elapsed_millis("build.max_capacity"));
    RXMESH_INFO("   --topology loop time = {} (ms)",
                m_timers.elapsed_millis("build.topology"));
    RXMESH_INFO("      ---topology accumulated alloc/faces = {}/{} (ms)",
                m_profile_topology_alloc_ms,
                m_profile_topology_faces_ms);
    RXMESH_INFO("   --prefix time = {} (ms)",
                m_timers.elapsed_millis("build.prefix"));
    RXMESH_INFO("   --lp capacity time = {} (ms)",
                m_timers.elapsed_millis("build.lp_capacity"));
    RXMESH_INFO("   --prefix cudaMemcpy time = {} (ms)",
                m_timers.elapsed_millis("build.prefix.cudaMemcpy"));
    RXMESH_INFO("   --input stats time = {} (ms)",
                m_timers.elapsed_millis("build.input_stats"));
    RXMESH_INFO("      ---stats edge/face loops = {}/{} (ms)",
                m_timers.elapsed_millis("stats.edge_loop"),
                m_timers.elapsed_millis("stats.face_loop"));
    RXMESH_INFO("2) populate_patch_stash time = {} (ms)",
                m_timers.elapsed_millis("populate_patch_stash"));
    RXMESH_INFO("3) patch graph coloring time = {} (ms)",
                m_timers.elapsed_millis("coloring"));
    RXMESH_INFO("4) build_device time = {} (ms)",
                m_timers.elapsed_millis("build_device"));
    RXMESH_INFO(" -build_device.cudaMalloc time = {} (ms)",
                m_timers.elapsed_millis("build_device.cudaMalloc"));
    RXMESH_INFO(" -build_device.real_patches time = {} (ms)",
                m_timers.elapsed_millis("build_device.real_patches"));
    RXMESH_INFO(" -build_device.sym_stash time = {} (ms)",
                m_timers.elapsed_millis("build_device.sym_stash"));
    RXMESH_INFO(" -buildHT time = {} (ms)", m_timers.elapsed_millis("buildHT"));
    RXMESH_INFO("   --buildHT.v/e/f time = {}/{}/{} (ms)",
                m_timers.elapsed_millis("buildHT.v"),
                m_timers.elapsed_millis("buildHT.e"),
                m_timers.elapsed_millis("buildHT.f"));
    RXMESH_INFO("   --lower_bound time = {} (ms)",
                m_timers.elapsed_millis("lower_bound"));
    RXMESH_INFO("   --ht.insert time = {} (ms)",
                m_timers.elapsed_millis("ht.insert"));
    RXMESH_INFO("   --hashtable.move time = {} (ms)",
                m_timers.elapsed_millis("hashtable.move"));
    RXMESH_INFO("   --LPHashTable time = {} (ms)",
                m_timers.elapsed_millis("LPHashTable"));
    RXMESH_INFO(" -bitmask time = {} (ms)", m_timers.elapsed_millis("bitmask"));
    RXMESH_INFO("   --bitmask active v/e/f time = {}/{}/{} (ms)",
                m_timers.elapsed_millis("bitmask.active_v"),
                m_timers.elapsed_millis("bitmask.active_e"),
                m_timers.elapsed_millis("bitmask.active_f"));
    RXMESH_INFO("   --bitmask owned v/e/f time = {}/{}/{} (ms)",
                m_timers.elapsed_millis("bitmask.owned_v"),
                m_timers.elapsed_millis("bitmask.owned_e"),
                m_timers.elapsed_millis("bitmask.owned_f"));
    RXMESH_INFO("   --bitmask.cudaMemcpy time = {} (ms)",
                m_timers.elapsed_millis("bitmask.cudaMemcpy"));
    RXMESH_INFO(
        " -patch counts/topology/dirty/stash/info copies = "
        "{}/{}/{}/{}/{} (ms)",
        m_timers.elapsed_millis("patch.counts.cudaMemcpy"),
        m_timers.elapsed_millis("patch.topology.cudaMemcpy"),
        m_timers.elapsed_millis("patch.dirty.cudaMemset"),
        m_timers.elapsed_millis("patch.patch_stash.cudaMemcpy"),
        m_timers.elapsed_millis("patch.patch_info.cudaMemcpy"));
    RXMESH_INFO(" -patch device stash/lock init = {}/{} (ms)",
                m_timers.elapsed_millis("patch.patch_stash.init"),
                m_timers.elapsed_millis("patch.lock.init"));

    RXMESH_INFO("5) PatchScheduler time = {} (ms)",
                m_timers.elapsed_millis("PatchScheduler"));
    RXMESH_INFO("6) compute_max_lp_capacity time = {} (ms)",
                m_timers.elapsed_millis("compute_max_lp_capacity"));
    RXMESH_INFO("7) create_handles time = {} (ms)",
                m_timers.elapsed_millis("create_handles"));
    RXMESH_INFO(
        "   --create_handles malloc/cudaMalloc/populate/cudaMemcpy = "
        "{}/{}/{}/{} (ms)",
        m_timers.elapsed_millis("create_handles.malloc"),
        m_timers.elapsed_millis("create_handles.cudaMalloc"),
        m_timers.elapsed_millis("create_handles.populate"),
        m_timers.elapsed_millis("create_handles.cudaMemcpy"));
    RXMESH_INFO("8) context.init time = {} (ms)",
                m_timers.elapsed_millis("context.init"));

    RXMESH_INFO("cudaMemcpy time = {} (ms)",
                m_timers.elapsed_millis("cudaMemcpy"));
    RXMESH_INFO("cudaMalloc time = {} (ms)",
                m_timers.elapsed_millis("cudaMalloc"));
    RXMESH_INFO("malloc time = {} (ms)", m_timers.elapsed_millis("malloc"));
}

RXMesh::~RXMesh()
{
    m_rxmesh_context.m_patch_scheduler.free();

    if (m_h_patches_info != nullptr) {
        for (uint32_t p = 0; p < get_max_num_patches(); ++p) {
            free(m_h_patches_info[p].ev);
            free(m_h_patches_info[p].fe);
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
    }
    GPU_FREE(m_d_evs_all);
    GPU_FREE(m_d_fes_all);
    GPU_FREE(m_d_active_mask_v_all);
    GPU_FREE(m_d_active_mask_e_all);
    GPU_FREE(m_d_active_mask_f_all);
    GPU_FREE(m_d_owned_mask_v_all);
    GPU_FREE(m_d_owned_mask_e_all);
    GPU_FREE(m_d_owned_mask_f_all);
    GPU_FREE(m_d_counts_all);
    GPU_FREE(m_d_dirty_all);
    GPU_FREE(m_d_patch_stashes_all);
    GPU_FREE(m_d_lp_v_tables_all);
    GPU_FREE(m_d_lp_e_tables_all);
    GPU_FREE(m_d_lp_f_tables_all);
    GPU_FREE(m_d_lp_v_stashes_all);
    GPU_FREE(m_d_lp_e_stashes_all);
    GPU_FREE(m_d_lp_f_stashes_all);
    GPU_FREE(m_d_patch_locks_all);
    GPU_FREE(m_d_patch_spins_all);
    GPU_FREE(m_d_patches_info);

    free(m_h_patches_info);
    m_rxmesh_context.release();

    GPU_FREE(m_d_vertex_prefix);
    GPU_FREE(m_d_edge_prefix);
    GPU_FREE(m_d_face_prefix);

    free(m_h_vertex_prefix);
    free(m_h_edge_prefix);
    free(m_h_face_prefix);

    free(m_h_v_handles);
    free(m_h_e_handles);
    free(m_h_f_handles);

    GPU_FREE(m_d_v_handles);
    GPU_FREE(m_d_e_handles);
    GPU_FREE(m_d_f_handles);
}

void RXMesh::build(const std::vector<std::vector<uint32_t>>& fv,
                   const std::string                         patcher_file)
{
    std::vector<uint32_t>                ff_values;
    std::vector<uint32_t>                ff_offset;
    std::vector<std::array<uint32_t, 2>> ev;

    m_max_capacity_lp_v = 0;
    m_max_capacity_lp_e = 0;
    m_max_capacity_lp_f = 0;

    m_timers.start("build.supporting_structures");
    build_supporting_structures(fv, ev, ff_offset, ff_values);
    m_timers.stop("build.supporting_structures");

    m_timers.start("build.patcher");
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
    m_timers.stop("build.patcher");


    m_timers.start("build.patch_arrays");
    m_num_patches     = m_patcher->get_num_patches();
    m_max_num_patches = static_cast<uint32_t>(
        std::ceil(m_patch_alloc_factor * static_cast<float>(m_num_patches)));

    m_h_patches_info =
        (PatchInfo*)malloc(get_max_num_patches() * sizeof(PatchInfo));
    for (uint32_t p = 0; p < get_max_num_patches(); ++p) {
        new (&m_h_patches_info[p]) PatchInfo();
    }
    m_h_patches_ltog_f.resize(get_num_patches());
    m_h_patches_ltog_e.resize(get_num_patches());
    m_h_patches_ltog_v.resize(get_num_patches());
    m_h_num_owned_f.resize(get_max_num_patches(), 0);
    m_h_num_owned_v.resize(get_max_num_patches(), 0);
    m_h_num_owned_e.resize(get_max_num_patches(), 0);
    m_timers.stop("build.patch_arrays");

    m_timers.start("build.ltog");
#pragma omp parallel for
    for (int p = 0; p < static_cast<int>(get_num_patches()); ++p) {
        build_single_patch_ltog(fv, ev, p);
    }
    m_timers.stop("build.ltog");

    // calc max elements for use in build_device (which populates
    // m_h_patches_info and thus we can not use calc_max_elements now)
    m_timers.start("build.max_capacity");
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
    m_timers.stop("build.max_capacity");

    m_timers.start("build.topology");
#pragma omp parallel for
    for (int p = 0; p < static_cast<int>(get_num_patches()); ++p) {
        build_single_patch_topology(fv, p);
    }
    m_timers.stop("build.topology");

    m_timers.start("build.prefix");
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
    m_timers.stop("build.prefix");


    // the hash table capacity should be at least 2* the size of the stash
    m_timers.start("build.lp_capacity");
    m_max_capacity_lp_v = 2 * LPHashTable::stash_size;
    m_max_capacity_lp_e = 2 * LPHashTable::stash_size;
    m_max_capacity_lp_f = 2 * LPHashTable::stash_size;
    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        m_max_capacity_lp_v = std::max(
            m_max_capacity_lp_v,
            static_cast<uint16_t>(
                std::ceil(m_capacity_factor *
                          static_cast<float>(m_h_patches_ltog_v[p].size() -
                                             m_h_num_owned_v[p]) /
                          m_lp_hashtable_load_factor)));

        m_max_capacity_lp_e = std::max(
            m_max_capacity_lp_e,
            static_cast<uint16_t>(
                std::ceil(m_capacity_factor *
                          static_cast<float>(m_h_patches_ltog_e[p].size() -
                                             m_h_num_owned_e[p]) /
                          m_lp_hashtable_load_factor)));

        m_max_capacity_lp_f = std::max(
            m_max_capacity_lp_f,
            static_cast<uint16_t>(
                std::ceil(m_capacity_factor *
                          static_cast<float>(m_h_patches_ltog_f[p].size() -
                                             m_h_num_owned_f[p]) /
                          m_lp_hashtable_load_factor)));
    }
    m_timers.stop("build.lp_capacity");

    m_timers.start("cudaMalloc");
    CUDA_ERROR(cudaMalloc((void**)&m_d_vertex_prefix, patches_1_bytes));
    CUDA_ERROR(cudaMalloc((void**)&m_d_edge_prefix, patches_1_bytes));
    CUDA_ERROR(cudaMalloc((void**)&m_d_face_prefix, patches_1_bytes));

    m_timers.stop("cudaMalloc");


    m_timers.start("cudaMemcpy");
    m_timers.start("build.prefix.cudaMemcpy");
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
    m_timers.stop("build.prefix.cudaMemcpy");
    m_timers.stop("cudaMemcpy");
}

void RXMesh::create_handles()
{
    // allocate host and device memory
    m_timers.start("malloc");
    m_timers.start("create_handles.malloc");
    m_h_v_handles =
        (VertexHandle*)malloc(sizeof(VertexHandle) * m_num_vertices);
    m_h_e_handles = (EdgeHandle*)malloc(sizeof(EdgeHandle) * m_num_edges);
    m_h_f_handles = (FaceHandle*)malloc(sizeof(FaceHandle) * m_num_faces);
    m_timers.stop("create_handles.malloc");
    m_timers.stop("malloc");

    m_timers.start("cudaMalloc");
    m_timers.start("create_handles.cudaMalloc");
    CUDA_ERROR(cudaMalloc((void**)&m_d_v_handles,
                          sizeof(VertexHandle) * m_num_vertices));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_e_handles, sizeof(EdgeHandle) * m_num_edges));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_f_handles, sizeof(FaceHandle) * m_num_faces));
    m_timers.stop("create_handles.cudaMalloc");
    m_timers.stop("cudaMalloc");

    // populate m_h_v_handles, m_h_e_handles, m_h_f_handles

    m_timers.start("create_handles.populate");
    int v_id(0), e_id(0), f_id(0);
    for (int p = 0; p < get_num_patches(); ++p) {
        int num_vertices = *(m_h_patches_info[p].num_vertices);
        int num_edges    = *(m_h_patches_info[p].num_edges);
        int num_faces    = *(m_h_patches_info[p].num_faces);


        for (int v = 0; v < num_vertices; ++v) {
            LocalVertexT vl(v);
            if (m_h_patches_info[p].is_owned(vl) &&
                !m_h_patches_info[p].is_deleted(vl)) {
                m_h_v_handles[v_id] = VertexHandle(p, vl);
                ++v_id;
            }
        }

        for (int e = 0; e < num_edges; ++e) {
            LocalEdgeT el(e);
            if (m_h_patches_info[p].is_owned(el) &&
                !m_h_patches_info[p].is_deleted(el)) {
                m_h_e_handles[e_id] = EdgeHandle(p, el);
                ++e_id;
            }
        }

        for (int f = 0; f < num_faces; ++f) {
            LocalFaceT fl(f);
            if (m_h_patches_info[p].is_owned(fl) &&
                !m_h_patches_info[p].is_deleted(fl)) {
                m_h_f_handles[f_id] = FaceHandle(p, fl);
                ++f_id;
            }
        }
    }
    m_timers.stop("create_handles.populate");

    // move handles to device
    m_timers.start("cudaMemcpy");
    m_timers.start("create_handles.cudaMemcpy");
    CUDA_ERROR(cudaMemcpy(m_d_v_handles,
                          m_h_v_handles,
                          sizeof(VertexHandle) * m_num_vertices,
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMemcpy(m_d_e_handles,
                          m_h_e_handles,
                          sizeof(EdgeHandle) * m_num_edges,
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMemcpy(m_d_f_handles,
                          m_h_f_handles,
                          sizeof(FaceHandle) * m_num_faces,
                          cudaMemcpyHostToDevice));
    m_timers.stop("create_handles.cudaMemcpy");
    m_timers.stop("cudaMemcpy");
}
void RXMesh::build_supporting_structures(
    const std::vector<std::vector<uint32_t>>& fv,
    std::vector<std::array<uint32_t, 2>>&     ev,
    std::vector<uint32_t>&                    ff_offset,
    std::vector<uint32_t>&                    ff_values)
{
    struct EdgeFaces
    {
        uint32_t              faces[2] = {0, 0};
        uint32_t              count    = 0;
        std::vector<uint32_t> extra;

        explicit EdgeFaces(uint32_t f)
        {
            push_back(f);
        }

        void push_back(uint32_t f)
        {
            if (count < 2) {
                faces[count] = f;
            } else {
                extra.push_back(f);
            }
            ++count;
        }

        uint32_t size() const
        {
            return count;
        }

        uint32_t operator[](uint32_t i) const
        {
            return (i < 2) ? faces[i] : extra[i - 2];
        }
    };

    m_num_faces    = static_cast<uint32_t>(fv.size());
    m_num_vertices = 0;
    m_num_edges    = 0;
    m_edges_map.clear();
    m_edges_map.max_load_factor(0.7f);

    m_input_max_edge_incident_faces = 0;
    m_input_max_face_adjacent_faces = 0;
    m_input_max_valence             = 0;
    m_is_input_closed               = true;
    m_is_input_edge_manifold        = true;

    // assuming manifold mesh i.e., #E = 1.5#F
    std::vector<EdgeFaces> edge_faces;
    const size_t           reserve_size =
        static_cast<size_t>(1.5f * static_cast<float>(m_num_faces));
    edge_faces.reserve(reserve_size);
    m_edges_map.reserve(reserve_size);
    ev.clear();
    ev.reserve(reserve_size);

    std::vector<uint32_t> ff_size(m_num_faces, 0);
    std::vector<uint32_t> vv_count;
    vv_count.reserve(std::max<size_t>(m_num_faces / 2, 1));
    uint32_t num_open_edges = 0;

    auto add_vertex_valence = [&](uint32_t v) {
        if (v >= vv_count.size()) {
            vv_count.resize(static_cast<size_t>(v) + 1, 0);
        }
        m_input_max_valence = std::max(m_input_max_valence, ++vv_count[v]);
    };

    m_timers.start("support.first_pass");
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
            m_num_vertices = std::max(m_num_vertices, v1);

            std::pair<uint32_t, uint32_t> edge = detail::edge_key(v0, v1);
            auto [e_iter, inserted] = m_edges_map.emplace(edge, m_num_edges);
            if (inserted) {
                ++m_num_edges;
                ev.push_back({v0, v1});
                edge_faces.emplace_back(f);
                ++num_open_edges;
                m_input_max_edge_incident_faces =
                    std::max(m_input_max_edge_incident_faces, 1u);
                add_vertex_valence(v0);
                add_vertex_valence(v1);
            } else {
                uint32_t edge_id        = (*e_iter).second;
                uint32_t incident_faces = edge_faces[edge_id].size();

                for (uint32_t f0 = 0; f0 < incident_faces; ++f0) {
                    uint32_t other_face = edge_faces[edge_id][f0];
                    ++ff_size[other_face];
                }
                ff_size[f] += incident_faces;

                edge_faces[edge_id].push_back(f);
                uint32_t new_incident_faces = incident_faces + 1;
                if (incident_faces == 1) {
                    --num_open_edges;
                }
                if (new_incident_faces > 2) {
                    m_is_input_edge_manifold = false;
                }
                m_input_max_edge_incident_faces = std::max(
                    m_input_max_edge_incident_faces, new_incident_faces);
            }
        }
    }
    m_timers.stop("support.first_pass");
    ++m_num_vertices;
    m_is_input_closed = (num_open_edges == 0);

    if (m_num_edges != static_cast<uint32_t>(m_edges_map.size())) {
        RXMESH_ERROR(
            "rxmesh::build_supporting_structures() m_num_edges ({}) should "
            "match the size of edge_map ({})",
            m_num_edges,
            m_edges_map.size());
        exit(EXIT_FAILURE);
    }

    ff_offset.resize(m_num_faces + 1);
    uint32_t ff_count = 0;
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        ff_offset[f] = ff_count;
        ff_count += ff_size[f];
        m_input_max_face_adjacent_faces =
            std::max(m_input_max_face_adjacent_faces, ff_size[f]);
    }
    ff_offset[m_num_faces] = ff_count;
    ff_values.clear();
    ff_values.resize(ff_offset.back());
    std::fill(ff_size.begin(), ff_size.end(), 0);

    m_timers.start("support.ff_values");
    for (uint32_t e = 0; e < m_num_edges; ++e) {
        const uint32_t incident_faces = edge_faces[e].size();
        for (uint32_t i = 0; i < incident_faces; ++i) {
            uint32_t f0 = edge_faces[e][i];
            for (uint32_t j = i + 1; j < incident_faces; ++j) {
                uint32_t f1 = edge_faces[e][j];

                uint32_t f0_offset = ff_size[f0]++;
                uint32_t f1_offset = ff_size[f1]++;
                f0_offset += ff_offset[f0];
                f1_offset += ff_offset[f1];

                ff_values[f0_offset] = f1;
                ff_values[f1_offset] = f0;
            }
        }
    }
    m_timers.stop("support.ff_values");
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
    m_timers.start("stats.edge_loop");
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
    m_timers.stop("stats.edge_loop");

    // calc max ff
    m_input_max_face_adjacent_faces = 0;
    m_timers.start("stats.face_loop");
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
    m_timers.stop("stats.face_loop");
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
    const std::vector<std::vector<uint32_t>>&   fv,
    const std::vector<std::array<uint32_t, 2>>& ev,
    const uint32_t                              patch_id)
{
    double profile_start = omp_get_wtime();

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
    const double      setup_ms = 1000.0 * (omp_get_wtime() - profile_start);

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

    profile_start          = omp_get_wtime();
    uint16_t local_face_id = 0;
    for (uint32_t f = p_start; f < p_end; ++f) {
        uint32_t face_id = m_patcher->get_patches_val()[f];
        add_new_face(face_id, local_face_id++);
    }

    for (uint32_t f = r_start; f < r_end; ++f) {
        uint32_t face_id = m_patcher->get_external_ribbon_val()[f];
        add_new_face(face_id, local_face_id++);
    }
    const double faces_ms = 1000.0 * (omp_get_wtime() - profile_start);

    // The previous loop over faces should already insert all owned edges and
    // vertices into `ltog`. However, we still iterate over owned edges and
    // owned vertices as a safeguard, in case the patcher behavior changes
    // in the future. At the moment, the patcher marks an edge as owned by
    // patch P only if at least one of its incident faces is owned by P. Same
    // thing for vertices

    // add edges owned by this patch
    profile_start = omp_get_wtime();
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
    const double edge_scan_ms = 1000.0 * (omp_get_wtime() - profile_start);

    // add vertices owned by this patch
    profile_start = omp_get_wtime();
    for (uint32_t v = 0; v < m_num_vertices; ++v) {
        // if the edge is owned by this patch but it was not added yet
        if (m_patcher->get_vertex_patch_id(v) == patch_id &&
            !is_vertex_added[v]) {
            m_h_patches_ltog_v[patch_id].push_back(v);
        }
    }
    const double vertex_scan_ms = 1000.0 * (omp_get_wtime() - profile_start);

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

    profile_start             = omp_get_wtime();
    m_h_num_owned_f[patch_id] = create_unique_mapping(
        m_h_patches_ltog_f[patch_id], m_patcher->get_face_patch());

    m_h_num_owned_e[patch_id] = create_unique_mapping(
        m_h_patches_ltog_e[patch_id], m_patcher->get_edge_patch());

    m_h_num_owned_v[patch_id] = create_unique_mapping(
        m_h_patches_ltog_v[patch_id], m_patcher->get_vertex_patch());
    const double partition_ms = 1000.0 * (omp_get_wtime() - profile_start);

#pragma omp critical(rxmesh_profile_ltog)
    {
        m_profile_ltog_setup_ms += setup_ms;
        m_profile_ltog_faces_ms += faces_ms;
        m_profile_ltog_edge_scan_ms += edge_scan_ms;
        m_profile_ltog_vertex_scan_ms += vertex_scan_ms;
        m_profile_ltog_partition_ms += partition_ms;
    }
}

void RXMesh::build_single_patch_topology(
    const std::vector<std::vector<uint32_t>>& fv,
    const uint32_t                            patch_id)
{
    double profile_start = omp_get_wtime();

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
    const double      alloc_ms = 1000.0 * (omp_get_wtime() - profile_start);

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


    profile_start = omp_get_wtime();
    for (uint32_t f = p_start; f < p_end; ++f) {
        uint32_t face_id = m_patcher->get_patches_val()[f];
        add_new_face(face_id);
    }

    for (uint32_t f = r_start; f < r_end; ++f) {
        uint32_t face_id = m_patcher->get_external_ribbon_val()[f];
        add_new_face(face_id);
    }
    const double faces_ms = 1000.0 * (omp_get_wtime() - profile_start);

#pragma omp critical(rxmesh_profile_topology)
    {
        m_profile_topology_alloc_ms += alloc_ms;
        m_profile_topology_faces_ms += faces_ms;
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
    const uint32_t max_num_patches     = get_max_num_patches();
    const uint16_t p_vertices_capacity = get_per_patch_max_vertex_capacity();
    const uint16_t p_edges_capacity    = get_per_patch_max_edge_capacity();
    const uint16_t p_faces_capacity    = get_per_patch_max_face_capacity();

    LPHashTable lp_v_capacity_probe(max_lp_hashtable_capacity<LocalVertexT>(),
                                    false);
    LPHashTable lp_e_capacity_probe(max_lp_hashtable_capacity<LocalEdgeT>(),
                                    false);
    LPHashTable lp_f_capacity_probe(max_lp_hashtable_capacity<LocalFaceT>(),
                                    false);
    const uint16_t lp_v_capacity = lp_v_capacity_probe.get_capacity();
    const uint16_t lp_e_capacity = lp_e_capacity_probe.get_capacity();
    const uint16_t lp_f_capacity = lp_f_capacity_probe.get_capacity();
    lp_v_capacity_probe.free();
    lp_e_capacity_probe.free();
    lp_f_capacity_probe.free();

    m_ev_stride_elems = static_cast<uint32_t>(p_edges_capacity) * 2u;
    m_fe_stride_elems =
        (static_cast<uint32_t>(p_faces_capacity) * 3u + 1u) & ~1u;
    m_counts_stride_elems = 4u;
    m_dirty_stride_elems  = 1u;

    m_mask_v_stride_words = static_cast<uint32_t>(
        detail::mask_num_bytes(p_vertices_capacity) / sizeof(uint32_t));
    m_mask_e_stride_words = static_cast<uint32_t>(
        detail::mask_num_bytes(p_edges_capacity) / sizeof(uint32_t));
    m_mask_f_stride_words = static_cast<uint32_t>(
        detail::mask_num_bytes(p_faces_capacity) / sizeof(uint32_t));

    m_timers.start("cudaMalloc");
    m_timers.start("build_device.cudaMalloc");
    CUDA_ERROR(cudaMalloc((void**)&m_d_patches_info,
                          max_num_patches * sizeof(PatchInfo)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_evs_all,
                   max_num_patches * m_ev_stride_elems * sizeof(LocalVertexT)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_fes_all,
                   max_num_patches * m_fe_stride_elems * sizeof(LocalEdgeT)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_active_mask_v_all,
                   max_num_patches * m_mask_v_stride_words * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_active_mask_e_all,
                   max_num_patches * m_mask_e_stride_words * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_active_mask_f_all,
                   max_num_patches * m_mask_f_stride_words * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_owned_mask_v_all,
                   max_num_patches * m_mask_v_stride_words * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_owned_mask_e_all,
                   max_num_patches * m_mask_e_stride_words * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_owned_mask_f_all,
                   max_num_patches * m_mask_f_stride_words * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_counts_all,
                   max_num_patches * m_counts_stride_elems * sizeof(uint16_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_dirty_all,
                   max_num_patches * m_dirty_stride_elems * sizeof(int)));
    CUDA_ERROR(cudaMalloc(
        (void**)&m_d_patch_stashes_all,
        max_num_patches * PatchStash::stash_size * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_lp_v_tables_all,
                          max_num_patches * lp_v_capacity * sizeof(LPPair)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_lp_e_tables_all,
                          max_num_patches * lp_e_capacity * sizeof(LPPair)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_lp_f_tables_all,
                          max_num_patches * lp_f_capacity * sizeof(LPPair)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_lp_v_stashes_all,
                   max_num_patches * LPHashTable::stash_size * sizeof(LPPair)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_lp_e_stashes_all,
                   max_num_patches * LPHashTable::stash_size * sizeof(LPPair)));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_lp_f_stashes_all,
                   max_num_patches * LPHashTable::stash_size * sizeof(LPPair)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_patch_locks_all,
                          max_num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&m_d_patch_spins_all,
                          max_num_patches * sizeof(uint32_t)));
    m_timers.stop("build_device.cudaMalloc");
    m_timers.stop("cudaMalloc");

    // make sure that if a patch stash of patch p has patch q, then q's patch
    // stash should have p in it
    m_timers.start("build_device.sym_stash");
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
    m_timers.stop("build_device.sym_stash");

    std::vector<PatchInfo> h_d_patches(max_num_patches);
    std::vector<uint16_t>  h_counts_all(max_num_patches * m_counts_stride_elems,
                                       0);
    std::vector<int> h_dirty_all(max_num_patches * m_dirty_stride_elems, 0);
    std::vector<LocalVertexT> h_evs_all(max_num_patches * m_ev_stride_elems);
    std::vector<LocalEdgeT>   h_fes_all(max_num_patches * m_fe_stride_elems);
    std::vector<uint32_t>     h_active_v_all(
        max_num_patches * m_mask_v_stride_words, 0);
    std::vector<uint32_t> h_active_e_all(
        max_num_patches * m_mask_e_stride_words, 0);
    std::vector<uint32_t> h_active_f_all(
        max_num_patches * m_mask_f_stride_words, 0);
    std::vector<uint32_t> h_owned_v_all(max_num_patches * m_mask_v_stride_words,
                                        0);
    std::vector<uint32_t> h_owned_e_all(max_num_patches * m_mask_e_stride_words,
                                        0);
    std::vector<uint32_t> h_owned_f_all(max_num_patches * m_mask_f_stride_words,
                                        0);
    std::vector<uint32_t> h_patch_stashes_all(
        max_num_patches * PatchStash::stash_size, INVALID32);
    std::vector<LPPair> h_lp_v_tables_all(max_num_patches * lp_v_capacity);
    std::vector<LPPair> h_lp_e_tables_all(max_num_patches * lp_e_capacity);
    std::vector<LPPair> h_lp_f_tables_all(max_num_patches * lp_f_capacity);
    std::vector<LPPair> h_lp_v_stashes_all(max_num_patches *
                                           LPHashTable::stash_size);
    std::vector<LPPair> h_lp_e_stashes_all(max_num_patches *
                                           LPHashTable::stash_size);
    std::vector<LPPair> h_lp_f_stashes_all(max_num_patches *
                                           LPHashTable::stash_size);

    auto fill_mask = [&](uint32_t*&             h_mask,
                         std::vector<uint32_t>& slab,
                         const uint32_t         slot_offset,
                         const uint16_t         capacity,
                         const uint16_t         num_set) {
        const size_t num_bytes = detail::mask_num_bytes(capacity);
        m_timers.start("malloc");
        h_mask = static_cast<uint32_t*>(malloc(num_bytes));
        m_timers.stop("malloc");
        memset(h_mask, 0, num_bytes);
        for (uint16_t i = 0; i < num_set; ++i) {
            detail::bitmask_set_bit(i, h_mask);
        }
        memcpy(slab.data() + slot_offset, h_mask, num_bytes);
    };

    auto bind_device_lp = [](LPHashTable&       d_hashtable,
                             const LPHashTable& h_hashtable,
                             LPPair*            d_table,
                             LPPair*            d_stash) {
        d_hashtable                = h_hashtable;
        d_hashtable.m_table        = d_table;
        d_hashtable.m_stash        = d_stash;
        d_hashtable.m_is_on_device = true;
    };

    auto build_ht = [&](const std::vector<std::vector<uint32_t>>& ltog,
                        const std::vector<uint32_t>&              p_ltog,
                        const std::vector<uint32_t>&              element_patch,
                        const std::vector<uint16_t>&              num_owned,
                        const uint16_t                            num_elements,
                        const uint16_t num_owned_elements,
                        const uint16_t cap,
                        PatchStash&    stash,
                        LPHashTable&   h_hashtable) {
        m_timers.start("buildHT");

        const uint16_t num_not_owned = num_elements - num_owned_elements;

        m_timers.start("LPHashTable");
        h_hashtable = LPHashTable(cap, false);
        m_timers.stop("LPHashTable");

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

        m_timers.stop("buildHT");
    };

    m_timers.start("build_device.real_patches");
    for (uint32_t p = 0; p < max_num_patches; ++p) {
        const bool     valid_patch = p < get_num_patches();
        const uint16_t p_num_vertices =
            valid_patch ? static_cast<uint16_t>(m_h_patches_ltog_v[p].size()) :
                          0;
        const uint16_t p_num_edges =
            valid_patch ? static_cast<uint16_t>(m_h_patches_ltog_e[p].size()) :
                          0;
        const uint16_t p_num_faces =
            valid_patch ? static_cast<uint16_t>(m_h_patches_ltog_f[p].size()) :
                          0;
        const uint16_t p_num_owned_vertices =
            valid_patch ? m_h_num_owned_v[p] : 0;
        const uint16_t p_num_owned_edges = valid_patch ? m_h_num_owned_e[p] : 0;
        const uint16_t p_num_owned_faces = valid_patch ? m_h_num_owned_f[p] : 0;

        PatchInfo& h_patch_info = m_h_patches_info[p];
        h_patch_info.num_faces =
            static_cast<uint16_t*>(malloc(3 * sizeof(uint16_t)));
        h_patch_info.num_edges         = h_patch_info.num_faces + 1;
        h_patch_info.num_vertices      = h_patch_info.num_faces + 2;
        h_patch_info.num_faces[0]      = p_num_faces;
        h_patch_info.num_edges[0]      = p_num_edges;
        h_patch_info.num_vertices[0]   = p_num_vertices;
        h_patch_info.vertices_capacity = p_vertices_capacity;
        h_patch_info.edges_capacity    = p_edges_capacity;
        h_patch_info.faces_capacity    = p_faces_capacity;
        h_patch_info.patch_id          = valid_patch ? p : INVALID32;
        h_patch_info.child_id          = INVALID32;
        h_patch_info.should_slice      = false;
        h_patch_info.dirty             = static_cast<int*>(malloc(sizeof(int)));
        h_patch_info.dirty[0]          = 0;

        h_patch_info.ev = static_cast<LocalVertexT*>(realloc(
            h_patch_info.ev, p_edges_capacity * 2 * sizeof(LocalVertexT)));
        h_patch_info.fe = static_cast<LocalEdgeT*>(realloc(
            h_patch_info.fe, p_faces_capacity * 3 * sizeof(LocalEdgeT)));

        const uint32_t counts_offset    = p * m_counts_stride_elems;
        h_counts_all[counts_offset + 0] = p_num_faces;
        h_counts_all[counts_offset + 1] = p_num_edges;
        h_counts_all[counts_offset + 2] = p_num_vertices;

        if (p_num_edges > 0) {
            memcpy(h_evs_all.data() + p * m_ev_stride_elems,
                   h_patch_info.ev,
                   p_num_edges * 2 * sizeof(LocalVertexT));
        }
        if (p_num_faces > 0) {
            memcpy(h_fes_all.data() + p * m_fe_stride_elems,
                   h_patch_info.fe,
                   p_num_faces * 3 * sizeof(LocalEdgeT));
        }

        m_timers.start("bitmask");
        fill_mask(h_patch_info.active_mask_v,
                  h_active_v_all,
                  p * m_mask_v_stride_words,
                  p_vertices_capacity,
                  p_num_vertices);
        fill_mask(h_patch_info.active_mask_e,
                  h_active_e_all,
                  p * m_mask_e_stride_words,
                  p_edges_capacity,
                  p_num_edges);
        fill_mask(h_patch_info.active_mask_f,
                  h_active_f_all,
                  p * m_mask_f_stride_words,
                  p_faces_capacity,
                  p_num_faces);
        fill_mask(h_patch_info.owned_mask_v,
                  h_owned_v_all,
                  p * m_mask_v_stride_words,
                  p_vertices_capacity,
                  p_num_owned_vertices);
        fill_mask(h_patch_info.owned_mask_e,
                  h_owned_e_all,
                  p * m_mask_e_stride_words,
                  p_edges_capacity,
                  p_num_owned_edges);
        fill_mask(h_patch_info.owned_mask_f,
                  h_owned_f_all,
                  p * m_mask_f_stride_words,
                  p_faces_capacity,
                  p_num_owned_faces);
        m_timers.stop("bitmask");

        memcpy(h_patch_stashes_all.data() + p * PatchStash::stash_size,
               h_patch_info.patch_stash.m_stash,
               PatchStash::stash_size * sizeof(uint32_t));

        if (valid_patch) {
            m_timers.start("buildHT.v");
            build_ht(m_h_patches_ltog_v,
                     m_h_patches_ltog_v[p],
                     m_patcher->get_vertex_patch(),
                     m_h_num_owned_v,
                     p_num_vertices,
                     p_num_owned_vertices,
                     max_lp_hashtable_capacity<LocalVertexT>(),
                     h_patch_info.patch_stash,
                     h_patch_info.lp_v);
            m_timers.stop("buildHT.v");

            m_timers.start("buildHT.e");
            build_ht(m_h_patches_ltog_e,
                     m_h_patches_ltog_e[p],
                     m_patcher->get_edge_patch(),
                     m_h_num_owned_e,
                     p_num_edges,
                     p_num_owned_edges,
                     max_lp_hashtable_capacity<LocalEdgeT>(),
                     h_patch_info.patch_stash,
                     h_patch_info.lp_e);
            m_timers.stop("buildHT.e");

            m_timers.start("buildHT.f");
            build_ht(m_h_patches_ltog_f,
                     m_h_patches_ltog_f[p],
                     m_patcher->get_face_patch(),
                     m_h_num_owned_f,
                     p_num_faces,
                     p_num_owned_faces,
                     max_lp_hashtable_capacity<LocalFaceT>(),
                     h_patch_info.patch_stash,
                     h_patch_info.lp_f);
            m_timers.stop("buildHT.f");
        } else {
            h_patch_info.color = INVALID32;
            m_timers.start("LPHashTable");
            h_patch_info.lp_v =
                LPHashTable(max_lp_hashtable_capacity<LocalVertexT>(), false);
            h_patch_info.lp_e =
                LPHashTable(max_lp_hashtable_capacity<LocalEdgeT>(), false);
            h_patch_info.lp_f =
                LPHashTable(max_lp_hashtable_capacity<LocalFaceT>(), false);
            m_timers.stop("LPHashTable");
        }

        memcpy(h_lp_v_tables_all.data() + p * lp_v_capacity,
               h_patch_info.lp_v.m_table,
               h_patch_info.lp_v.get_capacity() * sizeof(LPPair));
        memcpy(h_lp_e_tables_all.data() + p * lp_e_capacity,
               h_patch_info.lp_e.m_table,
               h_patch_info.lp_e.get_capacity() * sizeof(LPPair));
        memcpy(h_lp_f_tables_all.data() + p * lp_f_capacity,
               h_patch_info.lp_f.m_table,
               h_patch_info.lp_f.get_capacity() * sizeof(LPPair));
        memcpy(h_lp_v_stashes_all.data() + p * LPHashTable::stash_size,
               h_patch_info.lp_v.m_stash,
               LPHashTable::stash_size * sizeof(LPPair));
        memcpy(h_lp_e_stashes_all.data() + p * LPHashTable::stash_size,
               h_patch_info.lp_e.m_stash,
               LPHashTable::stash_size * sizeof(LPPair));
        memcpy(h_lp_f_stashes_all.data() + p * LPHashTable::stash_size,
               h_patch_info.lp_f.m_stash,
               LPHashTable::stash_size * sizeof(LPPair));

        PatchInfo& d_patch = h_d_patches[p];
        d_patch.ev         = m_d_evs_all + p * m_ev_stride_elems;
        d_patch.fe         = m_d_fes_all + p * m_fe_stride_elems;
        d_patch.active_mask_v =
            m_d_active_mask_v_all + p * m_mask_v_stride_words;
        d_patch.active_mask_e =
            m_d_active_mask_e_all + p * m_mask_e_stride_words;
        d_patch.active_mask_f =
            m_d_active_mask_f_all + p * m_mask_f_stride_words;
        d_patch.owned_mask_v = m_d_owned_mask_v_all + p * m_mask_v_stride_words;
        d_patch.owned_mask_e = m_d_owned_mask_e_all + p * m_mask_e_stride_words;
        d_patch.owned_mask_f = m_d_owned_mask_f_all + p * m_mask_f_stride_words;
        d_patch.num_faces    = m_d_counts_all + counts_offset;
        d_patch.num_edges    = d_patch.num_faces + 1;
        d_patch.num_vertices = d_patch.num_faces + 2;
        d_patch.vertices_capacity = p_vertices_capacity;
        d_patch.edges_capacity    = p_edges_capacity;
        d_patch.faces_capacity    = p_faces_capacity;
        d_patch.patch_id          = valid_patch ? p : INVALID32;
        d_patch.color             = h_patch_info.color;
        d_patch.patch_stash.m_stash =
            m_d_patch_stashes_all + p * PatchStash::stash_size;
        d_patch.patch_stash.m_is_on_device = true;
        bind_device_lp(d_patch.lp_v,
                       h_patch_info.lp_v,
                       m_d_lp_v_tables_all + p * lp_v_capacity,
                       m_d_lp_v_stashes_all + p * LPHashTable::stash_size);
        bind_device_lp(d_patch.lp_e,
                       h_patch_info.lp_e,
                       m_d_lp_e_tables_all + p * lp_e_capacity,
                       m_d_lp_e_stashes_all + p * LPHashTable::stash_size);
        bind_device_lp(d_patch.lp_f,
                       h_patch_info.lp_f,
                       m_d_lp_f_tables_all + p * lp_f_capacity,
                       m_d_lp_f_stashes_all + p * LPHashTable::stash_size);
        d_patch.lock.bind(m_d_patch_locks_all + p, m_d_patch_spins_all + p);
        d_patch.dirty        = m_d_dirty_all + p * m_dirty_stride_elems;
        d_patch.child_id     = INVALID32;
        d_patch.should_slice = false;
    }
    m_timers.stop("build_device.real_patches");

    m_timers.start("cudaMemcpy");
    CUDA_ERROR(cudaMemcpy(m_d_counts_all,
                          h_counts_all.data(),
                          h_counts_all.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_dirty_all,
                          h_dirty_all.data(),
                          h_dirty_all.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_evs_all,
                          h_evs_all.data(),
                          h_evs_all.size() * sizeof(LocalVertexT),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_fes_all,
                          h_fes_all.data(),
                          h_fes_all.size() * sizeof(LocalEdgeT),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_active_mask_v_all,
                          h_active_v_all.data(),
                          h_active_v_all.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_active_mask_e_all,
                          h_active_e_all.data(),
                          h_active_e_all.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_active_mask_f_all,
                          h_active_f_all.data(),
                          h_active_f_all.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_owned_mask_v_all,
                          h_owned_v_all.data(),
                          h_owned_v_all.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_owned_mask_e_all,
                          h_owned_e_all.data(),
                          h_owned_e_all.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_owned_mask_f_all,
                          h_owned_f_all.data(),
                          h_owned_f_all.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_patch_stashes_all,
                          h_patch_stashes_all.data(),
                          h_patch_stashes_all.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_lp_v_tables_all,
                          h_lp_v_tables_all.data(),
                          h_lp_v_tables_all.size() * sizeof(LPPair),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_lp_e_tables_all,
                          h_lp_e_tables_all.data(),
                          h_lp_e_tables_all.size() * sizeof(LPPair),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_lp_f_tables_all,
                          h_lp_f_tables_all.data(),
                          h_lp_f_tables_all.size() * sizeof(LPPair),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_lp_v_stashes_all,
                          h_lp_v_stashes_all.data(),
                          h_lp_v_stashes_all.size() * sizeof(LPPair),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_lp_e_stashes_all,
                          h_lp_e_stashes_all.data(),
                          h_lp_e_stashes_all.size() * sizeof(LPPair),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_lp_f_stashes_all,
                          h_lp_f_stashes_all.data(),
                          h_lp_f_stashes_all.size() * sizeof(LPPair),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_patches_info,
                          h_d_patches.data(),
                          h_d_patches.size() * sizeof(PatchInfo),
                          cudaMemcpyHostToDevice));
    m_timers.stop("cudaMemcpy");

    CUDA_ERROR(
        cudaMemset(m_d_patch_locks_all, 0, max_num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(
        m_d_patch_spins_all, INVALID8, max_num_patches * sizeof(uint32_t)));
}

void RXMesh::compute_max_lp_capacity()
{
    for (uint32_t p = 0; p < get_max_num_patches(); ++p) {
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
