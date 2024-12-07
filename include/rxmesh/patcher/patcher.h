#pragma once

#include <stdint.h>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>

#include "rxmesh/util/util.h"

#define CEREAL_RAPIDJSON_NAMESPACE CerealRapidjson

#include "cereal/archives/portable_binary.hpp"
#include "cereal/cereal.hpp"
#include "cereal/types/vector.hpp"

namespace rxmesh {

class RXMeshDynamic;

namespace patcher {

/**
 * @brief Takes an input mesh and partition it to patches using Lloyd algorithm
 * on the gpu
 */
class Patcher
{

    friend class RXMeshDynamic;

   public:
    Patcher() = default;

    Patcher(uint32_t                                  patch_size,
            const std::vector<uint32_t>&              ff_offset,
            const std::vector<uint32_t>&              ff_values,
            const std::vector<std::vector<uint32_t>>& fv,
            const std::unordered_map<std::pair<uint32_t, uint32_t>,
                                     uint32_t,
                                     ::rxmesh::detail::edge_key_hash> edges_map,
            const uint32_t num_vertices,
            const uint32_t num_edges,
            bool           use_metis);

    Patcher(std::string filename);

    ~Patcher();

    void print_statistics();

    uint32_t get_num_patches() const
    {
        return m_num_patches;
    }

    uint32_t get_patch_size() const
    {
        return m_patch_size;
    }

    std::vector<uint32_t>& get_face_patch()
    {
        return m_face_patch;
    }

    std::vector<uint32_t>& get_vertex_patch()
    {
        return m_vertex_patch;
    }

    std::vector<uint32_t>& get_edge_patch()
    {
        return m_edge_patch;
    }

    uint32_t* get_patches_val()
    {
        return m_patches_val.data();
    }

    uint32_t* get_patches_offset()
    {
        return m_patches_offset.data();
    }

    std::vector<uint32_t>& get_external_ribbon_val()
    {
        return m_ribbon_ext_val;
    }

    std::vector<uint32_t>& get_external_ribbon_offset()
    {
        return m_ribbon_ext_offset;
    }

    uint32_t get_face_patch_id(const uint32_t fid) const
    {
        return m_face_patch[fid];
    }

    uint32_t get_vertex_patch_id(const uint32_t vid) const
    {
        return m_vertex_patch[vid];
    }

    uint32_t get_edge_patch_id(const uint32_t eid) const
    {
        return m_edge_patch[eid];
    }
    uint32_t get_num_ext_ribbon_faces() const
    {
        return m_ribbon_ext_offset[m_num_patches - 1];
    }
    void get_max_min_avg_patch_size(uint32_t& min_p,
                                    uint32_t& max_p,
                                    uint32_t& avg_p) const
    {
        max_p = 0;
        min_p = m_num_faces;
        avg_p = 0;
        for (uint32_t p = 0; p < m_num_patches; p++) {
            uint32_t p_size =
                m_patches_offset[p] - ((p == 0) ? 0 : m_patches_offset[p - 1]);
            p_size += m_ribbon_ext_offset[p] -
                      ((p == 0) ? 0 : m_ribbon_ext_offset[p - 1]);
            avg_p += p_size;
            max_p = std::max(max_p, p_size);
            min_p = std::min(min_p, p_size);
        }
        avg_p = static_cast<uint32_t>(static_cast<float>(avg_p) /
                                      static_cast<float>(m_num_patches));
    }
    uint32_t get_num_components() const
    {
        return m_num_components;
    }

    double get_ribbon_overhead() const
    {
        return 100.0 * double(get_num_ext_ribbon_faces()) / double(m_num_faces);
    }

    float get_patching_time() const
    {
        return m_patching_time_ms;
    }

    uint32_t get_num_lloyd_run() const
    {
        return m_num_lloyd_run;
    }

    void save(std::string filename)
    {
        std::ofstream                       ss(filename, std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(ss);
        archive(*this);
    }


    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(CEREAL_NVP(m_patch_size),
                CEREAL_NVP(m_num_patches),
                CEREAL_NVP(m_num_vertices),
                CEREAL_NVP(m_num_edges),
                CEREAL_NVP(m_num_faces),
                CEREAL_NVP(m_num_seeds),
                CEREAL_NVP(m_max_num_patches),
                CEREAL_NVP(m_num_components),
                CEREAL_NVP(m_num_lloyd_run),
                CEREAL_NVP(m_face_patch),
                CEREAL_NVP(m_vertex_patch),
                CEREAL_NVP(m_edge_patch),
                CEREAL_NVP(m_patches_val),
                CEREAL_NVP(m_patches_offset),
                CEREAL_NVP(m_ribbon_ext_val),
                CEREAL_NVP(m_ribbon_ext_offset),
                CEREAL_NVP(m_patching_time_ms));
    }

   private:
    /**
     * @brief Allocate various auxiliary memory needed to store patches info on
     * the host
     */
    void allocate_memory(std::vector<uint32_t>& seeds);

    /**
     * @brief Allocate various temporarily memory on the device needed to
     * compute patches on the device
     * @param ff_offset offset indicate start (and end) to index ff_values to
     * get face-incident-faces
     * @param ff_values stores face-incident-faces in compressed format
     */
    void allocate_device_memory(const std::vector<uint32_t>& seeds,
                                const std::vector<uint32_t>& ff_offset,
                                const std::vector<uint32_t>& ff_values,
                                uint32_t*&                   d_face_patch,
                                uint32_t*&                   d_queue,
                                uint32_t*&                   d_queue_ptr,
                                uint32_t*&                   d_ff_values,
                                uint32_t*&                   d_ff_offset,
                                void*&     d_cub_temp_storage_scan,
                                void*&     d_cub_temp_storage_max,
                                size_t&    cub_scan_bytes,
                                size_t&    cub_max_bytes,
                                uint32_t*& d_seeds,
                                uint32_t*& d_new_num_patches,
                                uint32_t*& d_max_patch_size,
                                uint32_t*& d_patches_offset,
                                uint32_t*& d_patches_size,
                                uint32_t*& d_patches_val);

    void grid(const std::vector<std::vector<uint32_t>>& fv);


    /**
     * @brief form initial face assigement, compute the compressed storage of 
     * the patches (i.e., populate m_patches_val and m_patches_offset)
     */
    void compute_inital_compressed_patches();

    void assign_patch(
        const std::vector<std::vector<uint32_t>>&                 fv,
        const std::unordered_map<std::pair<uint32_t, uint32_t>,
                                 uint32_t,
                                 ::rxmesh::detail::edge_key_hash> edges_map);

    void initialize_random_seeds(std::vector<uint32_t>&       seeds,
                                 const std::vector<uint32_t>& ff_offset,
                                 const std::vector<uint32_t>& ff_values);

    void get_multi_components(std::vector<std::vector<uint32_t>>& components,
                              const std::vector<uint32_t>&        ff_offset,
                              const std::vector<uint32_t>&        ff_values);

    void initialize_random_seeds_single_component(std::vector<uint32_t>& seeds);
    void generate_random_seed_from_component(std::vector<uint32_t>& seeds,
                                             std::vector<uint32_t>& component,
                                             uint32_t               num_seeds);

    void extract_ribbons(const std::vector<std::vector<uint32_t>>& fv,
                         const std::vector<uint32_t>&              ff_offset,
                         const std::vector<uint32_t>&              ff_values);

    uint32_t construct_patches_compressed_format(uint32_t* d_face_patch,
                                                 void*  d_cub_temp_storage_scan,
                                                 void*  d_cub_temp_storage_max,
                                                 size_t cub_scan_bytes,
                                                 size_t cub_max_bytes,
                                                 uint32_t* d_max_patch_size,
                                                 uint32_t* d_patches_offset,
                                                 uint32_t* d_patches_size,
                                                 uint32_t* d_patches_val);

    void run_lloyd(uint32_t* d_face_patch,
                   uint32_t* d_queue,
                   uint32_t* d_queue_ptr,
                   uint32_t* d_ff_values,
                   uint32_t* d_ff_offset,
                   void*     d_cub_temp_storage_scan,
                   void*     d_cub_temp_storage_max,
                   size_t    cub_scan_bytes,
                   size_t    cub_max_bytes,
                   uint32_t* d_seeds,
                   uint32_t* d_new_num_patches,
                   uint32_t* d_max_patch_size,
                   uint32_t* d_patches_offset,
                   uint32_t* d_patches_size,
                   uint32_t* d_patches_val);

    void bfs(const std::vector<uint32_t>& ff_offset,
             const std::vector<uint32_t>& ff_values);

    void metis_kway(const std::vector<uint32_t>& ff_offset,
                    const std::vector<uint32_t>& ff_values);

    void calc_edge_cut(const std::vector<std::vector<uint32_t>>& fv,
                       const std::vector<uint32_t>&              ff_offset,
                       const std::vector<uint32_t>&              ff_values);

    uint32_t m_patch_size, m_num_patches, m_num_vertices, m_num_edges,
        m_num_faces, m_num_seeds, m_max_num_patches, m_num_components,
        m_num_lloyd_run;

    // store the face, vertex, edge patch
    std::vector<uint32_t> m_face_patch, m_vertex_patch, m_edge_patch;


    // Stores the patches in compressed format
    std::vector<uint32_t> m_patches_val, m_patches_offset;

    // Stores ribbon in compressed format
    std::vector<uint32_t> m_ribbon_ext_val, m_ribbon_ext_offset;

    // caching the time taken to construct the patches
    float m_patching_time_ms;
};

}  // namespace patcher
}  // namespace rxmesh