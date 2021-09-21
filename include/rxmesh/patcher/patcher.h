#pragma once

#include <stdint.h>
#include <functional>
namespace RXMESH {

namespace PATCHER {

class Patcher
{
   public:
    Patcher(uint32_t                                  patch_size,
            const std::vector<std::vector<uint32_t>>& fvn,
            const uint32_t                            num_vertices,
            const uint32_t                            num_edges,
            const bool                                is_multi_component = true,
            const bool                                quite = true);

    void execute(std::function<uint32_t(uint32_t, uint32_t)> get_edge_id,
                 const std::vector<std::vector<uint32_t>>&   ef);

    template <class T_d>
    void export_patches(const std::vector<std::vector<T_d>>& Verts);

    template <class T_d>
    void export_components(
        const std::vector<std::vector<T_d>>&      Verts,
        const std::vector<std::vector<uint32_t>>& components);

    template <class T_d>
    void export_ext_ribbon(const std::vector<std::vector<T_d>>& Verts,
                           int                                  patch_id);

    template <class T_d>
    void export_single_patch(const std::vector<std::vector<T_d>>& Verts,
                             int                                  patch_id);

    template <class T_d, typename EdgeIDFunc>
    void export_single_patch_edges(const std::vector<std::vector<T_d>>& Verts,
                                   int        patch_id,
                                   EdgeIDFunc get_edge_id);
    void print_statistics();


    //********************** Getter
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

    uint32_t* get_neighbour_patches()
    {
        return m_neighbour_patches.data();
    }

    uint32_t* get_neighbour_patches_offset()
    {
        return m_neighbour_patches_offset.data();
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
    //**************************************************************************


    ~Patcher();

   private:
    void mem_alloc();

    void assign_patch(std::function<uint32_t(uint32_t, uint32_t)> get_edge_id);

    void initialize_cluster_seeds();
    void initialize_random_seeds();
    void get_multi_components(std::vector<std::vector<uint32_t>>& components);

    void initialize_random_seeds_single_component();
    void generate_random_seed_from_component(std::vector<uint32_t>& component,
                                             uint32_t               num_seeds);

    void postprocess();
    void get_adjacent_faces(uint32_t face_id, std::vector<uint32_t>& ff) const;
    void get_incident_vertices(uint32_t face_id, std::vector<uint32_t>& fv);

    void     populate_ff(const std::vector<std::vector<uint32_t>>& ef,
                         std::vector<uint32_t>&                    h_ff_values,
                         std::vector<uint32_t>&                    h_ff_offset);
    uint32_t construct_patches_compressed_parallel(
        void*     d_cub_temp_storage_max,
        size_t    cub_temp_storage_bytes_max,
        uint32_t* d_patches_size,
        uint32_t* d_max_patch_size,
        void*     d_cub_temp_storage_scan,
        size_t    cub_temp_storage_bytes_scan,
        uint32_t* d_patches_offset,
        uint32_t* d_face_patch,
        uint32_t* d_patches_val);
    void parallel_execute(const std::vector<std::vector<uint32_t>>& ef);
    //********

    const std::vector<std::vector<uint32_t>>& m_fvn;

    uint32_t m_patch_size;
    uint32_t m_num_patches, m_num_vertices, m_num_edges, m_num_faces,
        m_num_seeds, m_max_num_patches;

    // store the face, vertex, edge patch
    std::vector<uint32_t> m_face_patch, m_vertex_patch, m_edge_patch;

    bool m_is_multi_component;
    bool m_quite;

    uint32_t m_num_components;

    // Stores the patches in compressed format
    std::vector<uint32_t> m_patches_val, m_patches_offset;

    // Stores ribbon in compressed format
    std::vector<uint32_t> m_ribbon_ext_val, m_ribbon_ext_offset;

    // Stores neighbour patches in compressed format
    std::vector<uint32_t> m_neighbour_patches, m_neighbour_patches_offset;

    // caching the time taken to construct the patches
    float m_patching_time_ms;

    // utility vectors
    std::vector<uint32_t> m_frontier, m_tf, m_seeds;
    uint32_t              m_num_lloyd_run = 0;
    //********
};

}  // namespace PATCHER
}  // namespace RXMESH