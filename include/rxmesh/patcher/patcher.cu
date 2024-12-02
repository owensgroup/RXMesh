#include <assert.h>
#include <stdint.h>
#include <functional>
#include <iomanip>
#include <queue>
#include <unordered_map>
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_scan.cuh"
#include "cuda_profiler_api.h"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patcher/patcher.h"
#include "rxmesh/patcher/patcher_kernel.cuh"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/timer.h"
#include "rxmesh/util/util.h"

#include "metis.h"

namespace rxmesh {


namespace patcher {

Patcher::Patcher(std::string filename)
{
    RXMESH_TRACE("Patcher: Reading {}", filename);
    std::ifstream                      is(filename, std::ios::binary);
    cereal::PortableBinaryInputArchive archive(is);
    archive(*this);
    print_statistics();
}

Patcher::Patcher(uint32_t                                        patch_size,
                 const std::vector<uint32_t>&                    ff_offset,
                 const std::vector<uint32_t>&                    ff_values,
                 const std::vector<std::vector<uint32_t>>&       fv,
                 const std::unordered_map<std::pair<uint32_t, uint32_t>,
                                          uint32_t,
                                          detail::edge_key_hash> edges_map,
                 const uint32_t                                  num_vertices,
                 const uint32_t                                  num_edges,
                 bool                                            use_metis)
    : m_patch_size(patch_size),
      m_num_patches(0),
      m_num_vertices(num_vertices),
      m_num_edges(num_edges),
      m_num_faces(fv.size()),
      m_num_seeds(0),
      m_max_num_patches(0),
      m_num_components(0),
      m_num_lloyd_run(0),
      m_patching_time_ms(0.0)

{

    m_num_patches =
        m_num_faces / m_patch_size + ((m_num_faces % m_patch_size) ? 1 : 0);

    m_max_num_patches = 5 * m_num_patches;

    m_num_seeds = m_num_patches;
    std::vector<uint32_t> seeds;

    uint32_t* d_face_patch            = nullptr;
    uint32_t* d_queue                 = nullptr;
    uint32_t* d_queue_ptr             = nullptr;
    uint32_t* d_ff_values             = nullptr;
    uint32_t* d_ff_offset             = nullptr;
    void*     d_cub_temp_storage_scan = nullptr;
    void*     d_cub_temp_storage_max  = nullptr;
    size_t    cub_scan_bytes          = 0;
    size_t    cub_max_bytes           = 0;
    uint32_t* d_seeds                 = nullptr;
    uint32_t* d_new_num_patches       = nullptr;
    uint32_t* d_max_patch_size        = nullptr;
    uint32_t* d_patches_offset        = nullptr;
    uint32_t* d_patches_size          = nullptr;
    uint32_t* d_patches_val           = nullptr;

    allocate_memory(seeds);

    // degenerate cases
    if (m_num_patches <= 1) {
        m_patches_offset[0] = m_num_faces;
        m_num_seeds         = 1;
        m_num_components    = 1;
        m_num_lloyd_run     = 0;
        for (uint32_t i = 0; i < m_num_faces; ++i) {
            m_face_patch[i]  = 0;
            m_patches_val[i] = i;
        }
        allocate_device_memory(seeds,
                               ff_offset,
                               ff_values,
                               d_face_patch,
                               d_queue,
                               d_queue_ptr,
                               d_ff_values,
                               d_ff_offset,
                               d_cub_temp_storage_scan,
                               d_cub_temp_storage_max,
                               cub_scan_bytes,
                               cub_max_bytes,
                               d_seeds,
                               d_new_num_patches,
                               d_max_patch_size,
                               d_patches_offset,
                               d_patches_size,
                               d_patches_val);
        assign_patch(fv, edges_map);
    } else {

        if (false) {
            grid(fv);
        } else {
            if (use_metis) {
                metis_kway(ff_offset, ff_values);
            } else {
                initialize_random_seeds(seeds, ff_offset, ff_values);
                allocate_device_memory(seeds,
                                       ff_offset,
                                       ff_values,
                                       d_face_patch,
                                       d_queue,
                                       d_queue_ptr,
                                       d_ff_values,
                                       d_ff_offset,
                                       d_cub_temp_storage_scan,
                                       d_cub_temp_storage_max,
                                       cub_scan_bytes,
                                       cub_max_bytes,
                                       d_seeds,
                                       d_new_num_patches,
                                       d_max_patch_size,
                                       d_patches_offset,
                                       d_patches_size,
                                       d_patches_val);
                run_lloyd(d_face_patch,
                          d_queue,
                          d_queue_ptr,
                          d_ff_values,
                          d_ff_offset,
                          d_cub_temp_storage_scan,
                          d_cub_temp_storage_max,
                          cub_scan_bytes,
                          cub_max_bytes,
                          d_seeds,
                          d_new_num_patches,
                          d_max_patch_size,
                          d_patches_offset,
                          d_patches_size,
                          d_patches_val);
            }
        }
        extract_ribbons(fv, ff_offset, ff_values);
        // bfs(ff_offset, ff_values);
        assign_patch(fv, edges_map);
    }


    calc_edge_cut(fv, ff_offset, ff_values);

    print_statistics();

    GPU_FREE(d_face_patch);
    GPU_FREE(d_queue);
    GPU_FREE(d_queue_ptr);
    GPU_FREE(d_ff_values);
    GPU_FREE(d_ff_offset);
    GPU_FREE(d_cub_temp_storage_scan);
    GPU_FREE(d_cub_temp_storage_max);
    GPU_FREE(d_seeds);
    GPU_FREE(d_new_num_patches);
    GPU_FREE(d_max_patch_size);
    GPU_FREE(d_patches_offset);
    GPU_FREE(d_patches_size);
    GPU_FREE(d_patches_val);
}

void Patcher::grid(const std::vector<std::vector<uint32_t>>& fv)
{
    // this only work if the input is a mesh coming from create_plane()
    // where are laid out sequenetially and so we can just group them using
    // their id

    // m_num_patches = DIVIDE_UP(m_num_faces, m_patch_size);
    // for (uint32_t f = 0; f < m_num_faces; ++f) {
    //     m_face_patch[f] = f / m_patch_size;
    // }

    uint32_t num_v          = std::sqrt(m_num_vertices);
    uint32_t num_v_per_part = std::floor(std::sqrt(float(m_patch_size / 2.f)));
    uint32_t num_parts      = num_v / num_v_per_part;

    m_num_patches = num_parts * num_parts;

    for (uint32_t f = 0; f < m_num_faces; ++f) {
        uint32_t minn(std::numeric_limits<uint32_t>::max());
        for (uint32_t v = 0; v < fv[f].size(); ++v) {
            minn = std::min(fv[f][v], minn);
        }

        uint32_t x = minn / num_v;
        uint32_t y = minn % num_v;

        uint32_t x_id = x / num_v_per_part;
        uint32_t y_id = y / num_v_per_part;

        uint32_t id = x_id * num_parts + y_id;

        m_face_patch[f] = id;
    }


    compute_inital_compressed_patches();
}

Patcher::~Patcher()
{
}

void Patcher::allocate_memory(std::vector<uint32_t>& seeds)
{
    seeds.reserve(m_num_seeds);

    // patches assigned to each face, vertex, and edge
    m_face_patch.resize(m_num_faces);
    std::fill(m_face_patch.begin(), m_face_patch.end(), INVALID32);

    m_vertex_patch.resize(m_num_vertices);
    std::fill(m_vertex_patch.begin(), m_vertex_patch.end(), INVALID32);

    m_edge_patch.resize(m_num_edges);
    std::fill(m_edge_patch.begin(), m_edge_patch.end(), INVALID32);

    // explicit patches in compressed format
    m_patches_val.resize(m_num_faces);

    // we allow up to double the number of faces due to patch bisecting
    m_patches_offset.resize(m_max_num_patches);

    // external ribbon. it assumes first that all faces will be in there and
    // then shrink to fit after the construction is done
    m_ribbon_ext_offset.resize(m_max_num_patches, 0);

    m_ribbon_ext_val.resize(m_num_faces);
}

void Patcher::allocate_device_memory(const std::vector<uint32_t>& seeds,
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
                                     uint32_t*& d_patches_val)
{
    // ff
    CUDA_ERROR(
        cudaMalloc((void**)&d_ff_values, ff_values.size() * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&d_ff_offset, ff_offset.size() * sizeof(uint32_t)));

    CUDA_ERROR(cudaMemcpy((void**)d_ff_values,
                          ff_values.data(),
                          ff_values.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMemcpy((void**)d_ff_offset,
                          ff_offset.data(),
                          ff_offset.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    // face/vertex/edge patch
    CUDA_ERROR(
        cudaMalloc((void**)&d_face_patch, m_num_faces * sizeof(uint32_t)));

    // seeds
    CUDA_ERROR(
        cudaMalloc((void**)&d_seeds, m_max_num_patches * sizeof(uint32_t)));

    CUDA_ERROR(cudaMemcpy((void**)d_seeds,
                          seeds.data(),
                          m_num_patches * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // utility
    // 0 -> queue start
    // 1-> queue end
    // 2-> next queue end
    std::vector<uint32_t> h_queue_ptr{0, m_num_patches, m_num_patches};
    CUDA_ERROR(cudaMalloc((void**)&d_queue, m_num_faces * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_queue_ptr, 3 * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_queue_ptr,
                          h_queue_ptr.data(),
                          3 * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // patch offset/size/value and max patch size
    CUDA_ERROR(cudaMalloc((void**)&d_patches_offset,
                          m_max_num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_patches_size,
                          m_max_num_patches * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&d_patches_val, m_num_faces * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_max_patch_size, sizeof(uint32_t)));

    CUDA_ERROR(cudaMalloc((void**)&d_new_num_patches, sizeof(uint32_t)));

    CUDA_ERROR(cudaMemcpy((void**)d_new_num_patches,
                          &m_num_patches,
                          sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // CUB temp memory
    d_cub_temp_storage_scan = nullptr;
    d_cub_temp_storage_max  = nullptr;
    cub_scan_bytes          = 0;
    cub_max_bytes           = 0;
    ::cub::DeviceScan::InclusiveSum(d_cub_temp_storage_scan,
                                    cub_scan_bytes,
                                    d_patches_size,
                                    d_patches_offset,
                                    m_max_num_patches);
    ::cub::DeviceReduce::Max(d_cub_temp_storage_max,
                             cub_max_bytes,
                             d_patches_size,
                             d_max_patch_size,
                             m_max_num_patches);
    CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage_scan, cub_scan_bytes));
    CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage_max, cub_max_bytes));
}

void Patcher::calc_edge_cut(const std::vector<std::vector<uint32_t>>& fv,
                            const std::vector<uint32_t>&              ff_offset,
                            const std::vector<uint32_t>&              ff_values)
{
    // given a graph where nodes represents faces in the mesh and two nodes
    // are connected in this graph if two faces share an edge, we calculate
    // the edge cut fo such a graph
    uint32_t face_edge_cut = 0;
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        for (uint32_t i = ff_offset[f]; i < ff_offset[f + 1]; ++i) {
            uint32_t n = ff_values[i];
            if (f < n && m_face_patch[f] != m_face_patch[n]) {
                face_edge_cut++;
            }
        }
    }

    uint32_t vertex_edge_cut = 0;

    using EdgeMapT = std::unordered_map<std::pair<uint32_t, uint32_t>,
                                        uint32_t,
                                        detail::edge_key_hash>;

    EdgeMapT edges_map;
    uint32_t num_edges = 0;

    for (uint32_t f = 0; f < m_num_faces; ++f) {
        for (uint32_t i = 0; i < fv[f].size(); ++i) {

            uint32_t v0 = fv[f][i];
            uint32_t v1 = fv[f][(i + 1) % fv[f].size()];

            std::pair<uint32_t, uint32_t> edge = detail::edge_key(v0, v1);

            auto e_iter = edges_map.find(edge);

            if (e_iter == edges_map.end()) {
                uint32_t edge_id = num_edges++;
                edges_map.insert(std::make_pair(edge, edge_id));

                if (m_vertex_patch[v0] != m_vertex_patch[v1]) {
                    vertex_edge_cut++;
                }
            }
        }
    }

    RXMESH_INFO("Patcher: (Face) Edge Cut = {}, (Vertex) Edge Cut = {} ",
                face_edge_cut,
                vertex_edge_cut);
}
void Patcher::print_statistics()
{
    RXMESH_TRACE("Patcher: num_patches = {}", m_num_patches);
    RXMESH_TRACE("Patcher: patches_size = {}", m_patch_size);
    RXMESH_TRACE("Patcher: num_components = {}", m_num_components);

    // patching time
    RXMESH_TRACE("Patcher: Num lloyd run = {}", m_num_lloyd_run);
    RXMESH_TRACE(
        "Patcher: Parallel patches construction time = {} (ms) and {} "
        "(ms/lloyd_run)",
        m_patching_time_ms,
        ((m_patching_time_ms == 0) ?
             0 :
             m_patching_time_ms / float(m_num_lloyd_run)));

    // max-min patch size
    uint32_t max_patch_size(0), min_patch_size(m_num_faces), avg_patch_size(0);
    get_max_min_avg_patch_size(min_patch_size, max_patch_size, avg_patch_size);
    RXMESH_TRACE(
        "Patcher: max_patch_size= {}, min_patch_size= {}, avg_patch_size= {}",
        max_patch_size,
        min_patch_size,
        avg_patch_size);

    RXMESH_TRACE("Patcher: number external ribbon faces = {} ({:02.2f}%)",
                 get_num_ext_ribbon_faces(),
                 get_ribbon_overhead());
}

void Patcher::initialize_random_seeds(std::vector<uint32_t>&       seeds,
                                      const std::vector<uint32_t>& ff_offset,
                                      const std::vector<uint32_t>& ff_values)
{

    // 1) Identify the components i.e., for each component list the faces
    // that belong to that it
    // 2) Generate number of (random) seeds in each component
    // proportional to the number of faces it contain

    std::vector<std::vector<uint32_t>> components;
    get_multi_components(components, ff_offset, ff_values);

    m_num_components = components.size();
    if (m_num_components == 1) {
        initialize_random_seeds_single_component(seeds);
    } else {
        if (m_num_seeds <= m_num_components) {
            // we have too many components so we increase the number of
            // seeds. this case should not be encountered frequently
            // since we generate only one seed per component
            m_num_seeds = m_num_components;
            for (auto& comp : components) {
                generate_random_seed_from_component(seeds, comp, 1);
            }
        } else {
            // if we have more seeds to give than the number of components,
            // then first secure that we have at least one seed per
            // component then we calculate the number of extra/remaining
            // seeds that will need be added. Every component then will have
            // a weight proportional to its size that tells how many of
            // these remaining seeds it can take

            uint32_t num_remaining_seeds      = m_num_seeds - m_num_components;
            uint32_t num_extra_seeds_inserted = 0;

            // sort the order of the component to be processed by their size
            std::vector<size_t> component_order(components.size());
            fill_with_sequential_numbers(component_order.data(),
                                         component_order.size());
            std::sort(component_order.begin(),
                      component_order.end(),
                      [&components](const size_t& a, const size_t& b) {
                          return components[a].size() > components[b].size();
                      });

            // process components in descending order with respect to their
            // size
            for (size_t c = 0; c < component_order.size(); ++c) {

                std::vector<uint32_t>& comp = components[component_order[c]];

                uint32_t size = comp.size();
                // this weight tells how many extra faces this component
                // have from num_remaining_seeds
                float weight =
                    static_cast<float>(size) / static_cast<float>(m_num_faces);
                uint32_t component_num_seeds = static_cast<uint32_t>(std::ceil(
                    weight * static_cast<float>(num_remaining_seeds)));


                num_extra_seeds_inserted += component_num_seeds;
                if (num_extra_seeds_inserted > num_remaining_seeds) {
                    if (num_extra_seeds_inserted - num_remaining_seeds >
                        component_num_seeds) {
                        component_num_seeds = 0;
                    } else {
                        component_num_seeds -=
                            (num_extra_seeds_inserted - num_remaining_seeds);
                    }
                }

                component_num_seeds += 1;
                generate_random_seed_from_component(
                    seeds, comp, component_num_seeds);
            }
        }
    }

    assert(m_num_patches == seeds.size());
}

void Patcher::initialize_random_seeds_single_component(
    std::vector<uint32_t>& seeds)
{
    // if not multi-component, just generate random number
    std::vector<uint32_t> rand_num(m_num_faces);
    fill_with_sequential_numbers(rand_num.data(), rand_num.size());
    random_shuffle(rand_num.data(), rand_num.size());
    seeds.resize(m_num_seeds);
    std::memcpy(seeds.data(), rand_num.data(), m_num_seeds * sizeof(uint32_t));
}

void Patcher::generate_random_seed_from_component(
    std::vector<uint32_t>& seeds,
    std::vector<uint32_t>& component,
    const uint32_t         num_seeds)
{
    // generate seeds from faces in component.
    // num_seeds is the number of seeds that will be generated
    uint32_t num_seeds_before = seeds.size();
    if (num_seeds < 1) {
        RXMESH_ERROR(
            "Patcher::generate_random_seed_in_component() num_seeds should be "
            "larger than 1");
    }

    random_shuffle(component.data(), component.size());
    seeds.resize(num_seeds_before + num_seeds);
    std::memcpy(seeds.data() + num_seeds_before,
                component.data(),
                num_seeds * sizeof(uint32_t));
}


void Patcher::get_multi_components(
    std::vector<std::vector<uint32_t>>& components,
    const std::vector<uint32_t>&        ff_offset,
    const std::vector<uint32_t>&        ff_values)
{
    std::vector<bool> visited(m_num_faces, false);
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        if (!visited[f]) {
            std::vector<uint32_t> current_component;
            // just a guess
            current_component.reserve(
                static_cast<uint32_t>(static_cast<double>(m_num_faces) / 10.0));

            std::queue<uint32_t> face_queue;
            face_queue.push(f);
            while (!face_queue.empty()) {
                uint32_t face = face_queue.front();
                face_queue.pop();
                uint32_t start = ff_offset[face];
                uint32_t end   = ff_offset[face + 1];
                for (uint32_t f = start; f < end; ++f) {
                    uint32_t n_face = ff_values[f];
                    if (!visited[n_face]) {
                        current_component.push_back(n_face);
                        face_queue.push(n_face);
                        visited[n_face] = true;
                    }
                }
            }

            components.push_back(current_component);
        }
    }
}

void Patcher::bfs(const std::vector<uint32_t>& ff_offset,
                  const std::vector<uint32_t>& ff_values)
{
    // BFS renumbering
    std::vector<uint32_t> bfs_patch_id(m_num_patches);

    std::vector<std::vector<uint32_t>> patch_neighbour;
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        std::vector<uint32_t> np;
        for (uint32_t f = (p == 0) ? 0 : m_patches_offset[p - 1];
             f < m_patches_offset[p];
             ++f) {
            uint32_t face = m_patches_val[f];
            for (uint32_t n = ff_offset[face]; n < ff_offset[face + 1]; ++n) {
                uint32_t n_face  = ff_values[n];
                uint32_t n_patch = m_face_patch[n_face];
                if (n_patch != p) {
                    if (find_index(n_patch, np) ==
                        std::numeric_limits<uint32_t>::max()) {
                        np.push_back(n_patch);
                    }
                }
            }
        }
        patch_neighbour.push_back(np);
    }
    std::vector<uint32_t> qu(1, 0);
    qu.reserve(m_num_patches);
    for (uint32_t p = 0; p < qu.size(); p++) {
        uint32_t patch      = qu[p];
        bfs_patch_id[patch] = p;
        for (uint32_t i = 0; i < patch_neighbour[patch].size(); i++) {
            uint32_t pn = patch_neighbour[patch][i];
            if (find_index(pn, qu) == std::numeric_limits<uint32_t>::max()) {
                qu.push_back(pn);
            }
        }
    }
    std::fill(m_patches_offset.begin(), m_patches_offset.end(), 0);
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        m_face_patch[f] = bfs_patch_id[m_face_patch[f]];
        m_patches_offset[m_face_patch[f]]++;
    }
    uint32_t acc = 0;
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        acc += m_patches_offset[p];
        m_patches_offset[p] = acc;
    }
    std::vector<uint32_t> temp_offset(m_num_patches, 0);
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        uint32_t p     = m_face_patch[f];
        uint32_t start = (p == 0) ? p : m_patches_offset[p - 1];
        m_patches_val[start + temp_offset[p]] = f;
        temp_offset[p]++;
    }
}

void Patcher::extract_ribbons(const std::vector<std::vector<uint32_t>>& fv,
                              const std::vector<uint32_t>& ff_offset,
                              const std::vector<uint32_t>& ff_values)
{
    // Post process the patches by extracting the ribbons
    // For patch P, we start first by identifying boundary faces; faces that has
    // an edge on P's boundary. These faces are captured by querying the
    // adjacent faces for each face in P. If any of these adjacent faces are not
    // in the same patch, then this face is a boundary face. From these boundary
    // faces we can extract boundary vertices. We also now know which patch is
    // neighbor to P. Then we can use the boundary vertices to find the faces
    // that are incident to these vertices on the neighbor patches
    std::vector<uint32_t> frontier;
    frontier.reserve(m_num_faces);

    std::vector<uint32_t> bd_vertices;
    bd_vertices.reserve(m_patch_size);

    // build vertex incident faces
    std::vector<std::vector<uint32_t>> vertex_incident_faces(
        m_num_vertices, std::vector<uint32_t>(10));
    for (uint32_t i = 0; i < vertex_incident_faces.size(); ++i) {
        vertex_incident_faces[i].clear();
    }
    for (uint32_t face = 0; face < m_num_faces; ++face) {
        for (uint32_t v = 0; v < fv[face].size(); ++v) {
            vertex_incident_faces[fv[face][v]].push_back(face);
        }
    }

    for (uint32_t cur_p = 0; cur_p < m_num_patches; ++cur_p) {

        uint32_t p_start = (cur_p == 0) ? 0 : m_patches_offset[cur_p - 1];
        uint32_t p_end   = m_patches_offset[cur_p];

        bd_vertices.clear();
        frontier.clear();


        //***** Pass One
        // 1) build a frontier of the boundary faces by loop over all faces and
        // add those that has an edge on the patch boundary
        for (uint32_t fb = p_start; fb < p_end; ++fb) {
            uint32_t face = m_patches_val[fb];

            bool     added = false;
            uint32_t start = ff_offset[face];
            uint32_t end   = ff_offset[face + 1];

            for (uint32_t g = start; g < end; ++g) {
                uint32_t n       = ff_values[g];
                uint32_t n_patch = get_face_patch_id(n);

                // n is boundary face if its patch is not the current patch we
                // are processing
                if (n_patch != cur_p) {
                    if (!added) {
                        frontier.push_back(face);
                        added = true;
                    }

                    // find/add the boundary vertices; these are the vertices
                    // that are shared between face and n

                    // add the common vertices in fv[face] and fv[n]
                    for (uint32_t i = 0; i < fv[face].size(); ++i) {
                        auto it_vf =
                            std::find(fv[n].begin(), fv[n].end(), fv[face][i]);
                        if (it_vf != fv[n].end()) {
                            bd_vertices.push_back(fv[face][i]);
                        }
                    }

                    // we don't break out of this loop because we want to get
                    // all the boundary vertices
                    // break;
                }
            }
        }

        // Sort boundary vertices so we can use binary_search
        std::sort(bd_vertices.begin(), bd_vertices.end());
        // remove duplicated vertices
        inplace_remove_duplicates_sorted(bd_vertices);


        //***** Pass Two

        // 3) for every vertex on the patch boundary, we add all the faces
        // that are incident to it and not in the current patch

        m_ribbon_ext_offset[cur_p] =
            (cur_p == 0) ? 0 : m_ribbon_ext_offset[cur_p - 1];
        uint32_t r_start = m_ribbon_ext_offset[cur_p];

        for (uint32_t v = 0; v < bd_vertices.size(); ++v) {
            uint32_t vert = bd_vertices[v];

            for (uint32_t f = 0; f < vertex_incident_faces[vert].size(); ++f) {
                uint32_t face = vertex_incident_faces[vert][f];
                if (get_face_patch_id(face) != cur_p) {
                    // make sure we have not added face before
                    bool     added = false;
                    uint32_t r_end = m_ribbon_ext_offset[cur_p];
                    for (uint32_t r = r_start; r < r_end; ++r) {
                        if (m_ribbon_ext_val[r] == face) {
                            added = true;
                            break;
                        }
                    }
                    if (!added) {

                        m_ribbon_ext_val[m_ribbon_ext_offset[cur_p]] = face;
                        m_ribbon_ext_offset[cur_p]++;
                        if (m_ribbon_ext_offset[cur_p] == m_num_faces) {
                            // need to expand m_ribbon_ext_val. This occurs
                            // mostly for small meshes with small patch size
                            // such that the amount overlap between exterior
                            // ribbon of different patches is larger than
                            // m_num_faces
                            uint32_t new_size = m_ribbon_ext_val.size() * 2;
                            m_ribbon_ext_val.resize(new_size);
                        }
                        assert(m_ribbon_ext_offset[cur_p] <=
                               m_ribbon_ext_val.size());
                    }
                }
            }
        }
    }

    m_ribbon_ext_val.resize(m_ribbon_ext_offset[m_num_patches - 1]);
}

void Patcher::assign_patch(
    const std::vector<std::vector<uint32_t>>&                 fv,
    const std::unordered_map<std::pair<uint32_t, uint32_t>,
                             uint32_t,
                             ::rxmesh::detail::edge_key_hash> edges_map)
{
    // For every patch p, for every face in the patch, find the three edges
    // that bound that face, and assign them to the patch. For boundary vertices
    // and edges assign them to one patch (TODO smallest face count). For now,
    // we assign it to the first patch

    for (uint32_t cur_p = 0; cur_p < m_num_patches; ++cur_p) {

        uint32_t p_start = (cur_p == 0) ? 0 : m_patches_offset[cur_p - 1];
        uint32_t p_end   = m_patches_offset[cur_p];

        for (uint32_t f = p_start; f < p_end; ++f) {

            uint32_t face = m_patches_val[f];

            uint32_t v1 = fv[face].back();
            for (uint32_t v = 0; v < fv[face].size(); ++v) {
                uint32_t v0 = fv[face][v];

                std::pair<uint32_t, uint32_t> key =
                    ::rxmesh::detail::edge_key(v0, v1);
                uint32_t edge_id = edges_map.at(key);

                if (m_vertex_patch[v0] == INVALID32) {
                    m_vertex_patch[v0] = cur_p;
                }

                if (m_edge_patch[edge_id] == INVALID32) {
                    m_edge_patch[edge_id] = cur_p;
                }

                v1 = v0;
            }
        }
    }


    /* for (uint32_t f = 0; f < m_num_faces; ++f) {
        uint32_t p0 = m_vertex_patch[fv[f][0]];
        uint32_t p1 = m_vertex_patch[fv[f][1]];
        uint32_t p2 = m_vertex_patch[fv[f][2]];

        if (p0 == p1 && p1 == p2 && p0 == p2) {
            // ideal case
            continue;
        }
        if (p0 != p1 && p0 != p2 && p1 != p2) {
            // hopeless case
            continue;
        }

        // find the index in fv[f] of the odd vertex i.e., the vertex with
        // different patch id
        uint32_t odd = (p0 == p1) ? 2 : ((p1 == p2) ? 0 : 1);

        // find the index in fv[f] of the common vertex i.e., any vertex other
        // than odd
        uint32_t common = (odd + 1) % 3;

        // the common patch
        uint32_t common_patch = m_vertex_patch[fv[f][common]];

        // re-assign the odd one to agree with the other two
        // only if this face is also assigned to the common patch
        uint32_t f_p = m_face_patch[f];
        if (f_p == common_patch) {
            m_vertex_patch[fv[f][odd]] = common_patch;
        }
    }*/

    // Refinement: every vertex get re-assigned to the patch where the most of
    // the vertex neighbors are assigned to
    /* std::vector<std::vector<uint32_t>> vv(m_num_vertices);

    for (auto& it : edges_map) {
        uint32_t v0 = it.first.first;
        uint32_t v1 = it.first.second;

        vv[v0].push_back(v1);
        vv[v1].push_back(v0);
    }

    for (uint32_t v = 0; v < m_num_vertices; ++v) {
        std::unordered_map<uint32_t, uint32_t> neighbour_patch;
        for (uint32_t i = 0; i < vv[v].size(); ++i) {
            uint32_t n       = vv[v][i];
            uint32_t n_patch = m_vertex_patch[n];
            neighbour_patch[n_patch] += 1;
        }

        uint32_t pop_patch       = INVALID32;
        uint32_t pop_patch_count = 0;
        for (auto& it : neighbour_patch) {
            uint32_t p       = it.first;
            uint32_t p_count = it.second;
            if (p_count > pop_patch_count) {
                pop_patch       = p;
                pop_patch_count = p_count;
            }
        }

        m_vertex_patch[v] = pop_patch;
    }*/
}

void Patcher::run_lloyd(uint32_t* d_face_patch,
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
                        uint32_t* d_patches_val)
{
    std::vector<uint32_t> h_queue_ptr{0, m_num_patches, m_num_patches};

    // CUDA_ERROR(cudaProfilerStart());
    GPUTimer timer;
    timer.start();

    m_num_lloyd_run = 0;
    while (true) {
        ++m_num_lloyd_run;

        const uint32_t threads_s = 256;
        const uint32_t blocks_s  = DIVIDE_UP(m_num_patches, threads_s);
        const uint32_t threads_f = 256;
        const uint32_t blocks_f  = DIVIDE_UP(m_num_faces, threads_f);

        // add more seeds if needed
        if (m_num_lloyd_run % 5 == 0 && m_num_lloyd_run > 0) {
            uint32_t threshold = m_patch_size;

            CUDA_ERROR(cudaMemcpy(d_new_num_patches,
                                  &m_num_patches,
                                  sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
            add_more_seeds<<<m_num_patches, 1>>>(m_num_patches,
                                                 d_new_num_patches,
                                                 d_seeds,
                                                 d_patches_offset,
                                                 d_patches_val,
                                                 threshold);

            CUDA_ERROR(cudaMemcpy(&m_num_patches,
                                  d_new_num_patches,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            if (m_num_patches >= m_max_num_patches) {
                RXMESH_ERROR(
                    "Patcher::run_lloyd() m_num_patches exceeds "
                    "m_max_num_patches");
            }
        }
        h_queue_ptr[0] = 0;
        h_queue_ptr[1] = m_num_patches;
        h_queue_ptr[2] = m_num_patches;
        CUDA_ERROR(cudaMemcpy(d_queue_ptr,
                              h_queue_ptr.data(),
                              3 * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        rxmesh::memset<<<blocks_f, threads_f>>>(
            d_face_patch, INVALID32, m_num_faces);

        rxmesh::memcopy<<<blocks_s, threads_s>>>(
            d_queue, d_seeds, m_num_patches);

        rxmesh::memset<<<blocks_s, threads_s>>>(
            d_patches_size, 0u, m_num_patches);

        write_initial_face_patch<<<blocks_s, threads_s>>>(
            m_num_patches, d_face_patch, d_seeds, d_patches_size);

        // Cluster seed propagation
        while (true) {
            // Launch enough threads to cover all the faces. However, only
            // subset will do actual work depending on the queue size
            cluster_seed_propagation<<<blocks_f, threads_f>>>(m_num_faces,
                                                              m_num_patches,
                                                              d_queue_ptr,
                                                              d_queue,
                                                              d_face_patch,
                                                              d_patches_size,
                                                              d_ff_offset,
                                                              d_ff_values);

            reset_queue_ptr<<<1, 1>>>(d_queue_ptr);

            CUDA_ERROR(cudaMemcpy(h_queue_ptr.data(),
                                  d_queue_ptr,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            if (h_queue_ptr[0] >= m_num_faces) {
                break;
            }
        }

        uint32_t max_patch_size =
            construct_patches_compressed_format(d_face_patch,
                                                d_cub_temp_storage_scan,
                                                d_cub_temp_storage_max,
                                                cub_scan_bytes,
                                                cub_max_bytes,
                                                d_max_patch_size,
                                                d_patches_offset,
                                                d_patches_size,
                                                d_patches_val);

        // Interior
        uint32_t threads_i   = 512;
        uint32_t shmem_bytes = max_patch_size * (sizeof(uint32_t));
        rxmesh::memset<<<blocks_f, threads_f>>>(
            d_queue, INVALID32, m_num_faces);
        interior<<<m_num_patches, threads_i, shmem_bytes>>>(m_num_patches,
                                                            d_patches_offset,
                                                            d_patches_val,
                                                            d_face_patch,
                                                            d_seeds,
                                                            d_ff_offset,
                                                            d_ff_values,
                                                            d_queue);

        if (max_patch_size < m_patch_size) {
            shift<<<blocks_f, threads_f>>>(
                m_num_faces, d_face_patch, d_patches_val);

            break;
        }
    }


    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    m_patching_time_ms = timer.elapsed_millis();
    // CUDA_ERROR(cudaProfilerStop());


    // move data to host
    m_num_seeds = m_num_patches;

    CUDA_ERROR(cudaMemcpy(m_face_patch.data(),
                          d_face_patch,
                          sizeof(uint32_t) * m_num_faces,
                          cudaMemcpyDeviceToHost));
    m_patches_offset.resize(m_num_patches);
    CUDA_ERROR(cudaMemcpy(m_patches_offset.data(),
                          d_patches_offset,
                          sizeof(uint32_t) * m_num_patches,
                          cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(m_patches_val.data(),
                          d_patches_val,
                          sizeof(uint32_t) * m_num_faces,
                          cudaMemcpyDeviceToHost));
}

uint32_t Patcher::construct_patches_compressed_format(
    uint32_t* d_face_patch,
    void*     d_cub_temp_storage_scan,
    void*     d_cub_temp_storage_max,
    size_t    cub_scan_bytes,
    size_t    cub_max_bytes,
    uint32_t* d_max_patch_size,
    uint32_t* d_patches_offset,
    uint32_t* d_patches_size,
    uint32_t* d_patches_val)
{
    uint32_t       max_patch_size = 0;
    const uint32_t threads_s      = 256;
    const uint32_t blocks_s       = DIVIDE_UP(m_num_patches, threads_s);
    const uint32_t threads_f      = 256;
    const uint32_t blocks_f       = DIVIDE_UP(m_num_faces, threads_f);

    // Compute max patch size
    max_patch_size = 0;
    ::cub::DeviceReduce::Max(d_cub_temp_storage_max,
                             cub_max_bytes,
                             d_patches_size,
                             d_max_patch_size,
                             m_num_patches);
    CUDA_ERROR(cudaMemcpy(&max_patch_size,
                          d_max_patch_size,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // Construct compressed patches
    ::cub::DeviceScan::InclusiveSum(d_cub_temp_storage_scan,
                                    cub_scan_bytes,
                                    d_patches_size,
                                    d_patches_offset,
                                    m_num_patches);
    rxmesh::memset<<<blocks_s, threads_s>>>(d_patches_size, 0u, m_num_patches);

    construct_patches_compressed<<<blocks_f, threads_f>>>(m_num_faces,
                                                          d_face_patch,
                                                          m_num_patches,
                                                          d_patches_offset,
                                                          d_patches_size,
                                                          d_patches_val);

    return max_patch_size;
}


void Patcher::metis_kway(const std::vector<uint32_t>& ff_offset,
                         const std::vector<uint32_t>& ff_values)
{

    std::vector<idx_t> xadj(ff_offset.size());
    std::vector<idx_t> adjncy(ff_values.size());

    for (uint32_t i = 0; i < ff_offset.size(); ++i) {
        xadj[i] = ff_offset[i];
    }

    for (uint32_t i = 0; i < ff_values.size(); ++i) {
        adjncy[i] = ff_values[i];
    }

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] =
        METIS_OBJTYPE_VOL;  // Total communication volume minimization.
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_CONTIG]    = 0;
    options[METIS_OPTION_COMPRESS]  = 0;
    options[METIS_OPTION_DBGLVL]    = METIS_DBG_TIME;

    // number of vertices in the graph
    idx_t              nvtxs  = m_num_faces;
    idx_t              ncon   = 1;
    idx_t*             vwgt   = NULL;
    idx_t*             vsize  = NULL;
    idx_t*             adjwgt = NULL;
    idx_t              nparts = DIVIDE_UP(m_num_faces, m_patch_size);
    real_t*            tpwgts = NULL;
    real_t*            ubvec  = NULL;
    idx_t              objval = 0;
    std::vector<idx_t> part(nvtxs, 0);

    CPUTimer timer;
    timer.start();

    int metis_status = METIS_PartGraphKway(&nvtxs,
                                           &ncon,
                                           xadj.data(),
                                           adjncy.data(),
                                           vwgt,
                                           vsize,
                                           adjwgt,
                                           &nparts,
                                           tpwgts,
                                           ubvec,
                                           options,
                                           &objval,
                                           part.data());
    timer.stop();
    m_patching_time_ms = timer.elapsed_millis();

    if (metis_status == METIS_ERROR_INPUT) {
        RXMESH_ERROR("METIS ERROR INPUT");
        exit(EXIT_FAILURE);
    } else if (metis_status == METIS_ERROR_MEMORY) {
        RXMESH_ERROR("\n METIS ERROR MEMORY \n");
        exit(EXIT_FAILURE);
    } else if (metis_status == METIS_ERROR) {
        RXMESH_ERROR("\n METIS ERROR\n");
        exit(EXIT_FAILURE);
    }

    m_num_patches = nparts;

    for (uint32_t f = 0; f < m_num_faces; ++f) {
        m_face_patch[f] = part[f];
    }

    compute_inital_compressed_patches();
}

void Patcher::compute_inital_compressed_patches()
{
    m_patches_offset.resize(m_num_patches, 0);

    std::vector<uint32_t> patches_size(m_num_patches, 0);
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        patches_size[m_face_patch[f]]++;
    }

    std::inclusive_scan(
        patches_size.begin(), patches_size.end(), m_patches_offset.begin());

    if (m_patches_offset.back() != m_num_faces) {
        RXMESH_ERROR(
            "Patcher::compute_inital_compressed_patches()  Error with creating "
            "patch graph");
        exit(EXIT_FAILURE);
    }

    std::fill(patches_size.begin(), patches_size.end(), 0);

    for (uint32_t f = 0; f < m_num_faces; ++f) {
        int p = m_face_patch[f];

        uint32_t id = (p == 0) ? 0 : m_patches_offset[p - 1];

        id += patches_size[p]++;

        m_patches_val[id] = f;
    }
}
}  // namespace patcher
}  // namespace rxmesh