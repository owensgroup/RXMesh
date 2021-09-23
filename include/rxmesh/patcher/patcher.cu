#include <assert.h>
#include <stdint.h>
#include <functional>
#include <iomanip>
#include <queue>
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_scan.cuh"
#include "cuda_profiler_api.h"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patcher/patcher.h"
#include "rxmesh/patcher/patcher_kernel.cuh"
#include "rxmesh/util/export_tools.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/timer.h"
#include "rxmesh/util/util.h"


extern std::vector<std::vector<float>> Verts;  // TODO remove this
namespace rxmesh {


namespace patcher {

Patcher::Patcher(uint32_t                                  patch_size,
                 const std::vector<std::vector<uint32_t>>& fvn,
                 const uint32_t                            num_vertices,
                 const uint32_t                            num_edges,
                 const bool is_multi_component /* = true*/,
                 const bool quite /*=true*/)
    : m_patch_size(patch_size),
      m_fvn(fvn),
      m_num_vertices(num_vertices),
      m_num_edges(num_edges),
      m_num_faces(fvn.size()),
      m_num_seeds(0),
      m_max_num_patches(0),
      m_is_multi_component(is_multi_component),
      m_quite(quite),
      m_num_components(0),
      m_patching_time_ms(0)
{

    m_num_patches =
        m_num_faces / m_patch_size + ((m_num_faces % m_patch_size) ? 1 : 0);

    m_max_num_patches = 5 * m_num_patches;

    m_num_seeds = m_num_patches;

    mem_alloc();
}

void Patcher::mem_alloc()
{

    // patches assigned to each face, vertex, and edge
    m_face_patch.resize(m_num_faces);
    std::fill(m_face_patch.begin(), m_face_patch.end(), INVALID32);

    m_vertex_patch.resize(m_num_vertices);
    std::fill(m_vertex_patch.begin(), m_vertex_patch.end(), INVALID32);

    m_edge_patch.resize(m_num_edges);
    std::fill(m_edge_patch.begin(), m_edge_patch.end(), INVALID32);

    // explicit patches in compressed format
    m_patches_val.resize(m_num_faces);

    // we allow upto double the number of faces due to patch bisecting
    m_patches_offset.resize(m_max_num_patches);

    // used to track the frontier and current seeds
    m_frontier.resize(m_num_faces, INVALID32);
    m_tf.resize(3);
    m_seeds.reserve(m_num_seeds);

    // external ribbon. it assumes first that all faces will be in there and
    // then shrink to fit after the construction is done
    m_ribbon_ext_offset.resize(m_max_num_patches, 0);

    m_ribbon_ext_val.resize(m_num_faces);
}

Patcher::~Patcher()
{
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
        m_patching_time_ms / float(m_num_lloyd_run));

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

    /*std::string filename = "patch_dist.txt";
    filename = STRINGIFY(OUTPUT_DIR) + filename;
    std::fstream file(filename.c_str(), std::ios::out);
    file.precision(15);
    for (uint32_t p = 0; p < m_num_patches; p++) {
        uint32_t p_size =
            m_patches_offset[p] - ((p == 0) ? 0 : m_patches_offset[p - 1]);
        file << p_size << "\n";
    }
    file.close();*/
}

template <class T_d>
void Patcher::export_ext_ribbon(const std::vector<std::vector<T_d>>& Verts,
                                int                                  patch_id)
{
    uint32_t start = ((patch_id == 0) ? 0 : m_ribbon_ext_offset[patch_id - 1]);
    export_face_list("ribbon_ext" + std::to_string(patch_id) + ".obj",
                     m_fvn,
                     Verts,
                     m_ribbon_ext_offset[patch_id] - start,
                     m_ribbon_ext_val.data() + start);
}

template <class T_d>
void Patcher::export_patches(const std::vector<std::vector<T_d>>& Verts)
{
    export_attribute_VTK("patches.vtk",
                         m_fvn,
                         Verts,
                         1,
                         m_face_patch.data(),
                         m_vertex_patch.data(),
                         false);

    /*if (!m_vertex_patch.empty()) {
        export_as_cubes_VTK(
            "patches_vertex.vtk", m_num_vertices, 0.05f, m_vertex_patch.data(),
            [&Verts](uint32_t i) { return Verts[i][0]; },
            [&Verts](uint32_t i) { return Verts[i][1]; },
            [&Verts](uint32_t i) { return Verts[i][2]; }, m_num_patches, false);
    }*/
}


template <class T_d>
void Patcher::export_components(
    const std::vector<std::vector<T_d>>&      Verts,
    const std::vector<std::vector<uint32_t>>& components)
{

    uint32_t           num_components = components.size();
    std::vector<float> rand_color(num_components + 1);
    for (uint32_t i = 0; i < num_components; ++i) {
        rand_color[i] = float(rand()) / float(RAND_MAX);
    }
    rand_color[num_components] = 0.0f;
    std::vector<uint32_t> face_component(m_num_faces, INVALID32);
    uint32_t              comp_id = 0;
    for (const auto& comp : components) {
        for (const auto& cf : comp) {
            assert(face_component[cf] == INVALID32);
            face_component[cf] = comp_id;
        }
        ++comp_id;
    }
    export_attribute_VTK("components.vtk",
                         m_fvn,
                         Verts,
                         1,
                         face_component.data(),
                         face_component.data(),
                         num_components,
                         false,
                         rand_color.data());
}


template <class T_d>
void Patcher::export_single_patch(const std::vector<std::vector<T_d>>& Verts,
                                  int                                  patch_id)
{

    std::vector<uint32_t> vf1(3);

    std::string filename =
        STRINGIFY(OUTPUT_DIR) + ("patch_" + std::to_string(patch_id) + ".obj");

    std::fstream file(filename, std::ios::out);

    for (uint32_t i = 0; i < Verts.size(); ++i) {
        file << "v " << Verts[i][0] << " " << Verts[i][1] << " " << Verts[i][2]
             << std::endl;
    }

    uint32_t p_start = (patch_id == 0) ? 0 : m_patches_offset[patch_id - 1];
    uint32_t p_end   = m_patches_offset[patch_id];

    for (uint32_t fb = p_start; fb < p_end; ++fb) {
        uint32_t face = m_patches_val[fb];

        get_incident_vertices(face, vf1);

        file << "f " << vf1[0] + 1 << " " << vf1[1] + 1 << " " << vf1[2] + 1
             << std::endl;
    }
}

template <class T_d, typename EdgeIDFunc>
void Patcher::export_single_patch_edges(
    const std::vector<std::vector<T_d>>& Verts,
    int                                  patch_id,
    EdgeIDFunc                           get_edge_id)
{
    // export edges of that are assigned to patch_id

    std::string filename = STRINGIFY(OUTPUT_DIR) +
                           ("patch_edges_" + std::to_string(patch_id) + ".obj");

    std::fstream file(filename, std::ios::out);

    for (uint32_t i = 0; i < Verts.size(); ++i) {
        file << "v " << Verts[i][0] << " " << Verts[i][1] << " " << Verts[i][2]
             << std::endl;
    }


    std::vector<uint32_t> vf1(3);

    for (uint32_t f = 0; f < m_num_faces; ++f) {
        get_incident_vertices(f, vf1);

        uint32_t v1 = vf1.back();
        for (uint32_t v = 0; v < vf1.size(); ++v) {
            uint32_t v0 = vf1[v];

            uint32_t edge_id = get_edge_id(v0, v1);

            if (get_edge_patch_id(edge_id) == patch_id) {
                file << "f " << v0 + 1 << " " << v1 + 1 << " " << v0 + 1
                     << std::endl;
            }
            v1 = v0;
        }
    }
}

void Patcher::execute(std::function<uint32_t(uint32_t, uint32_t)> get_edge_id,
                      const std::vector<std::vector<uint32_t>>&   ef)
{

    // degenerate cases
    if (m_num_patches <= 1) {
        m_patches_offset[0] = m_num_faces;

        for (uint32_t i = 0; i < m_num_faces; ++i) {
            m_face_patch[i]  = 0;
            m_patches_val[i] = i;
        }
        m_neighbour_patches_offset.resize(1, 0);
        assign_patch(get_edge_id);
        if (!m_quite) {
            print_statistics();
        }
        return;
    }

    parallel_execute(ef);

    postprocess();

    // export_patches(Verts);
    // for (uint32_t i = 0; i < m_num_patches;++i){
    //	export_ext_ribbon(Verts, i);
    //}

    m_ribbon_ext_val.resize(m_ribbon_ext_offset[m_num_patches - 1]);

    // assign patches to vertices and edges
    assign_patch(get_edge_id);

    // export_single_patch_edges(Verts, 0, get_edge_id);


    if (!m_quite) {
        print_statistics();
    }
}

void Patcher::initialize_cluster_seeds()
{
    // cluster i.e., start from one triangle and grow in bfs style from it
    // for experiments only

    double   r = double(rand()) / double(RAND_MAX);
    uint32_t rand_face =
        static_cast<uint32_t>(r * static_cast<double>(m_num_faces - 1));
    std::queue<uint32_t> qu;
    qu.push(rand_face);

    std::vector<uint32_t> n_faces(3);
    std::vector<uint32_t> taken;
    taken.push_back(rand_face);

    while (true) {
        uint32_t current_face = qu.front();
        qu.pop();

        m_seeds.push_back(current_face);

        if (m_seeds.size() == m_num_seeds) {
            return;
        }

        get_adjacent_faces(current_face, n_faces);

        for (uint32_t i = 0; i < n_faces.size(); i++) {
            uint32_t ff = n_faces[i];
            if (ff == SPECIAL || ff == INVALID32 ||
                find_index(ff, taken) != std::numeric_limits<uint32_t>::max()) {
                continue;
            }
            qu.push(ff);
            taken.push_back(ff);
        }
    }
}

void Patcher::initialize_random_seeds()
{
    // random
    if (!m_is_multi_component) {
        initialize_random_seeds_single_component();
    } else {
        // if multi-component,
        // 1) Identify the components i.e., for each component list the faces
        // that belong to that it
        // 2) Generate number of (random) seeds in each component
        // proportional to the number of faces it contain

        std::vector<std::vector<uint32_t>> components;
        get_multi_components(components);

        // export_components(Verts, components);

        m_num_components = components.size();
        if (m_num_components == 1) {
            initialize_random_seeds_single_component();
        } else {
            if (m_num_seeds <= m_num_components) {
                // we have too many components so we increase the number of
                // seeds. this case should not be encountered frequently
                // since we generate only one seed per component
                m_num_seeds = m_num_components;
                for (auto& comp : components) {
                    generate_random_seed_from_component(comp, 1);
                }
            } else {
                // if we have more seeds to give than the number of components,
                // then first secure that we have at least one seed per
                // component then we calculate the number of extra/remaining
                // seeds that will need be added. Every component then will have
                // a weight proportional to its size that tells how many of
                // these remaining seeds it can take

                uint32_t num_remaining_seeds = m_num_seeds - m_num_components;
                uint32_t num_extra_seeds_inserted = 0;

                // sort the order of the component to be processed by their size
                std::vector<size_t> component_order(components.size());
                fill_with_sequential_numbers(component_order.data(),
                                             component_order.size());
                std::sort(component_order.begin(),
                          component_order.end(),
                          [&components](const size_t& a, const size_t& b) {
                              return components[a].size() >
                                     components[b].size();
                          });

                // process components in descending order with repsect to their
                // size
                for (size_t c = 0; c < component_order.size(); ++c) {

                    std::vector<uint32_t>& comp =
                        components[component_order[c]];

                    uint32_t size = comp.size();
                    // this weight tells how many extra faces this component
                    // have from num_remaining_seeds
                    float weight = static_cast<float>(size) /
                                   static_cast<float>(m_num_faces);
                    uint32_t component_num_seeds =
                        static_cast<uint32_t>(std::ceil(
                            weight * static_cast<float>(num_remaining_seeds)));


                    num_extra_seeds_inserted += component_num_seeds;
                    if (num_extra_seeds_inserted > num_remaining_seeds) {
                        if (num_extra_seeds_inserted - num_remaining_seeds >
                            component_num_seeds) {
                            component_num_seeds = 0;
                        } else {
                            component_num_seeds -= (num_extra_seeds_inserted -
                                                    num_remaining_seeds);
                        }
                    }

                    component_num_seeds += 1;
                    generate_random_seed_from_component(comp,
                                                        component_num_seeds);
                }
            }
        }
    }


    // export_face_list("seeds.obj", m_fvn, Verts, uint32_t(m_seeds.size()),
    //                 m_seeds.data());
}

void Patcher::initialize_random_seeds_single_component()
{
    // if not multi-component, just generate random number
    std::vector<uint32_t> rand_num(m_num_faces);
    fill_with_sequential_numbers(rand_num.data(), rand_num.size());
    random_shuffle(rand_num.data(), rand_num.size());
    m_seeds.resize(m_num_seeds);
    std::memcpy(
        m_seeds.data(), rand_num.data(), m_num_seeds * sizeof(uint32_t));
}

void Patcher::generate_random_seed_from_component(
    std::vector<uint32_t>& component,
    const uint32_t         num_seeds)
{
    // generate seeds from faces in component.
    // num_seeds is the number of seeds that will be generated
    uint32_t num_seeds_before = m_seeds.size();
    if (num_seeds < 1) {
        RXMESH_ERROR(
            "Patcher::generate_random_seed_in_component() num_seeds should be "
            "larger than 1");
    }

    random_shuffle(component.data(), component.size());
    m_seeds.resize(num_seeds_before + num_seeds);
    std::memcpy(m_seeds.data() + num_seeds_before,
                component.data(),
                num_seeds * sizeof(uint32_t));
}


void Patcher::get_multi_components(
    std::vector<std::vector<uint32_t>>& components)
{
    std::vector<bool>     visited(m_num_faces, false);
    std::vector<uint32_t> ff(3);
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        if (!visited[f]) {
            std::vector<uint32_t> current_component;
            // just a guess
            current_component.reserve(
                static_cast<uint32_t>(static_cast<double>(m_num_faces) / 10.0));

            std::queue<uint32_t> face_queue;
            face_queue.push(f);
            while (!face_queue.empty()) {
                uint32_t current_face = face_queue.front();
                face_queue.pop();
                get_adjacent_faces(current_face, ff);

                for (const auto& f : ff) {
                    if (!visited[f]) {
                        current_component.push_back(f);
                        face_queue.push(f);
                        visited[f] = true;
                    }
                }
            }

            components.push_back(current_component);
        }
    }
}

void Patcher::postprocess()
{
    // Post process the patches by extracting the ribbons and populate the
    // neighbour patches storage
    //
    // For patch P, we start first by identifying boundary faces; faces that has
    // an edge on P's boundary. These faces are captured by querying the
    // adjacent faces for each face in P. If any of these adjacent faces are not
    // in the same patch, then this face is a boundary face. From these boundary
    // faces we can extract boundary vertices. We also now know which patch is
    // neighbor to P. Then we can use the boundary vertices to find the faces
    // that are incident to these vertices on the neighbor patches

    std::vector<uint32_t> bd_vertices;
    bd_vertices.reserve(m_patch_size);
    std::vector<uint32_t> vf1(3), vf2(3);

    m_neighbour_patches_offset.resize(m_num_patches);
    m_neighbour_patches.reserve(m_num_patches * 3);

    // build vertex incident faces
    std::vector<std::vector<uint32_t>> vertex_incident_faces(
        m_num_vertices, std::vector<uint32_t>(10));
    for (uint32_t i = 0; i < vertex_incident_faces.size(); ++i) {
        vertex_incident_faces[i].clear();
    }
    for (uint32_t face = 0; face < m_num_faces; ++face) {
        get_incident_vertices(face, vf1);
        for (uint32_t v = 0; v < vf1.size(); ++v) {
            vertex_incident_faces[vf1[v]].push_back(face);
        }
    }

    for (uint32_t cur_p = 0; cur_p < m_num_patches; ++cur_p) {

        uint32_t p_start = (cur_p == 0) ? 0 : m_patches_offset[cur_p - 1];
        uint32_t p_end   = m_patches_offset[cur_p];

        m_neighbour_patches_offset[cur_p] =
            (cur_p == 0) ? 0 : m_neighbour_patches_offset[cur_p - 1];
        uint32_t neighbour_patch_start = m_neighbour_patches_offset[cur_p];

        bd_vertices.clear();
        m_frontier.clear();


        //***** Pass One
        // 1) build a frontier of the boundary faces by loop over all faces and
        // add those that has an edge on the patch boundary
        for (uint32_t fb = p_start; fb < p_end; ++fb) {
            uint32_t face = m_patches_val[fb];

            get_adjacent_faces(face, m_tf);

            bool added = false;
            for (uint32_t g = 0; g < m_tf.size(); ++g) {
                uint32_t n       = m_tf[g];
                uint32_t n_patch = get_face_patch_id(n);

                // n is boundary face if its patch is not the current patch we
                // are processing
                if (n_patch != cur_p) {
                    if (!added) {
                        m_frontier.push_back(face);
                        added = true;
                    }

                    // add n_patch as a neighbour patch to the current patch
                    auto itt = std::find(
                        m_neighbour_patches.begin() + neighbour_patch_start,
                        m_neighbour_patches.end(),
                        n_patch);

                    if (itt == m_neighbour_patches.end()) {
                        m_neighbour_patches.push_back(n_patch);
                        ++m_neighbour_patches_offset[cur_p];
                        assert(m_neighbour_patches_offset[cur_p] ==
                               m_neighbour_patches.size());
                    }

                    // find/add the boundary vertices; these are the vertices
                    // that are shared between face and n
                    get_incident_vertices(face, vf1);
                    get_incident_vertices(n, vf2);

                    // add the common vertices in vf1 and vf2
                    for (uint32_t i = 0; i < vf1.size(); ++i) {
                        auto it_vf = std::find(vf2.begin(), vf2.end(), vf1[i]);
                        if (it_vf != vf2.end()) {
                            bd_vertices.push_back(vf1[i]);
                        }
                    }


                    // we don't break out of this loop because we want to get
                    // all the neighbour patches and boundary vertices
                    // break;
                }
            }
        }

        // Sort boundary vertices so we can use binary_search
        std::sort(bd_vertices.begin(), bd_vertices.end());
        // remove duplicated vertices
        inplace_remove_duplicates_sorted(bd_vertices);


        // export_as_cubes("cubes" + std::to_string(cur_p) + ".obj",
        //    bd_vertices.size(), 0.01f,
        //    [&bd_vertices](uint32_t i) {return Verts[bd_vertices[i]][0]; },
        //    [&bd_vertices](uint32_t i) {return Verts[bd_vertices[i]][1]; },
        //    [&bd_vertices](uint32_t i) {return Verts[bd_vertices[i]][2]; });


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
}

void Patcher::get_adjacent_faces(uint32_t               face_id,
                                 std::vector<uint32_t>& ff) const
{
    if (m_fvn.size() != 0) {
        // We account here for non-manifold cases where a face might not be
        // adjacent to just three faces
        uint32_t size = m_fvn[face_id].size() - 3;
        ff.resize(size);
        std::memcpy(
            ff.data(), m_fvn[face_id].data() + 3, size * sizeof(uint32_t));
    } else {
        RXMESH_ERROR(
            "Patcher::get_adjacent_faces() can not get adjacent faces!!");
    }
}

void Patcher::get_incident_vertices(uint32_t face_id, std::vector<uint32_t>& fv)
{
    if (m_fvn.size() != 0) {
        fv.resize(3);
        std::memcpy(fv.data(), m_fvn[face_id].data(), 3 * sizeof(uint32_t));
    } else {
        RXMESH_ERROR(
            "Patcher::get_incident_vertices() can not get adjacent faces!!");
    }
}

void Patcher::assign_patch(
    std::function<uint32_t(uint32_t, uint32_t)> get_edge_id)
{
    // For every patch p, for every face in the patch, find the three edges
    // that bound that face, and assign them to the patch. For boundary vertices
    // and edges assign them to one patch (TODO smallest face count). For now,
    // we assign it to the first patch

    std::vector<uint32_t> vf1(3);

    for (uint32_t cur_p = 0; cur_p < m_num_patches; ++cur_p) {

        uint32_t p_start = (cur_p == 0) ? 0 : m_patches_offset[cur_p - 1];
        uint32_t p_end   = m_patches_offset[cur_p];

        for (uint32_t f = p_start; f < p_end; ++f) {

            uint32_t face = m_patches_val[f];

            get_incident_vertices(face, vf1);

            uint32_t v1 = vf1.back();
            for (uint32_t v = 0; v < vf1.size(); ++v) {
                uint32_t v0 = vf1[v];

                uint32_t edge_id = get_edge_id(v0, v1);

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
}

void Patcher::populate_ff(const std::vector<std::vector<uint32_t>>& ef,
                          std::vector<uint32_t>&                    h_ff_values,
                          std::vector<uint32_t>&                    h_ff_offset)
{
    assert(ef.size() == m_num_edges);
    uint32_t                           total_ff_values = 0;
    std::vector<std::vector<uint32_t>> h_ff_values_vec;
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        std::vector<uint32_t> ff;
        ff.reserve(3);
        h_ff_values_vec.push_back(ff);
    }
    for (uint32_t e = 0; e < ef.size(); ++e) {
        for (uint32_t f0 = 0; f0 < ef[e].size() - 1; ++f0) {
            uint32_t face0 = ef[e][f0];
            for (uint32_t f1 = f0 + 1; f1 < ef[e].size(); ++f1) {
                uint32_t face1 = ef[e][f1];
                total_ff_values += 2;
                h_ff_values_vec[face0].push_back(face1);
                h_ff_values_vec[face1].push_back(face0);
            }
        }
    }

    h_ff_offset.clear();
    h_ff_offset.resize(m_num_faces);
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        uint32_t s = 0;
        if (f != 0) {
            s = h_ff_offset[f - 1];
        }
        h_ff_offset[f] = s + h_ff_values_vec[f].size();
    }
    assert(h_ff_offset.back() == total_ff_values);
    h_ff_values.clear();
    h_ff_values.reserve(total_ff_values);
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        for (uint32_t ff = 0; ff < h_ff_values_vec[f].size(); ff++) {
            h_ff_values.push_back(h_ff_values_vec[f][ff]);
        }
    }
}

void Patcher::parallel_execute(const std::vector<std::vector<uint32_t>>& ef)
{
    // TODO use streams
    // TODO we don't need ef. We only use it to compute FF which we already
    // compute in RXMesh build_local method before invoking patcher.

    // adjacent faces
    uint32_t *d_ff_values(nullptr), *d_ff_offset(nullptr);
    {
        std::vector<uint32_t> h_ff_values, h_ff_offset;
        populate_ff(ef, h_ff_values, h_ff_offset);
        assert(h_ff_offset.size() == m_num_faces);
        CUDA_ERROR(cudaMalloc((void**)&d_ff_values,
                              h_ff_values.size() * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc((void**)&d_ff_offset,
                              h_ff_offset.size() * sizeof(uint32_t)));

        CUDA_ERROR(cudaMemcpy(d_ff_values,
                              h_ff_values.data(),
                              h_ff_values.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_ff_offset,
                              h_ff_offset.data(),
                              h_ff_offset.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }


    // faces patch
    uint32_t* d_face_patch = nullptr;
    CUDA_ERROR(
        cudaMalloc((void**)&d_face_patch, m_num_faces * sizeof(uint32_t)));

    // seeds (allocate m_max_num_patches but copy only m_num_patches)
    initialize_random_seeds();
    uint32_t* d_seeds = nullptr;
    assert(m_num_patches == m_seeds.size());
    CUDA_ERROR(
        cudaMalloc((void**)&d_seeds, m_max_num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_seeds,
                          m_seeds.data(),
                          m_num_patches * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));


    // queue of size num_faces
    // queue_start and queue_end
    uint32_t* d_queue = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_queue, m_num_faces * sizeof(uint32_t)));

    // 0 -> queue start
    // 1-> queue end
    // 2-> next queue end
    std::vector<uint32_t> h_queue_ptr{0, m_num_patches, m_num_patches};
    uint32_t*             d_queue_ptr;
    CUDA_ERROR(cudaMalloc((void**)&d_queue_ptr, 3 * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_queue_ptr,
                          h_queue_ptr.data(),
                          3 * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // patches offset, values, and size
    uint32_t *d_patches_offset, *d_patches_val, *d_patches_size,
        *d_max_patch_size;
    CUDA_ERROR(cudaMalloc((void**)&d_patches_offset,
                          m_max_num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_patches_size,
                          m_max_num_patches * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&d_patches_val, m_num_faces * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_max_patch_size, sizeof(uint32_t)));
    void * d_cub_temp_storage_scan(nullptr), *d_cub_temp_storage_max(nullptr);
    size_t cub_temp_storage_bytes_scan = 0;
    size_t cub_temp_storage_bytes_max  = 0;
    ::cub::DeviceScan::InclusiveSum(d_cub_temp_storage_scan,
                                    cub_temp_storage_bytes_scan,
                                    d_patches_size,
                                    d_patches_offset,
                                    m_max_num_patches);
    ::cub::DeviceReduce::Max(d_cub_temp_storage_max,
                             cub_temp_storage_bytes_max,
                             d_patches_size,
                             d_max_patch_size,
                             m_max_num_patches);
    CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage_scan,
                          cub_temp_storage_bytes_scan));
    CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage_max,
                          cub_temp_storage_bytes_max));

    // Lloyd iterations loop
    uint32_t* d_new_num_patches = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_new_num_patches, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_new_num_patches,
                          &m_num_patches,
                          sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    /* const char separator = ' ';
    const int  numWidth = 15;
    if (!m_quite) {
        std::cout << std::endl;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator)
                  << "iter";
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator)
                  << "#";
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator)
                  << "avg";
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator)
                  << "stddev";
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator)
                  << "max";
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator)
                  << "min" << std::endl
                  << std::endl;
    }*/

    /*auto draw = [&]() {
        CUDA_ERROR(cudaDeviceSynchronize());
        CUDA_ERROR(cudaMemcpy(m_face_patch, d_face_patch,
                              m_num_faces * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < m_num_faces; ++i) {
            m_face_patch[i] = m_face_patch[i] >> 1;
        }
        export_patches(Verts);
    };*/


    CUDA_ERROR(cudaProfilerStart());
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

            /*{
            //add new seeds only to the top 10% large patches
                CUDA_ERROR(cudaMemcpy(m_patches_offset.data(), d_patches_offset,
                                      m_num_patches * sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost));
                std::vector<uint32_t> sorted_patches(m_num_patches);
                for (uint32_t p = 0; p < m_num_patches; ++p) {
                    sorted_patches[p] = (p == 0) ? 0 : m_patches_offset[p - 1];
                    sorted_patches[p] = m_patches_offset[p] - sorted_patches[p];
                }
                std::sort(sorted_patches.begin(), sorted_patches.end());
                auto     dd = std::upper_bound(sorted_patches.begin(),
                                           sorted_patches.end(), m_patch_size);
                uint32_t large_patches_start = dd - sorted_patches.begin();
                uint32_t large_patches_num =
                    sorted_patches.size() - large_patches_start;
                threshold = sorted_patches[sorted_patches.size() -
                                           0.1 * large_patches_num] -
                            1;
            }*/

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
                    "Patcher::parallel_execute() m_num_patches exceeds "
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

        rxmesh::memcpy<<<blocks_s, threads_s>>>(
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
            construct_patches_compressed_parallel(d_cub_temp_storage_max,
                                                  cub_temp_storage_bytes_max,
                                                  d_patches_size,
                                                  d_max_patch_size,
                                                  d_cub_temp_storage_scan,
                                                  cub_temp_storage_bytes_scan,
                                                  d_patches_offset,
                                                  d_face_patch,
                                                  d_patches_val);

        // draw();


        /*uint32_t* d_second_queue;
        {
            CUDA_ERROR(cudaMalloc((void**)&d_second_queue,
                                  m_num_faces * sizeof(uint32_t)));
            CUDA_ERROR(
                cudaMemset(d_second_queue, 0, m_num_faces * sizeof(uint32_t)));
        }*/

        // Interior
        uint32_t threads_i   = 512;
        uint32_t shmem_bytes = max_patch_size * (sizeof(uint32_t));
        rxmesh::memset<<<blocks_f, threads_f>>>(
            d_queue, INVALID32, m_num_faces);
        interior<<<m_num_patches, threads_i, shmem_bytes>>>(
            m_num_patches,
            d_patches_offset,
            d_patches_val,
            d_face_patch,
            d_seeds,
            d_ff_offset,
            d_ff_values,
            d_queue /*, d_second_queue*/);

        /*{
            std::vector<uint32_t> second_queue(m_num_faces);
            CUDA_ERROR(cudaMemcpy(second_queue.data(), d_second_queue,
                                  m_num_faces * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            std::vector<uint32_t> face_list;
            for (uint32_t i = 0; i < second_queue.size(); ++i) {
                if (second_queue[i] == 1) {
                    face_list.push_back(i);
                }
            }
            export_face_list("second_queue.obj", m_fvn, Verts,
                             uint32_t(face_list.size()), face_list.data());
        }*/
        /*{
            printf("\n d_face_patch");
            ::rxmesh::print_arr_host<<<1, 1>>>(m_num_faces, d_face_patch);
            CUDA_ERROR(cudaDeviceSynchronize());
        }*/

        /* if (!m_quite) {
            CUDA_ERROR(cudaDeviceSynchronize());
            double   my_avg(0), my_stddev(0);
            uint32_t my_max(0), my_min(0);
            CUDA_ERROR(cudaMemcpy(m_patches_offset.data(), d_patches_offset,
                                  m_num_patches * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            compute_avg_stddev_max_min_rs(m_patches_offset.data(),
                                          m_num_patches, my_avg, my_stddev,
                                          my_max, my_min);
            std::cout << std::left << std::setw(numWidth)
                      << std::setfill(separator) << m_num_lloyd_run;
            std::cout << std::left << std::setw(numWidth)
                      << std::setfill(separator) << m_num_patches;
            std::cout << std::left << std::setw(numWidth)
                      << std::setfill(separator) << my_avg;
            std::cout << std::left << std::setw(numWidth)
                      << std::setfill(separator) << my_stddev;
            std::cout << std::left << std::setw(numWidth)
                      << std::setfill(separator) << my_max;
            std::cout << std::left << std::setw(numWidth)
                      << std::setfill(separator) << my_min << std::endl;
        }*/

        if (max_patch_size < m_patch_size) {
            break;
        }
    }


    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    m_patching_time_ms = timer.elapsed_millis();
    CUDA_ERROR(cudaProfilerStop());


    // move data to host
    m_num_seeds = m_num_patches;
    m_seeds.resize(m_num_seeds);
    CUDA_ERROR(cudaMemcpy(m_seeds.data(),
                          d_seeds,
                          m_num_seeds * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
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


    // draw();

    for (uint32_t i = 0; i < m_num_faces; ++i) {
        m_face_patch[i]  = m_face_patch[i] >> 1;
        m_patches_val[i] = m_patches_val[i] >> 1;
    }


    GPU_FREE(d_ff_values);
    GPU_FREE(d_ff_offset);
    GPU_FREE(d_face_patch);
    GPU_FREE(d_seeds);
    GPU_FREE(d_queue);
    GPU_FREE(d_patches_offset);
    GPU_FREE(d_patches_size);
    GPU_FREE(d_patches_val);
    GPU_FREE(d_queue_ptr);
    GPU_FREE(d_cub_temp_storage_scan);
    GPU_FREE(d_cub_temp_storage_max);
    GPU_FREE(d_max_patch_size);
    GPU_FREE(d_new_num_patches);
}

uint32_t Patcher::construct_patches_compressed_parallel(
    void*     d_cub_temp_storage_max,
    size_t    cub_temp_storage_bytes_max,
    uint32_t* d_patches_size,
    uint32_t* d_max_patch_size,
    void*     d_cub_temp_storage_scan,
    size_t    cub_temp_storage_bytes_scan,
    uint32_t* d_patches_offset,
    uint32_t* d_face_patch,
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
                             cub_temp_storage_bytes_max,
                             d_patches_size,
                             d_max_patch_size,
                             m_num_patches);
    CUDA_ERROR(cudaMemcpy(&max_patch_size,
                          d_max_patch_size,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // Construct compressed patches
    ::cub::DeviceScan::InclusiveSum(d_cub_temp_storage_scan,
                                    cub_temp_storage_bytes_scan,
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

template void Patcher::export_single_patch(
    const std::vector<std::vector<double>>& Verts,
    int                                     patch_id);
template void Patcher::export_single_patch(
    const std::vector<std::vector<float>>& Verts,
    int                                    patch_id);
template void Patcher::export_patches(
    const std::vector<std::vector<double>>& Verts);
template void Patcher::export_patches(
    const std::vector<std::vector<float>>& Verts);
template void Patcher::export_ext_ribbon(
    const std::vector<std::vector<double>>& Verts,
    int                                     patch_id);
template void Patcher::export_ext_ribbon(
    const std::vector<std::vector<float>>& Verts,
    int                                    patch_id);

}  // namespace patcher
}  // namespace rxmesh