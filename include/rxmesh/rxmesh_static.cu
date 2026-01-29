#include "rxmesh/rxmesh_static.h"

#include "rxmesh/rxmesh_static.inl"

namespace rxmesh {
RXMeshStatic::RXMeshStatic(const std::string file_path,
                           const std::string patcher_file,
                           const uint32_t    patch_size,
                           const float       capacity_factor,
                           const float       patch_alloc_factor,
                           const float       lp_hashtable_load_factor)
    : RXMesh(patch_size)
{
    m_num_regions = 1;

    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>>    vertices;
    if (!import_obj(file_path, vertices, fv)) {
        RXMESH_ERROR(
            "RXMeshStatic::RXMeshStatic could not read the input file {}",
            file_path);
        exit(EXIT_FAILURE);
    }

    this->init(fv,
               patcher_file,
               capacity_factor,
               patch_alloc_factor,
               lp_hashtable_load_factor);

    m_attr_container = std::make_shared<AttributeContainer>();

    std::string name = extract_file_name(file_path);
#if USE_POLYSCOPE
    name = polyscope::guessNiceNameFromPath(file_path);
#endif
    add_vertex_coordinates(vertices, name);
}

RXMeshStatic::RXMeshStatic(std::vector<std::vector<uint32_t>>& fv,
                           const std::string                   patcher_file,
                           const uint32_t                      patch_size,
                           const float                         capacity_factor,
                           const float patch_alloc_factor,
                           const float lp_hashtable_load_factor)
    : RXMesh(patch_size), m_input_vertex_coordinates(nullptr)
{
    m_num_regions = 1;
    this->init(fv,
               patcher_file,
               capacity_factor,
               patch_alloc_factor,
               lp_hashtable_load_factor);
    m_attr_container = std::make_shared<AttributeContainer>();
}

RXMeshStatic::RXMeshStatic(const std::vector<std::string> files_path,
                           const uint32_t                 patch_size)
    : RXMesh(patch_size)
{
    m_num_regions = static_cast<int>(files_path.size());

    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>>    vertices;

    std::vector<int> region_num_faces;
    std::vector<int> region_num_vertices;

    for (auto path : files_path) {
        if (!import_obj(path, vertices, fv, true)) {
            RXMESH_ERROR(
                "RXMeshStatic::RXMeshStatic could not read the input file "
                "{}",
                path);
            exit(EXIT_FAILURE);
        }
        region_num_faces.push_back(static_cast<int>(fv.size()));
        region_num_vertices.push_back(static_cast<int>(vertices.size()));
    }

    this->init(fv, "", 1.0, 1.0, 0.8);

    m_attr_container = std::make_shared<AttributeContainer>();

    std::string name;
    for (auto path : files_path) {
        name += extract_file_name(path);
    }

#if USE_POLYSCOPE
    name = polyscope::guessNiceNameFromPath(name);
#endif
    add_vertex_coordinates(vertices, name);

    // add region labels for faces, vertices, and edges
    m_face_label = add_face_attribute<int>("rx:face_label", 1, LOCATION_ALL);
    m_edge_label = add_edge_attribute<int>("rx:edge_label", 1, LOCATION_ALL);
    m_vertex_label =
        add_vertex_attribute<int>("rx:vertex_label", 1, LOCATION_ALL);

    for_each_face(
        HOST,
        [=](const FaceHandle fh) {
            int id = map_to_global(fh);

            auto upper = std::upper_bound(
                region_num_faces.begin(), region_num_faces.end(), id);

            int label = static_cast<int>(
                std::distance(region_num_faces.begin(), upper));

            (*m_face_label)(fh) = label;
        },
        NULL,
        false);

    for_each_vertex(
        HOST,
        [=](const VertexHandle vh) {
            int id = map_to_global(vh);

            auto upper = std::upper_bound(
                region_num_vertices.begin(), region_num_vertices.end(), id);

            int label = static_cast<int>(
                std::distance(region_num_vertices.begin(), upper));

            (*m_vertex_label)(vh) = label;
        },
        NULL,
        false);
    m_face_label->move(HOST, DEVICE);
    m_vertex_label->move(HOST, DEVICE);

    add_edge_labels(*m_face_label, *m_edge_label);

    m_edge_label->move(DEVICE, HOST);

#if USE_POLYSCOPE
    m_polyscope_mesh->addFaceScalarQuantity("rx:FLabel", *m_face_label);
    m_polyscope_mesh->addEdgeScalarQuantity("rx:ELabel", *m_edge_label);
    m_polyscope_mesh->addVertexScalarQuantity("rx:VLabel", *m_vertex_label);
#endif
}

void RXMeshStatic::add_vertex_coordinates(
    std::vector<std::vector<float>>& vertices,
    std::string                      mesh_name)
{
    if (m_input_vertex_coordinates == nullptr) {

        m_input_vertex_coordinates =
            this->add_vertex_attribute<float>(vertices, "rx:vertices");

#if USE_POLYSCOPE
        polyscope::init();
        m_polyscope_mesh_name = mesh_name.empty() ? "RXMesh" : mesh_name;
        m_polyscope_mesh_name += std::to_string(rand());
        this->register_polyscope();
        render_vertex_patch();
        render_edge_patch();
        render_face_patch();
#endif
    }
}


#if USE_POLYSCOPE
polyscope::SurfaceMesh* RXMeshStatic::get_polyscope_mesh()
{
    return m_polyscope_mesh;
}

polyscope::SurfaceMesh* RXMeshStatic::render_patch(const uint32_t p,
                                                   bool with_vertex_patch,
                                                   bool with_edge_patch,
                                                   bool with_face_patch)
{
    std::vector<std::array<uint32_t, 3>> fv;
    fv.reserve(m_h_patches_info[p].num_faces[0]);
    add_patch_to_polyscope(p, fv, true);

    auto ps = polyscope::registerSurfaceMesh(
        m_polyscope_mesh_name + "_patch_" + std::to_string(p),
        *m_input_vertex_coordinates,
        fv,
        m_polyscope_edges_map);

    if (with_vertex_patch) {
        render_vertex_patch_and_local_id(p, ps);
    }
    if (with_edge_patch) {
        render_edge_patch_and_local_id(p, ps);
    }
    if (with_face_patch) {
        render_face_patch_and_local_id(p, ps);
    }

    return ps;
}

void RXMeshStatic::render_face_patch_and_local_id(
    const uint32_t          p,
    polyscope::SurfaceMesh* polyscope_mesh)
{
    std::string      p_name = "rx:FPatch" + std::to_string(p);
    std::string      l_name = "rx:FLocal" + std::to_string(p);
    std::vector<int> patch_id(m_h_patches_info[p].num_faces[0], -1);
    std::vector<int> local_id(m_h_patches_info[p].num_faces[0], -1);

    for (uint16_t f = 0; f < this->m_h_patches_info[p].num_faces[0]; ++f) {
        const LocalFaceT lf(f);
        if (!this->m_h_patches_info[p].is_deleted(lf)) {

            const FaceHandle fh = get_owner_handle<FaceHandle>({p, lf});

            patch_id[f] = fh.patch_id();
            local_id[f] = fh.local_id();
        }
    }

    patch_id.erase(std::remove(patch_id.begin(), patch_id.end(), -1),
                   patch_id.end());
    std::pair<double, double> p_range(0.0, double(get_num_patches() - 1));
    polyscope_mesh->addFaceScalarQuantity(p_name, patch_id)
        ->setMapRange(p_range);


    local_id.erase(std::remove(local_id.begin(), local_id.end(), -1),
                   local_id.end());
    std::pair<double, double> l_range(
        0.0, double(*std::max_element(local_id.begin(), local_id.end()) - 1));
    polyscope_mesh->addFaceScalarQuantity(l_name, local_id)
        ->setMapRange(l_range);
}

void RXMeshStatic::render_edge_patch_and_local_id(
    const uint32_t          p,
    polyscope::SurfaceMesh* polyscope_mesh)
{
    std::string      p_name = "rx:EPatch" + std::to_string(p);
    std::string      l_name = "rx:ELocal" + std::to_string(p);
    std::vector<int> patch_id(get_num_edges(), -(int(p) + 1));
    std::vector<int> local_id(get_num_edges(), -(int(p) + 1));

    for_each_edge(HOST, [&](EdgeHandle eh) {
        patch_id[linear_id(eh)] = eh.patch_id();
        local_id[linear_id(eh)] = eh.local_id();
    });

    std::pair<double, double> p_range(0.0, double(get_num_patches() - 1));
    polyscope_mesh->addEdgeScalarQuantity(p_name, patch_id)
        ->setMapRange(p_range);

    std::pair<double, double> l_range(
        0.0, double(*std::max_element(local_id.begin(), local_id.end()) - 1));
    polyscope_mesh->addEdgeScalarQuantity(l_name, local_id)
        ->setMapRange(l_range);
}

void RXMeshStatic::render_vertex_patch_and_local_id(
    const uint32_t          p,
    polyscope::SurfaceMesh* polyscope_mesh)
{
    std::string      p_name = "rx:VPatch" + std::to_string(p);
    std::string      l_name = "rx:VLocal" + std::to_string(p);
    std::vector<int> patch_id(get_num_vertices(), -(int(p) + 1));
    std::vector<int> local_id(get_num_vertices(), -(int(p) + 1));

    for_each_vertex(HOST, [&](VertexHandle vh) {
        patch_id[linear_id(vh)] = vh.patch_id();
        local_id[linear_id(vh)] = vh.local_id();
    });

    std::pair<double, double> p_range(0.0, double(get_num_patches() - 1));
    polyscope_mesh->addVertexScalarQuantity(p_name, patch_id)
        ->setMapRange(p_range);

    std::pair<double, double> l_range(
        0.0, double(*std::max_element(local_id.begin(), local_id.end()) - 1));
    polyscope_mesh->addVertexScalarQuantity(l_name, local_id)
        ->setMapRange(l_range);
}

polyscope::SurfaceFaceScalarQuantity* RXMeshStatic::render_face_patch()
{
    std::string name       = "rx:FPatch";
    auto        face_patch = this->add_face_attribute<uint32_t>(name, 1, HOST);
    for_each_face(HOST,
                  [&](FaceHandle fh) { (*face_patch)(fh) = fh.patch_id(); });
    auto ret = m_polyscope_mesh->addFaceScalarQuantity(name, *face_patch);
    remove_attribute(name);

    std::pair<double, double> range(0.0, double(get_num_patches() - 1));
    ret->setMapRange(range);

    return ret;
}

polyscope::SurfaceEdgeScalarQuantity* RXMeshStatic::render_edge_patch()
{
    std::string name       = "rx:EPatch";
    auto        edge_patch = this->add_edge_attribute<uint32_t>(name, 1, HOST);
    for_each_edge(HOST,
                  [&](EdgeHandle eh) { (*edge_patch)(eh) = eh.patch_id(); });
    auto ret = m_polyscope_mesh->addEdgeScalarQuantity(name, *edge_patch);
    remove_attribute(name);

    std::pair<double, double> range(0.0, double(get_num_patches() - 1));
    ret->setMapRange(range);

    return ret;
}

polyscope::SurfaceVertexScalarQuantity* RXMeshStatic::render_vertex_patch()
{
    std::string name  = "rx:VPatch";
    auto vertex_patch = this->add_vertex_attribute<uint32_t>(name, 1, HOST);
    for_each_vertex(
        HOST, [&](VertexHandle vh) { (*vertex_patch)(vh) = vh.patch_id(); });
    auto ret = m_polyscope_mesh->addVertexScalarQuantity(name, *vertex_patch);
    remove_attribute(name);
    return ret;
}

polyscope::SurfaceFaceScalarQuantity* RXMeshStatic::render_face_local_id()
{
    std::string name    = "rx:FLocal";
    auto        f_local = add_face_attribute<uint16_t>(name, 1);

    for_each_face(HOST,
                  [&](const FaceHandle fh) { (*f_local)(fh) = fh.local_id(); });

    auto ret = m_polyscope_mesh->addFaceScalarQuantity(name, *f_local);

    remove_attribute(name);
    return ret;
}

polyscope::SurfaceEdgeScalarQuantity* RXMeshStatic::render_edge_local_id()
{
    std::string name    = "rx:ELocal";
    auto        e_local = add_edge_attribute<uint16_t>(name, 1);

    for_each_edge(HOST,
                  [&](const EdgeHandle eh) { (*e_local)(eh) = eh.local_id(); });

    auto ret = m_polyscope_mesh->addEdgeScalarQuantity(name, *e_local);

    remove_attribute(name);
    return ret;
}

polyscope::SurfaceVertexScalarQuantity* RXMeshStatic::render_vertex_local_id()
{
    std::string name    = "rx:VLocal";
    auto        v_local = add_vertex_attribute<uint16_t>(name, 1);

    for_each_vertex(
        HOST, [&](const VertexHandle vh) { (*v_local)(vh) = vh.local_id(); });

    auto ret = m_polyscope_mesh->addVertexScalarQuantity(name, *v_local);

    remove_attribute(name);
    return ret;
}

void RXMeshStatic::add_patch_to_polyscope(
    const uint32_t                        p,
    std::vector<std::array<uint32_t, 3>>& fv,
    bool                                  with_ribbon)
{
    for (uint16_t f = 0; f < this->m_h_patches_info[p].num_faces[0]; ++f) {
        if (!detail::is_deleted(f, this->m_h_patches_info[p].active_mask_f)) {
            if (!with_ribbon) {
                if (!detail::is_owned(f,
                                      this->m_h_patches_info[p].owned_mask_f)) {
                    continue;
                }
            }

            std::array<uint32_t, 3> face;
            for (uint32_t e = 0; e < 3; ++e) {
                LocalEdgeT edge = this->m_h_patches_info[p].fe[3 * f + e];
                flag_t     dir(0);
                Context::unpack_edge_dir(edge.id, edge.id, dir);
                uint16_t     eid0 = (2 * edge.id) + dir;
                VertexHandle vh(p, {this->m_h_patches_info[p].ev[eid0].id});
                uint32_t     v_org0 = linear_id(vh);
                face[e]             = v_org0;
                assert(v_org0 < get_num_vertices());
            }
            fv.push_back(face);
        }
    }
}

void RXMeshStatic::update_polyscope_edge_map()
{
    m_polyscope_edges_map.clear();

    for (uint32_t p = 0; p < this->m_num_patches; ++p) {
        for (uint16_t e = 0; e < this->m_h_patches_info[p].num_edges[0]; ++e) {
            LocalEdgeT local_e(e);
            if (!this->m_h_patches_info[p].is_deleted(local_e) &&
                this->m_h_patches_info[p].is_owned(local_e)) {

                VertexHandle v0(p,
                                {this->m_h_patches_info[p].ev[2 * e + 0].id});
                VertexHandle v1(p,
                                {this->m_h_patches_info[p].ev[2 * e + 1].id});

                uint32_t v_org0 = linear_id(v0);
                uint32_t v_org1 = linear_id(v1);

                auto key = detail::edge_key(v_org0, v_org1);
                assert(v_org0 < get_num_vertices());
                assert(v_org1 < get_num_vertices());
                auto e_iter = m_polyscope_edges_map.find(key);
                if (e_iter == m_polyscope_edges_map.end()) {
                    EdgeHandle eh(p, {e});
                    m_polyscope_edges_map.insert(
                        std::make_pair(key, linear_id(eh)));
                }
            }
        }
    }
}

void RXMeshStatic::register_polyscope()
{
    update_polyscope_edge_map();

    std::vector<std::array<uint32_t, 3>> fv;
    fv.reserve(get_num_faces());

    for (uint32_t p = 0; p < this->m_num_patches; ++p) {
        add_patch_to_polyscope(p, fv, false);
    }

    m_polyscope_mesh =
        polyscope::registerSurfaceMesh(m_polyscope_mesh_name,
                                       *m_input_vertex_coordinates,
                                       fv,
                                       m_polyscope_edges_map);
}
#endif

bool RXMeshStatic::does_attribute_exist(const std::string& name)
{
    return m_attr_container->does_exist(name.c_str());
}

void RXMeshStatic::remove_attribute(const std::string& name)
{
    if (!this->does_attribute_exist(name)) {
        RXMESH_WARN(
            "RXMeshStatic::remove_attribute() trying to remove an "
            "attribute that does not exit with name {}",
            name);
        return;
    }

    m_attr_container->remove(name.c_str());
}

std::shared_ptr<VertexAttribute<float>>
RXMeshStatic::get_input_vertex_coordinates()
{
    if (!m_input_vertex_coordinates) {
        RXMESH_ERROR(
            "RXMeshStatic::get_input_vertex_coordinates input vertex was "
            "not initialized. Call RXMeshStatic with constructor to the "
            "obj file path");
        exit(EXIT_FAILURE);
    }
    return m_input_vertex_coordinates;
}

int RXMeshStatic::get_num_regions() const
{
    return m_num_regions;
}

std::shared_ptr<FaceAttribute<int>> RXMeshStatic::get_face_region_label()
{
    if (!m_face_label) {
        RXMESH_ERROR(
            "RXMeshStatic::get_face_region_label() there is no region "
            "label.");
    }
    return m_face_label;
}

std::shared_ptr<EdgeAttribute<int>> RXMeshStatic::get_edge_region_label()
{
    if (!m_edge_label) {
        RXMESH_ERROR(
            "RXMeshStatic::get_edge_region_label() there is no region "
            "label.");
    }
    return m_edge_label;
}

std::shared_ptr<VertexAttribute<int>> RXMeshStatic::get_vertex_region_label()
{
    if (!m_vertex_label) {
        RXMESH_ERROR(
            "RXMeshStatic::get_vertex_region_label() there is no region "
            "label.");
    }
    return m_vertex_label;
}

void RXMeshStatic::scale(glm::fvec3 lower, glm::fvec3 upper)
{
    if (lower[0] > upper[0] || lower[1] > upper[1] || lower[2] > upper[2]) {
        RXMESH_ERROR(
            "RXMeshStatic::scale() can not scale the mesh since the lower "
            "corner ({},{},{}) is higher than upper corner ({},{},{}).",
            lower[0],
            lower[1],
            lower[2],
            upper[0],
            upper[1],
            upper[2]);
        return;
    }

    glm::vec3 bb_lower(0), bb_upper(0);

    bounding_box(bb_lower, bb_upper);

    glm::vec3 factor;
    for (int i = 0; i < 3; ++i) {
        factor[i] =
            (upper[i] - lower[i]) / ((bb_upper[i] - bb_lower[i]) +
                                     std::numeric_limits<float>::epsilon());
    }

    float the_factor = std::min(std::min(factor[0], factor[1]), factor[2]);

    auto coord = *get_input_vertex_coordinates();

    for_each_vertex(HOST, [&](const VertexHandle vh) {
        for (int i = 0; i < 3; ++i) {
            coord(vh, i) += (lower[i] - bb_lower[i]);
            coord(vh, i) *= the_factor;
        }
    });

    coord.move(HOST, DEVICE);

#if USE_POLYSCOPE
    get_polyscope_mesh()->updateVertexPositions(coord);
#endif
}

void RXMeshStatic::bounding_box(glm::vec3& lower, glm::vec3& upper)
{
    lower[0] = std::numeric_limits<float>::max();
    lower[1] = std::numeric_limits<float>::max();
    lower[2] = std::numeric_limits<float>::max();

    upper[0] = std::numeric_limits<float>::lowest();
    upper[1] = std::numeric_limits<float>::lowest();
    upper[2] = std::numeric_limits<float>::lowest();

    auto coord = *get_input_vertex_coordinates();

    for_each_vertex(
        HOST,
        [&](const VertexHandle vh) {
            glm::vec3 v(coord(vh, 0), coord(vh, 1), coord(vh, 2));
            for (int i = 0; i < 3; ++i) {
                lower[i] = std::min(lower[i], v[i]);
                upper[i] = std::max(upper[i], v[i]);
            }
        },
        NULL,
        false);
}

uint32_t RXMeshStatic::map_to_global(const VertexHandle vh) const
{
    auto pl = vh.unpack();
    return m_h_patches_ltog_v[pl.first][pl.second];
}

uint32_t RXMeshStatic::map_to_global(const EdgeHandle eh) const
{
    auto pl = eh.unpack();
    return m_h_patches_ltog_e[pl.first][pl.second];
}

uint32_t RXMeshStatic::map_to_global(const FaceHandle fh) const
{
    auto pl = fh.unpack();
    return m_h_patches_ltog_f[pl.first][pl.second];
}

void RXMeshStatic::create_face_list(std::vector<glm::uvec3>& f_list) const
{
    f_list.reserve(get_num_faces());

    for (uint32_t p = 0; p < this->m_num_patches; ++p) {
        const uint32_t p_num_faces = this->m_h_patches_info[p].num_faces[0];
        for (uint32_t f = 0; f < p_num_faces; ++f) {
            if (!detail::is_deleted(f,
                                    this->m_h_patches_info[p].active_mask_f) &&
                detail::is_owned(f, this->m_h_patches_info[p].owned_mask_f)) {

                glm::uvec3 face;

                for (uint32_t e = 0; e < 3; ++e) {
                    uint16_t edge = this->m_h_patches_info[p].fe[3 * f + e].id;
                    flag_t   dir(0);
                    Context::unpack_edge_dir(edge, edge, dir);
                    uint16_t     e_id = (2 * edge) + dir;
                    uint16_t     v    = this->m_h_patches_info[p].ev[e_id].id;
                    VertexHandle vh(p, v);
                    uint32_t     vid = linear_id(vh);
                    face[e]          = vid;
                }
                f_list.push_back(face);
            }
        }
    }
}

void RXMeshStatic::add_edge_labels(FaceAttribute<int>& face_label,
                                   EdgeAttribute<int>& edge_label)
{
    run_query_kernel<Op::FE, 256>(
        [face_label, edge_label] __device__(const FaceHandle   fh,
                                            const EdgeIterator iter) {
            int label = face_label(fh);

            edge_label(iter[0]) = label;
            edge_label(iter[1]) = label;
            edge_label(iter[2]) = label;
        });
}


// Explicit instantiations

#define RXMESH_STATIC_INSTANTIATE_SCALAR(T)                                \
    template std::shared_ptr<FaceAttribute<T>>                             \
    RXMeshStatic::add_face_attribute<T>(                                   \
        const std::string&, uint32_t, locationT, layoutT);                 \
    template std::shared_ptr<FaceAttribute<T>>                             \
    RXMeshStatic::add_face_attribute<T>(                                   \
        const std::vector<std::vector<T>>&, const std::string&, layoutT);  \
    template std::shared_ptr<FaceAttribute<T>>                             \
    RXMeshStatic::add_face_attribute<T>(                                   \
        const std::vector<T>&, const std::string&, layoutT);               \
    template std::shared_ptr<FaceAttribute<T>>                             \
    RXMeshStatic::add_face_attribute_like<T>(const std::string&,           \
                                             const FaceAttribute<T>&);     \
    template std::shared_ptr<EdgeAttribute<T>>                             \
    RXMeshStatic::add_edge_attribute<T>(                                   \
        const std::string&, uint32_t, locationT, layoutT);                 \
    template std::shared_ptr<EdgeAttribute<T>>                             \
    RXMeshStatic::add_edge_attribute_like<T>(const std::string&,           \
                                             const EdgeAttribute<T>&);     \
    template std::shared_ptr<VertexAttribute<T>>                           \
    RXMeshStatic::add_vertex_attribute<T>(                                 \
        const std::string&, uint32_t, locationT, layoutT);                 \
    template std::shared_ptr<VertexAttribute<T>>                           \
    RXMeshStatic::add_vertex_attribute_like<T>(const std::string&,         \
                                               const VertexAttribute<T>&); \
    template std::shared_ptr<VertexAttribute<T>>                           \
    RXMeshStatic::add_vertex_attribute<T>(                                 \
        const std::vector<std::vector<T>>&, const std::string&, layoutT);  \
    template std::shared_ptr<VertexAttribute<T>>                           \
    RXMeshStatic::add_vertex_attribute<T>(                                 \
        const std::vector<T>&, const std::string&, layoutT);               \
    template void RXMeshStatic::get_boundary_vertices<T>(                  \
        VertexAttribute<T>&, bool, cudaStream_t) const;                    \
    template void RXMeshStatic::export_obj<T>(                             \
        const std::string&, const VertexAttribute<T>&) const;              \
    template void RXMeshStatic::create_vertex_list<T>(                     \
        std::vector<glm::vec3>&, const VertexAttribute<T>&) const;

RXMESH_STATIC_INSTANTIATE_SCALAR(float)
RXMESH_STATIC_INSTANTIATE_SCALAR(double)
RXMESH_STATIC_INSTANTIATE_SCALAR(bool)
RXMESH_STATIC_INSTANTIATE_SCALAR(int)
RXMESH_STATIC_INSTANTIATE_SCALAR(int8_t)
RXMESH_STATIC_INSTANTIATE_SCALAR(uint8_t)
RXMESH_STATIC_INSTANTIATE_SCALAR(uint16_t)
RXMESH_STATIC_INSTANTIATE_SCALAR(uint32_t)
RXMESH_STATIC_INSTANTIATE_SCALAR(uint64_t)

#undef RXMESH_STATIC_INSTANTIATE_SCALAR

// Handle payload types used as attributes (but not for boundary/export helpers)
template std::shared_ptr<VertexAttribute<VertexHandle>>
RXMeshStatic::add_vertex_attribute<VertexHandle>(const std::string&,
                                                 uint32_t,
                                                 locationT,
                                                 layoutT);
template std::shared_ptr<VertexAttribute<VertexHandle>>
RXMeshStatic::add_vertex_attribute<VertexHandle>(
    const std::vector<std::vector<VertexHandle>>&,
    const std::string&,
    layoutT);
template std::shared_ptr<VertexAttribute<VertexHandle>>
RXMeshStatic::add_vertex_attribute<VertexHandle>(
    const std::vector<VertexHandle>&,
    const std::string&,
    layoutT);
template std::shared_ptr<VertexAttribute<VertexHandle>>
RXMeshStatic::add_vertex_attribute_like<VertexHandle>(
    const std::string&,
    const VertexAttribute<VertexHandle>&);

template std::shared_ptr<EdgeAttribute<EdgeHandle>>
RXMeshStatic::add_edge_attribute<EdgeHandle>(const std::string&,
                                             uint32_t,
                                             locationT,
                                             layoutT);
template std::shared_ptr<EdgeAttribute<EdgeHandle>>
RXMeshStatic::add_edge_attribute_like<EdgeHandle>(
    const std::string&,
    const EdgeAttribute<EdgeHandle>&);

template std::shared_ptr<FaceAttribute<FaceHandle>>
RXMeshStatic::add_face_attribute<FaceHandle>(const std::string&,
                                             uint32_t,
                                             locationT,
                                             layoutT);
template std::shared_ptr<FaceAttribute<FaceHandle>>
RXMeshStatic::add_face_attribute<FaceHandle>(
    const std::vector<std::vector<FaceHandle>>&,
    const std::string&,
    layoutT);
template std::shared_ptr<FaceAttribute<FaceHandle>>
RXMeshStatic::add_face_attribute<FaceHandle>(const std::vector<FaceHandle>&,
                                             const std::string&,
                                             layoutT);

// linear_id / get_owner_handle for the three handle types
template uint32_t RXMeshStatic::linear_id<VertexHandle>(VertexHandle) const;
template uint32_t RXMeshStatic::linear_id<EdgeHandle>(EdgeHandle) const;
template uint32_t RXMeshStatic::linear_id<FaceHandle>(FaceHandle) const;

template VertexHandle RXMeshStatic::get_owner_handle<VertexHandle>(
    VertexHandle) const;
template EdgeHandle RXMeshStatic::get_owner_handle<EdgeHandle>(
    EdgeHandle) const;
template FaceHandle RXMeshStatic::get_owner_handle<FaceHandle>(
    FaceHandle) const;

template void RXMeshStatic::prepare_launch_box<128>(
    const std::vector<Op>,
    LaunchBox<128>&,
    const void*,
    const bool,
    const bool,
    const bool,
    std::function<size_t(uint32_t, uint32_t, uint32_t)>) const;
template void RXMeshStatic::prepare_launch_box<256>(
    const std::vector<Op>,
    LaunchBox<256>&,
    const void*,
    const bool,
    const bool,
    const bool,
    std::function<size_t(uint32_t, uint32_t, uint32_t)>) const;
template void RXMeshStatic::prepare_launch_box<384>(
    const std::vector<Op>,
    LaunchBox<384>&,
    const void*,
    const bool,
    const bool,
    const bool,
    std::function<size_t(uint32_t, uint32_t, uint32_t)>) const;
template void RXMeshStatic::prepare_launch_box<512>(
    const std::vector<Op>,
    LaunchBox<512>&,
    const void*,
    const bool,
    const bool,
    const bool,
    std::function<size_t(uint32_t, uint32_t, uint32_t)>) const;
template void RXMeshStatic::prepare_launch_box<768>(
    const std::vector<Op>,
    LaunchBox<768>&,
    const void*,
    const bool,
    const bool,
    const bool,
    std::function<size_t(uint32_t, uint32_t, uint32_t)>) const;
template void RXMeshStatic::prepare_launch_box<1024>(
    const std::vector<Op>,
    LaunchBox<1024>&,
    const void*,
    const bool,
    const bool,
    const bool,
    std::function<size_t(uint32_t, uint32_t, uint32_t)>) const;

template size_t RXMeshStatic::calc_shared_memory<128>(const Op,
                                                      const bool,
                                                      bool) const;
template size_t RXMeshStatic::calc_shared_memory<256>(const Op,
                                                      const bool,
                                                      bool) const;
template size_t RXMeshStatic::calc_shared_memory<512>(const Op,
                                                      const bool,
                                                      bool) const;
template size_t RXMeshStatic::calc_shared_memory<768>(const Op,
                                                      const bool,
                                                      bool) const;
template size_t RXMeshStatic::calc_shared_memory<1024>(const Op,
                                                       const bool,
                                                       bool) const;

// ---- Explicit instantiations for get_region_label ----
template std::shared_ptr<Attribute<int, VertexHandle>>
RXMeshStatic::get_region_label<VertexHandle>();
template std::shared_ptr<Attribute<int, EdgeHandle>>
RXMeshStatic::get_region_label<EdgeHandle>();
template std::shared_ptr<Attribute<int, FaceHandle>>
RXMeshStatic::get_region_label<FaceHandle>();

}  // namespace rxmesh
