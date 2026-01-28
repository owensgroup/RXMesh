// Template implementations for RXMeshStatic that are safe to share.
// Kept in a separate file so:
// - the header can still instantiate for user/app-defined types
// - the library TU (`rxmesh_static.cu`) can explicitly instantiate core types

#pragma once

namespace rxmesh {

template <class T>
std::shared_ptr<FaceAttribute<T>> RXMeshStatic::add_face_attribute(
    const std::string& name,
    uint32_t           num_attributes,
    locationT          location,
    layoutT            layout)
{
    return m_attr_container->template add<FaceAttribute<T>>(
        name.c_str(), num_attributes, location, layout, this);
}

template <class T>
std::shared_ptr<FaceAttribute<T>> RXMeshStatic::add_face_attribute(
    const std::vector<std::vector<T>>& f_attributes,
    const std::string&                 name,
    layoutT                            layout)
{
    if (f_attributes.empty()) {
        RXMESH_ERROR(
            "RXMeshStatic::add_face_attribute() input attribute is empty");
    }

    if (f_attributes.size() != get_num_faces()) {
        RXMESH_ERROR(
            "RXMeshStatic::add_face_attribute() input attribute size ({}) "
            "is not the same as number of faces in the input mesh ({})",
            f_attributes.size(),
            get_num_faces());
    }

    uint32_t num_attributes = static_cast<uint32_t>(f_attributes[0].size());

    auto ret = m_attr_container->template add<FaceAttribute<T>>(
        name.c_str(), num_attributes, LOCATION_ALL, layout, this);

    // populate the attribute before returning it
    const int num_patches = this->get_num_patches();
#pragma omp parallel for
    for (int p = 0; p < num_patches; ++p) {
        for (uint16_t f = 0; f < this->m_h_num_owned_f[p]; ++f) {
            const FaceHandle f_handle(static_cast<uint32_t>(p), f);
            uint32_t global_f = m_h_patches_ltog_f[p][f];
            for (uint32_t a = 0; a < num_attributes; ++a) {
                (*ret)(f_handle, a) = f_attributes[global_f][a];
            }
        }
    }

    // move to device
    ret->move(rxmesh::HOST, rxmesh::DEVICE);
    return ret;
}

template <class T>
std::shared_ptr<FaceAttribute<T>> RXMeshStatic::add_face_attribute(
    const std::vector<T>& f_attributes,
    const std::string&    name,
    layoutT               layout)
{
    if (f_attributes.empty()) {
        RXMESH_ERROR(
            "RXMeshStatic::add_face_attribute() input attribute is empty");
    }

    if (f_attributes.size() != get_num_faces()) {
        RXMESH_ERROR(
            "RXMeshStatic::add_face_attribute() input attribute size ({}) "
            "is not the same as number of faces in the input mesh ({})",
            f_attributes.size(),
            get_num_faces());
    }

    uint32_t num_attributes = 1;

    auto ret = m_attr_container->template add<FaceAttribute<T>>(
        name.c_str(), num_attributes, LOCATION_ALL, layout, this);

    // populate the attribute before returning it
    const int num_patches = this->get_num_patches();
#pragma omp parallel for
    for (int p = 0; p < num_patches; ++p) {
        for (uint16_t f = 0; f < this->m_h_num_owned_f[p]; ++f) {
            const FaceHandle f_handle(static_cast<uint32_t>(p), f);
            uint32_t global_f = m_h_patches_ltog_f[p][f];
            (*ret)(f_handle, 0) = f_attributes[global_f];
        }
    }

    // move to device
    ret->move(rxmesh::HOST, rxmesh::DEVICE);
    return ret;
}

template <class T>
std::shared_ptr<EdgeAttribute<T>> RXMeshStatic::add_edge_attribute(
    const std::string& name,
    uint32_t           num_attributes,
    locationT          location,
    layoutT            layout)
{
    return m_attr_container->template add<EdgeAttribute<T>>(
        name.c_str(), num_attributes, location, layout, this);
}

template <class T>
std::shared_ptr<EdgeAttribute<T>> RXMeshStatic::add_edge_attribute_like(
    const std::string&      name,
    const EdgeAttribute<T>& other)
{
    return add_edge_attribute<T>(name,
                                 other.get_num_attributes(),
                                 other.get_allocated(),
                                 other.get_layout());
}

template <class T>
std::shared_ptr<VertexAttribute<T>> RXMeshStatic::add_vertex_attribute(
    const std::string& name,
    uint32_t           num_attributes,
    locationT          location,
    layoutT            layout)
{
    return m_attr_container->template add<VertexAttribute<T>>(
        name.c_str(), num_attributes, location, layout, this);
}

template <class T>
std::shared_ptr<VertexAttribute<T>> RXMeshStatic::add_vertex_attribute_like(
    const std::string&        name,
    const VertexAttribute<T>& other)
{
    return add_vertex_attribute<T>(name,
                                   other.get_num_attributes(),
                                   other.get_allocated(),
                                   other.get_layout());
}

template <class T>
std::shared_ptr<VertexAttribute<T>> RXMeshStatic::add_vertex_attribute(
    const std::vector<std::vector<T>>& v_attributes,
    const std::string&                 name,
    layoutT                            layout)
{
    if (v_attributes.empty()) {
        RXMESH_ERROR("RXMeshStatic::add_vertex_attribute() input attribute is "
                     "empty");
    }

    if (v_attributes.size() != get_num_vertices()) {
        RXMESH_ERROR(
            "RXMeshStatic::add_vertex_attribute() input attribute size "
            "({}) is not the same as number of vertices in the input mesh "
            "({})",
            v_attributes.size(),
            get_num_vertices());
    }

    uint32_t num_attributes = static_cast<uint32_t>(v_attributes[0].size());

    auto ret = m_attr_container->template add<VertexAttribute<T>>(
        name.c_str(), num_attributes, LOCATION_ALL, layout, this);

    // populate the attribute before returning it
    const int num_patches = this->get_num_patches();
#pragma omp parallel for
    for (int p = 0; p < num_patches; ++p) {
        for (uint16_t v = 0; v < this->m_h_num_owned_v[p]; ++v) {
            const VertexHandle v_handle(static_cast<uint32_t>(p), v);
            uint32_t global_v = m_h_patches_ltog_v[p][v];
            for (uint32_t a = 0; a < num_attributes; ++a) {
                (*ret)(v_handle, a) = v_attributes[global_v][a];
            }
        }
    }

    // move to device
    ret->move(rxmesh::HOST, rxmesh::DEVICE);
    return ret;
}

template <class T>
std::shared_ptr<VertexAttribute<T>> RXMeshStatic::add_vertex_attribute(
    const std::vector<T>& v_attributes,
    const std::string&    name,
    layoutT               layout)
{
    if (v_attributes.empty()) {
        RXMESH_ERROR("RXMeshStatic::add_vertex_attribute() input attribute is "
                     "empty");
    }

    if (v_attributes.size() != get_num_vertices()) {
        RXMESH_ERROR(
            "RXMeshStatic::add_vertex_attribute() input attribute size "
            "({}) is not the same as number of vertices in the input mesh "
            "({})",
            v_attributes.size(),
            get_num_vertices());
    }

    uint32_t num_attributes = 1;

    auto ret = m_attr_container->template add<VertexAttribute<T>>(
        name.c_str(), num_attributes, LOCATION_ALL, layout, this);

    // populate the attribute before returning it
    const int num_patches = this->get_num_patches();
#pragma omp parallel for
    for (int p = 0; p < num_patches; ++p) {
        for (uint16_t v = 0; v < this->m_h_num_owned_v[p]; ++v) {
            const VertexHandle v_handle(static_cast<uint32_t>(p), v);
            uint32_t global_v = m_h_patches_ltog_v[p][v];
            (*ret)(v_handle, 0) = v_attributes[global_v];
        }
    }

    // move to device
    ret->move(rxmesh::HOST, rxmesh::DEVICE);
    return ret;
}

template <class T, class HandleT>
std::shared_ptr<Attribute<T, HandleT>> RXMeshStatic::add_attribute(
    const std::string& name,
    uint32_t           num_attributes,
    locationT          location,
    layoutT            layout)
{
    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        return add_vertex_attribute<T>(name, num_attributes, location, layout);
    }

    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        return add_edge_attribute<T>(name, num_attributes, location, layout);
    }

    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        return add_face_attribute<T>(name, num_attributes, location, layout);
    }
}

template <class T, class HandleT>
std::shared_ptr<Attribute<T, HandleT>> RXMeshStatic::add_attribute_like(
    const std::string&           name,
    const Attribute<T, HandleT>& other)
{
    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        return add_vertex_attribute<T>(name,
                                       other.get_num_attributes(),
                                       other.get_allocated(),
                                       other.get_layout());
    }

    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        return add_edge_attribute<T>(name,
                                     other.get_num_attributes(),
                                     other.get_allocated(),
                                     other.get_layout());
    }

    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        return add_face_attribute<T>(name,
                                     other.get_num_attributes(),
                                     other.get_allocated(),
                                     other.get_layout());
    }
}

template <typename T>
void RXMeshStatic::get_boundary_vertices(VertexAttribute<T>& boundary_v,
                                        bool                move_to_host,
                                        cudaStream_t        stream) const
{
    if (!boundary_v.is_device_allocated()) {
        RXMESH_ERROR(
            "RXMeshStatic::get_boundary_vertices the input/output "
            "VertexAttribute (i.e., boundary_v) should be allocated on "
            "device since the boundary vertices are identified first on "
            "the device (before optionally moving them to the host). "
            "Returning without calculating the boundary vertices!");
        return;
    }

    boundary_v.reset(0, LOCATION_ALL);

    constexpr uint32_t blockThreads = 256;

    int max_shmem_bytes = 89 * 1024;
    CUDA_ERROR(cudaFuncSetAttribute(
        (void*)detail::identify_boundary_vertices<blockThreads, T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_shmem_bytes));

    LaunchBox<blockThreads> lb;

    prepare_launch_box(
        {Op::EF, Op::EV},
        lb,
        (void*)detail::identify_boundary_vertices<blockThreads, T>,
        false,
        false,
        false,
        [&](uint32_t v, uint32_t e, uint32_t f) {
            return detail::mask_num_bytes(e) + ShmemAllocator::default_alignment;
        });

    detail::identify_boundary_vertices<blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn, stream>>>(
            get_context(), boundary_v);

    if (move_to_host && boundary_v.is_host_allocated()) {
        boundary_v.move(DEVICE, HOST, stream);
    }
}

template <typename HandleT>
uint32_t RXMeshStatic::linear_id(HandleT input) const
{
    if (!input.is_valid()) {
        RXMESH_ERROR("RXMeshStatic::linear_id() input handle is not valid");
    }

    if (input.patch_id() >= get_num_patches()) {
        RXMESH_ERROR(
            "RXMeshStatic::linear_id() patch index ({}) is out-of-bound",
            input.patch_id());
    }

    const HandleT owner_handle = get_owner_handle(input);

    uint32_t p_id = owner_handle.patch_id();
    uint16_t ret  = owner_handle.local_id();

    ret = this->m_h_patches_info[p_id].count_num_owned(
        m_h_patches_info[p_id].get_owned_mask<HandleT>(),
        m_h_patches_info[p_id].get_active_mask<HandleT>(),
        ret);

    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        return ret + m_h_vertex_prefix[p_id];
    }
    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        return ret + m_h_edge_prefix[p_id];
    }
    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        return ret + m_h_face_prefix[p_id];
    }
}

template <typename HandleT>
HandleT RXMeshStatic::get_owner_handle(const HandleT input) const
{
    return get_context().get_owner_handle(input, m_h_patches_info);
}

template <typename T>
void RXMeshStatic::export_obj(const std::string&        filename,
                             const VertexAttribute<T>& coords) const
{
    std::string  fn = filename;
    std::fstream file(fn, std::ios::out);
    file.precision(30);

    std::vector<glm::vec3> v_list;
    create_vertex_list(v_list, coords);

    assert(get_num_vertices() == v_list.size());

    for (uint32_t v = 0; v < v_list.size(); ++v) {
        file << "v " << v_list[v][0] << " " << v_list[v][1] << " "
             << v_list[v][2] << " \n";
    }

    std::vector<glm::uvec3> f_list;
    create_face_list(f_list);

    assert(f_list.size() == get_num_faces());

    for (uint32_t f = 0; f < f_list.size(); ++f) {
        file << "f ";
        for (uint32_t i = 0; i < 3; ++i) {
            file << f_list[f][i] + 1 << " ";
        }
        file << "\n";
    }

    file.close();
}

template <typename T>
void RXMeshStatic::create_vertex_list(std::vector<glm::vec3>&   v_list,
                                     const VertexAttribute<T>& coords) const
{
    v_list.resize(get_num_vertices());
    for_each_vertex(
        HOST,
        [&](const VertexHandle vh) {
            uint32_t vid   = linear_id(vh);
            v_list[vid][0] = coords(vh, 0);
            v_list[vid][1] = coords(vh, 1);
            v_list[vid][2] = coords(vh, 2);
        },
        NULL,
        false);
}

}  // namespace rxmesh

