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
            uint32_t         global_f = m_h_patches_ltog_f[p][f];
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
            uint32_t         global_f = m_h_patches_ltog_f[p][f];
            (*ret)(f_handle, 0)       = f_attributes[global_f];
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
        RXMESH_ERROR(
            "RXMeshStatic::add_vertex_attribute() input attribute is "
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
            uint32_t           global_v = m_h_patches_ltog_v[p][v];
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
        RXMESH_ERROR(
            "RXMeshStatic::add_vertex_attribute() input attribute is "
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
            uint32_t           global_v = m_h_patches_ltog_v[p][v];
            (*ret)(v_handle, 0)         = v_attributes[global_v];
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
            return detail::mask_num_bytes(e) +
                   ShmemAllocator::default_alignment;
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

template <class T>
std::shared_ptr<FaceAttribute<T>> RXMeshStatic::add_face_attribute_like(
    const std::string&      name,
    const FaceAttribute<T>& other)
{
    return add_face_attribute<T>(name,
                                 other.get_num_attributes(),
                                 other.get_allocated(),
                                 other.get_layout());
}

template <typename HandleT>
std::shared_ptr<Attribute<int, HandleT>> RXMeshStatic::get_region_label()
{
    if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        return get_face_region_label();
    }

    if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        return get_edge_region_label();
    }

    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        return get_vertex_region_label();
    }
}

template <uint32_t blockThreads>
void RXMeshStatic::prepare_launch_box(
    const std::vector<Op>                               op,
    LaunchBox<blockThreads>&                            launch_box,
    const void*                                         kernel,
    const bool                                          oriented,
    const bool                                          with_vertex_valence,
    const bool                                          is_concurrent,
    std::function<size_t(uint32_t, uint32_t, uint32_t)> user_shmem) const
{

    launch_box.blocks         = this->m_num_patches;
    launch_box.smem_bytes_dyn = 0;

    for (auto o : op) {
        size_t sh =
            this->template calc_shared_memory<blockThreads>(o, oriented, false);
        if (is_concurrent) {
            launch_box.smem_bytes_dyn += sh;
        } else {
            launch_box.smem_bytes_dyn = std::max(launch_box.smem_bytes_dyn, sh);
        }
    }

    launch_box.smem_bytes_dyn += user_shmem(
        m_max_vertices_per_patch, m_max_edges_per_patch, m_max_faces_per_patch);

    if (with_vertex_valence) {
        if (get_input_max_valence() > 256) {
            RXMESH_ERROR(
                "RXMeshStatic::prepare_launch_box() input max valence if "
                "greater than 256 and thus using uint8_t to store the "
                "vertex valence will lead to overflow");
        }
        launch_box.smem_bytes_dyn +=
            this->m_max_vertices_per_patch * sizeof(uint8_t) +
            ShmemAllocator::default_alignment;
    }


    RXMESH_TRACE(
        "RXMeshStatic::calc_shared_memory() launching {} blocks with "
        "{} threads on the device",
        launch_box.blocks,
        blockThreads);


    check_shared_memory(launch_box.smem_bytes_dyn,
                        launch_box.smem_bytes_static,
                        launch_box.num_registers_per_thread,
                        launch_box.local_mem_per_thread,
                        blockThreads,
                        kernel);
}

template <uint32_t blockThreads>
size_t RXMeshStatic::calc_shared_memory(const Op   op,
                                        const bool oriented,
                                        bool       use_capacity) const
{
    uint32_t max_v(this->m_max_vertices_per_patch),
        max_e(this->m_max_edges_per_patch), max_f(this->m_max_faces_per_patch);

    if (use_capacity) {
        max_v = get_per_patch_max_vertex_capacity();
        max_e = get_per_patch_max_edge_capacity();
        max_f = get_per_patch_max_face_capacity();
    }


    // if (oriented && !(op == Op::VV || op == Op::VE)) {
    //     RXMESH_ERROR(
    //         "RXMeshStatic::calc_shared_memory() Oriented is only "
    //         "allowed on VV and VE. The input op is {}",
    //         op_to_string(op));
    // }

    if ((op == Op::EVDiamond || op == Op::EE) && !m_is_input_edge_manifold) {
        RXMESH_ERROR(
            "RXMeshStatic::calc_shared_memory() Op::EVDiamond and Op::EE "
            "only works on edge manifold mesh. The input mesh is not edge "
            "manifold");
    }

    size_t dynamic_smem = 0;


    if (op == Op::FE) {
        // only FE will be loaded
        dynamic_smem = 3 * max_f * sizeof(uint16_t);

        // store participant bitmask
        dynamic_smem += max_bitmask_size<LocalFaceT>();

        // store not-owned bitmask
        dynamic_smem += max_bitmask_size<LocalEdgeT>();

        // stores edges LP hashtable
        dynamic_smem +=
            sizeof(LPPair) * max_lp_hashtable_capacity<LocalEdgeT>();

        // for possible padding for alignment
        // 4 since there are 4 calls for ShmemAllocator.alloc
        dynamic_smem += ShmemAllocator::default_alignment * 4;

    } else if (op == Op::EV) {
        // only EV will be loaded
        dynamic_smem = 2 * max_e * sizeof(uint16_t);

        // store participant bitmask
        dynamic_smem += max_bitmask_size<LocalEdgeT>();

        // store not-owned bitmask
        dynamic_smem += max_bitmask_size<LocalVertexT>();

        // stores vertex LP hashtable
        dynamic_smem +=
            sizeof(LPPair) * max_lp_hashtable_capacity<LocalVertexT>();

        // for possible padding for alignment
        // 4 since there are 4 calls for ShmemAllocator.alloc
        dynamic_smem += ShmemAllocator::default_alignment * 4;

    } else if (op == Op::FV) {
        // We load both FE and EV. We don't change EV.
        // FE are updated to contain FV instead of FE by reading from
        // EV. After that, we can throw EV away so we can load hashtable
        dynamic_smem += 3 * max_f * sizeof(uint16_t);
        dynamic_smem += 2 * max_e * sizeof(uint16_t);

        // store participant bitmask
        dynamic_smem += max_bitmask_size<LocalFaceT>();

        // store not-owned bitmask
        dynamic_smem += max_bitmask_size<LocalVertexT>();

        //  stores vertex LP hashtable
        uint32_t table_bytes =
            sizeof(LPPair) * max_lp_hashtable_capacity<LocalVertexT>();
        if (table_bytes > 2 * max_e * sizeof(uint16_t)) {

            dynamic_smem += table_bytes - 2 * max_e * sizeof(uint16_t);
        }
        // for possible padding for alignment
        // 5 since there are 5 calls for ShmemAllocator.alloc
        dynamic_smem += ShmemAllocator::default_alignment * 5;
        // TODO no need for extra memory to load not-owned vertices local
        // and patch id. We load them and overwrite EV.
    } else if (op == Op::VE) {
        // load EV and then transpose it in place
        // The transpose needs two buffer; one for prefix sum and another
        // for the actual output
        // The prefix sum will be stored in place (where EV is loaded)
        // The output will be stored in another buffer with size equal to
        // the EV (i.e., 2*#edges) since this output buffer will store the
        // nnz and the nnz of a matrix the same before/after transpose
        // Normally, the number of vertices is way less than 2*#E but in
        // dynamic mesh, we can not predicate these numbers since some
        // of these edges could be deleted (marked deleted in the bitmask)
        // so, we allocate the buffer that hold EV (which will also hold the
        // offset) to be the max of #V and 2#E
        dynamic_smem = std::max(max_v + 1, 2 * max_e) * sizeof(uint16_t);
        dynamic_smem += (2 * max_e) * sizeof(uint16_t);

        // store participant bitmask
        dynamic_smem += max_bitmask_size<LocalVertexT>();

        // store not-owned bitmask
        dynamic_smem += max_bitmask_size<LocalEdgeT>();

        // stores edge LP hashtable
        uint32_t lp_smem =
            sizeof(LPPair) * max_lp_hashtable_capacity<LocalEdgeT>();

        // temp memory needed for block_mat_transpose to store the prefix
        // and local incremental
        uint32_t temp_size_local = (2 * max_v + 1) * sizeof(uint16_t);

        // For oriented VE, we additionally need to store FE and EF
        // along with the (transposed) VE. FE needs 3*max_num_faces. Since
        // oriented is only done on manifold, EF needs only 2*max_num_edges
        // since every edge is neighbor to maximum of two faces (which we
        // write on the same place as the extra EV)

        uint32_t fe_ef_smem =
            (2 * max_e) * sizeof(uint16_t) + (3 * max_f) * sizeof(uint16_t);

        if (oriented) {
            dynamic_smem +=
                std::max(std::max(lp_smem, fe_ef_smem), temp_size_local);
        } else {
            dynamic_smem += std::max(temp_size_local, lp_smem);
        }

        // for possible padding for alignment
        // 4 since there are 4 calls for ShmemAllocator.alloc
        dynamic_smem += ShmemAllocator::default_alignment * 8;

    } else if (op == Op::EF) {
        // same as Op::VE but with faces
        dynamic_smem = std::max(max_e + 1, 3 * max_f) * sizeof(uint16_t);
        dynamic_smem += (3 * max_f) * sizeof(uint16_t);

        // store participant bitmask
        dynamic_smem += max_bitmask_size<LocalEdgeT>();

        // store not-owned bitmask
        dynamic_smem += max_bitmask_size<LocalFaceT>();

        // temp memory needed for block_mat_transpose to store the prefix
        // and local incremental
        uint32_t temp_size_local = (2 * max_e + 1) * sizeof(uint16_t);

        // stores the face LP hashtable
        uint32_t lp_smem =
            sizeof(LPPair) * max_lp_hashtable_capacity<LocalFaceT>();

        dynamic_smem += std::max(temp_size_local, lp_smem);

        // for possible padding for alignment
        // 4 since there are 4 calls for ShmemAllocator.alloc
        dynamic_smem += ShmemAllocator::default_alignment * 6;

    } else if (op == Op::VF) {
        // load EV and FE simultaneously. Changes FE to FV using EV. Then
        // transpose FV in place and use EV to store the values/output while
        // using FV to store the prefix sum. Thus, the space used to store
        // EV should be max(3*#faces, 2*#edges)
        dynamic_smem = std::max(3 * max_f, 1 + max_v) * sizeof(uint16_t);
        dynamic_smem += std::max(3 * max_f, 2 * max_e) * sizeof(uint16_t) +
                        sizeof(uint16_t);

        // store participant bitmask
        dynamic_smem += max_bitmask_size<LocalVertexT>();

        // store not-owned bitmask
        dynamic_smem += max_bitmask_size<LocalFaceT>();

        // temp memory needed for block_mat_transpose to store the prefix
        // and local incremental
        uint32_t temp_size_local = (2 * max_v + 1) * sizeof(uint16_t);

        // stores the face LP hashtable
        uint32_t lp_shmem =
            sizeof(LPPair) * max_lp_hashtable_capacity<LocalFaceT>();

        // we load the hashtable after we transpose, thus we take the max
        // of the (temp) memory needed for transpose and LP hashtable
        dynamic_smem += std::max(temp_size_local, lp_shmem);

        // for possible padding for alignment
        // 5 since there are 5 calls for ShmemAllocator.alloc
        dynamic_smem += ShmemAllocator::default_alignment * 5;

    } else if (op == Op::VV) {
        // similar to VE but we also need to store the EV even after
        // we do the transpose. After that, we can throw EV away and load
        // the hash table
        dynamic_smem = std::max(max_v + 1, 2 * max_e) * sizeof(uint16_t);
        dynamic_smem += (2 * max_e) * sizeof(uint16_t);

        // store participant bitmask
        dynamic_smem += max_bitmask_size<LocalVertexT>();

        // store not-owned bitmask
        dynamic_smem += max_bitmask_size<LocalVertexT>();

        // duplicate EV
        uint32_t ev_smem = (2 * max_e) * sizeof(uint16_t);

        // temp memory needed for block_mat_transpose to store the prefix
        // and local incremental
        uint32_t temp_size_local = (2 * max_v + 1) * sizeof(uint16_t);

        // stores the vertex LP hashtable
        uint32_t lp_smem =
            sizeof(LPPair) * max_lp_hashtable_capacity<LocalVertexT>();

        if (oriented) {
            // For oriented VV, we additionally need to store FE and EF
            // along with the (transposed) VE. FE needs 3*max_num_faces.
            // Since oriented is only done on manifold, EF needs only
            // 2*max_num_edges since every edge is neighbor to maximum of
            // two faces (which we write on the same place as the extra EV).
            // With VV, we need to reload EV again (since it is overwritten)
            uint32_t fe_smem = (3 * max_f) * sizeof(uint16_t);

            dynamic_smem +=
                std::max(std::max(lp_smem, ev_smem + temp_size_local),
                         fe_smem + ev_smem);

        } else {
            dynamic_smem += std::max(lp_smem, ev_smem + temp_size_local);
        }


        // for possible padding for alignment
        // 5 since there are 5 calls for ShmemAllocator.alloc
        dynamic_smem += ShmemAllocator::default_alignment * 8;

    } else if (op == Op::FF) {
        // FF needs to store FE and EF along side with the output itself
        // FE needs 3*max_num_faces
        // EF is FE transpose
        // FF is max_num_faces + (on average) 3*max_num_faces
        // Since we have so many boundary faces (due to ribbons), they will
        // make up this averaging

        uint32_t fe_smem = 3 * max_f * sizeof(uint16_t);
        uint32_t ef_smem =
            (std::max(max_e + 1, 3 * max_f) + (3 * max_f)) * sizeof(uint16_t);


        // temp memory needed for block_mat_transpose to store the prefix
        // and local incremental
        uint32_t temp_size_local = (2 * max_e + 1) * sizeof(uint16_t);

        // the output FF
        dynamic_smem += 4 * max_f * sizeof(uint16_t);

        // store participant bitmask
        dynamic_smem += max_bitmask_size<LocalFaceT>();

        // store not-owned bitmask
        dynamic_smem += max_bitmask_size<LocalFaceT>();

        // stores vertex LP hashtable
        uint32_t lp_smem =
            sizeof(LPPair) * max_lp_hashtable_capacity<LocalFaceT>();

        dynamic_smem +=
            std::max(lp_smem, ef_smem + std::max(fe_smem, temp_size_local));

        // for possible padding for alignment
        // 6 since there are 6 calls for ShmemAllocator.alloc
        dynamic_smem += ShmemAllocator::default_alignment * 9;


    } else if (op == Op::EVDiamond) {
        // to load EV and also store the results which contains 4 vertices
        // for each edge
        dynamic_smem = 4 * max_e * sizeof(uint16_t);

        // to store FE
        uint32_t fe_smem = 3 * max_f * sizeof(uint16_t);

        // store participant bitmask
        dynamic_smem += max_bitmask_size<LocalEdgeT>();

        // store not-owned bitmask
        dynamic_smem += max_bitmask_size<LocalVertexT>();

        // stores vertex LP hashtable
        uint32_t lp_smem =
            sizeof(LPPair) * max_lp_hashtable_capacity<LocalVertexT>();

        dynamic_smem += std::max(fe_smem, lp_smem);

        // for possible padding for alignment
        // 4 since there are 4 calls for ShmemAllocator.alloc
        dynamic_smem += ShmemAllocator::default_alignment * 5;
    } else if (op == Op::EE) {
        // to store the results i.e., 4 edges for each edge
        dynamic_smem = 4 * max_e * sizeof(uint16_t);

        // store participant bitmask
        dynamic_smem += max_bitmask_size<LocalEdgeT>();

        // store not-owned bitmask
        dynamic_smem += max_bitmask_size<LocalEdgeT>();

        // stores vertex LP hashtable
        uint32_t lp_smem =
            sizeof(LPPair) * max_lp_hashtable_capacity<LocalEdgeT>();


        // for possible padding for alignment
        // 4 since there are 4 calls for ShmemAllocator.alloc
        dynamic_smem += ShmemAllocator::default_alignment * 4;
    }

    if (oriented) {
        if (op == Op::VE) {
            // For VE, we need to add the extra memory we needed for VV that
            // load EV beside the VE
            dynamic_smem += (2 * max_e) * sizeof(uint16_t);
        }
        // For oriented VV or VE, we additionally need to store FE and EF
        // along with the (transposed) VE. FE needs 3*max_num_faces. Since
        // oriented is only done on manifold, EF needs only 2*max_num_edges
        // since every edge is neighbor to maximum of two faces (which we
        // write on the same place as the extra EV). With VV, we need
        // to reload EV again (since it is overwritten) but we don't need to
        // do this for VE
        dynamic_smem += (3 * max_f) * sizeof(uint16_t);
    }

    return dynamic_smem;
}

}  // namespace rxmesh
