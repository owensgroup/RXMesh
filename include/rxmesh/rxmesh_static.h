#pragma once
#include <assert.h>
#include <fstream>
#include <memory>

#include <cuda_profiler_api.h>

#include "rxmesh/attribute.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/for_each.cuh"
#include "rxmesh/kernels/shmem_allocator.cuh"
#include "rxmesh/launch_box.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/types.h"
#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/timer.h"
#if USE_POLYSCOPE
#include "polyscope/surface_mesh.h"
#endif

namespace rxmesh {

/**
 * @brief This class is responsible for query operations of static meshes. It
 * extends RXMesh with methods needed to launch kernel and do computation on the
 * mesh as well as managing mesh attributes
 */
class RXMeshStatic : public RXMesh
{
   public:
    RXMeshStatic(const RXMeshStatic&) = delete;

    /**
     * @brief Constructor using path to obj file
     * @param file_path path to an obj file
     * @param quite run in quite mode
     */
    RXMeshStatic(const std::string file_path,
                 const bool        quite        = false,
                 const std::string patcher_file = "")
        : RXMesh()
    {
        std::vector<std::vector<uint32_t>> fv;
        std::vector<std::vector<float>>    vertices;
        if (!import_obj(file_path, vertices, fv)) {
            RXMESH_ERROR(
                "RXMeshStatic::RXMeshStatic could not read the input file {}",
                file_path);
            exit(EXIT_FAILURE);
        }

        this->init(fv, patcher_file, quite);

        m_attr_container = std::make_shared<AttributeContainer>();

        m_input_vertex_coordinates =
            this->add_vertex_attribute<float>(vertices, "rx:vertices");

#if USE_POLYSCOPE
        polyscope::init();
        m_polyscope_mesh_name = polyscope::guessNiceNameFromPath(file_path);
        this->register_polyscope();
#endif
    };

    /**
     * @brief Constructor using triangles and vertices
     * @param fv Face incident vertices as read from an obj file
     * @param quite run in quite mode
     */
    RXMeshStatic(std::vector<std::vector<uint32_t>>& fv,
                 const bool                          quite        = false,
                 const std::string                   patcher_file = "")
        : RXMesh(), m_input_vertex_coordinates(nullptr)
    {
        this->init(fv, patcher_file, quite);
        m_attr_container = std::make_shared<AttributeContainer>();
    };

    virtual ~RXMeshStatic()
    {
    }

#if USE_POLYSCOPE
    /**
     * @brief return a pointer to polyscope surface which has been registered
     * with this instance
     */
    polyscope::SurfaceMesh* get_polyscope_mesh()
    {
        return m_polyscope_mesh;
    }


    /**
     * @brief add a patch as a separate SurfaceMesh to polyscope renderer. The
     * patch is added along with its ribbon which could be helpful for debugging
     * @param p the patch id which will be added
     */
    polyscope::SurfaceMesh* render_patch(const uint32_t p)
    {
        std::vector<std::array<uint32_t, 3>> fv;
        fv.reserve(m_h_patches_info[p].num_faces[0]);
        add_patch_to_polyscope(p, fv, true);

        return polyscope::registerSurfaceMesh(
            m_polyscope_mesh_name + "_patch_" + std::to_string(p),
            *m_input_vertex_coordinates,
            fv,
            m_polyscope_edges_map);
    }

    /**
     * @brief add the face's patch scalar quantity to a polyscope instance
     * (polyscope_mesh) for specific patch. polyscope_mesh should be the one
     * returned from render_patch call with the same input patch (p)
     * @param p patch id for which the face patch will be added
     * @param polyscope_mesh the SurfaceMesh pointer returned by calling
     * render_patch with the same input patch
     */
    polyscope::SurfaceFaceScalarQuantity* polyscope_render_face_patch(
        const uint32_t          p,
        polyscope::SurfaceMesh* polyscope_mesh)
    {
        std::string      name = "rx:FPatch" + std::to_string(p);
        std::vector<int> patch_id(m_h_patches_info[p].num_faces[0], -1);

        for (uint16_t f = 0; f < this->m_h_patches_info[p].num_faces[0]; ++f) {
            const LocalFaceT lf(f);
            if (!this->m_h_patches_info[p].is_deleted(lf)) {

                const FaceHandle fh = Context::get_owner_handle<FaceHandle>(
                    {p, lf}, nullptr, m_h_patches_info);

                patch_id[f] = fh.unpack().first;
            }
        }

        patch_id.erase(std::remove(patch_id.begin(), patch_id.end(), -1),
                       patch_id.end());
        return polyscope_mesh->addFaceScalarQuantity(name, patch_id);
    }

    /**
     * @brief add the edge's patch scalar quantity to a polyscope instance
     * (polyscope_mesh) for specific patch. polyscope_mesh should be the one
     * returned from render_patch call with the same input patch (p)
     * @param p patch id for which the face patch will be added
     * @param polyscope_mesh the SurfaceMesh pointer returned by calling
     * render_patch with the same input patch
     */
    polyscope::SurfaceEdgeScalarQuantity* polyscope_render_edge_patch(
        const uint32_t          p,
        polyscope::SurfaceMesh* polyscope_mesh)
    {
        std::string name = "rx:EPatch" + std::to_string(p);
        // unlike polyscope_render_face_patch and  where the size of this
        // std::vector is the size of number faces in patch, here we
        // use the total number of edges since we pass to polyscope the edge map
        // for the whole mesh (not just for this patch) (see render_patch) and
        // thus it expects the size of this quantity to be the same size i.e.,
        // total number of edges
        std::vector<int> patch_id(get_num_edges(), -(p + 1));

        for_each_edge(HOST, [&](EdgeHandle eh) {
            patch_id[linear_id(eh)] = eh.unpack().first;
        });

        return polyscope_mesh->addEdgeScalarQuantity(name, patch_id);
    }

    /**
     * @brief add the vertex's patch scalar quantity to a polyscope instance
     * (polyscope_mesh) for specific patch. polyscope_mesh should be the one
     * returned from render_patch call with the same input patch (p)
     * @param p patch id for which the face patch will be added
     * @param polyscope_mesh the SurfaceMesh pointer returned by calling
     * render_patch with the same input patch
     */
    polyscope::SurfaceVertexScalarQuantity* polyscope_render_vertex_patch(
        const uint32_t          p,
        polyscope::SurfaceMesh* polyscope_mesh)
    {
        std::string name = "rx:VPatch" + std::to_string(p);
        // unlike polyscope_render_face_patch and  where the size of this
        // std::vector is the size of number faces in patch, here we
        // use the total number of vertices since we pass to polyscope the
        // vertex position for the whole mesh (not just for this patch) (see
        // render_patch) and thus it expects the size of this quantity to be the
        // same size i.e., total number of vertices
        std::vector<int> patch_id(get_num_vertices(), -(p + 1));

        for_each_vertex(HOST, [&](VertexHandle vh) {
            patch_id[linear_id(vh)] = vh.unpack().first;
        });

        return polyscope_mesh->addVertexScalarQuantity(name, patch_id);
    }


    /**
     * @brief add the face's patch scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's face scalar quantity
     */
    polyscope::SurfaceFaceScalarQuantity* polyscope_render_face_patch()
    {
        std::string name = "rx:FPatch";
        auto face_patch  = this->add_face_attribute<uint32_t>(name, 1, HOST);
        for_each_face(HOST, [&](FaceHandle fh) {
            (*face_patch)(fh) = fh.unpack().first;
        });
        auto ret = m_polyscope_mesh->addFaceScalarQuantity(name, *face_patch);
        remove_attribute(name);
        return ret;
    }

    /**
     * @brief add the edge's patch scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's edge scalar quantity
     */
    polyscope::SurfaceEdgeScalarQuantity* polyscope_render_edge_patch()
    {
        std::string name = "rx:EPatch";
        auto edge_patch  = this->add_edge_attribute<uint32_t>(name, 1, HOST);
        for_each_edge(HOST, [&](EdgeHandle eh) {
            (*edge_patch)(eh) = eh.unpack().first;
        });
        auto ret = m_polyscope_mesh->addEdgeScalarQuantity(name, *edge_patch);
        remove_attribute(name);
        return ret;
    }


    /**
     * @brief add the vertex's patch scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's vertex scalar quantity
     */
    polyscope::SurfaceVertexScalarQuantity* polyscope_render_vertex_patch()
    {
        std::string name  = "rx:VPatch";
        auto vertex_patch = this->add_vertex_attribute<uint32_t>(name, 1, HOST);
        for_each_vertex(HOST, [&](VertexHandle vh) {
            (*vertex_patch)(vh) = vh.unpack().first;
        });
        auto ret =
            m_polyscope_mesh->addVertexScalarQuantity(name, *vertex_patch);
        remove_attribute(name);
        return ret;
    }

#endif

    /**
     * @brief Apply a lambda function on all vertices in the mesh
     * @tparam LambdaT type of the lambda function (inferred)
     * @param location the execution location
     * @param apply lambda function to be applied on all vertices. The lambda
     * function signature takes a VertexHandle
     * @param stream the stream used to run the kernel in case of DEVICE
     * execution location
     */
    template <typename LambdaT>
    void for_each_vertex(locationT    location,
                         LambdaT      apply,
                         cudaStream_t stream = NULL)
    {
        if ((location & HOST) == HOST) {
            const int num_patches = this->get_num_patches();
#pragma omp parallel for
            for (int p = 0; p < num_patches; ++p) {
                for (uint16_t v = 0;
                     v < this->m_h_patches_info[p].num_vertices[0];
                     ++v) {

                    if (detail::is_owned(v, m_h_patches_info[p].owned_mask_v) &&
                        !detail::is_deleted(
                            v, m_h_patches_info[p].active_mask_v)) {

                        const VertexHandle v_handle(static_cast<uint32_t>(p),
                                                    v);
                        apply(v_handle);
                    }
                }
            }
        }

        if ((location & DEVICE) == DEVICE) {
            if constexpr (IS_HD_LAMBDA(LambdaT) || IS_D_LAMBDA(LambdaT)) {

                const int num_patches = this->get_num_patches();
                const int threads     = 256;
                detail::for_each_vertex<<<num_patches, threads, 0, stream>>>(
                    num_patches, this->m_d_patches_info, apply);
            } else {
                RXMESH_ERROR(
                    "RXMeshStatic::for_each_vertex() Input lambda function "
                    "should be annotated with  __device__ for execution on "
                    "device");
            }
        }
    }

    /**
     * @brief Apply a lambda function on all edges in the mesh
     * @tparam LambdaT type of the lambda function (inferred)
     * @param location the execution location
     * @param apply lambda function to be applied on all edges. The lambda
     * function signature takes a EdgeHandle
     * @param stream the stream used to run the kernel in case of DEVICE
     * execution location
     */
    template <typename LambdaT>
    void for_each_edge(locationT    location,
                       LambdaT      apply,
                       cudaStream_t stream = NULL)
    {
        if ((location & HOST) == HOST) {
            const int num_patches = this->get_num_patches();
#pragma omp parallel for
            for (int p = 0; p < num_patches; ++p) {
                for (uint16_t e = 0; e < this->m_h_patches_info[p].num_edges[0];
                     ++e) {

                    if (detail::is_owned(e, m_h_patches_info[p].owned_mask_e) &&
                        !detail::is_deleted(
                            e, m_h_patches_info[p].active_mask_e)) {

                        const EdgeHandle e_handle(static_cast<uint32_t>(p), e);
                        apply(e_handle);
                    }
                }
            }
        }

        if ((location & DEVICE) == DEVICE) {
            if constexpr (IS_HD_LAMBDA(LambdaT) || IS_D_LAMBDA(LambdaT)) {

                const int num_patches = this->get_num_patches();
                const int threads     = 256;
                detail::for_each_edge<<<num_patches, threads, 0, stream>>>(
                    num_patches, this->m_d_patches_info, apply);
            } else {
                RXMESH_ERROR(
                    "RXMeshStatic::for_each_edge() Input lambda function "
                    "should be annotated with  __device__ for execution on "
                    "device");
            }
        }
    }

    /**
     * @brief Apply a lambda function on all faces in the mesh
     * @tparam LambdaT type of the lambda function (inferred)
     * @param location the execution location
     * @param apply lambda function to be applied on all faces. The lambda
     * function signature takes a FaceHandle
     * @param stream the stream used to run the kernel in case of DEVICE
     * execution location
     */
    template <typename LambdaT>
    void for_each_face(locationT    location,
                       LambdaT      apply,
                       cudaStream_t stream = NULL)
    {
        if ((location & HOST) == HOST) {
            const int num_patches = this->get_num_patches();
#pragma omp parallel for
            for (int p = 0; p < num_patches; ++p) {
                for (int f = 0; f < this->m_h_patches_info[p].num_faces[0];
                     ++f) {

                    if (detail::is_owned(f, m_h_patches_info[p].owned_mask_f) &&
                        !detail::is_deleted(
                            f, m_h_patches_info[p].active_mask_f)) {
                        const FaceHandle f_handle(static_cast<uint32_t>(p), f);
                        apply(f_handle);
                    }
                }
            }
        }

        if ((location & DEVICE) == DEVICE) {
            if constexpr (IS_HD_LAMBDA(LambdaT) || IS_D_LAMBDA(LambdaT)) {

                const int num_patches = this->get_num_patches();
                const int threads     = 256;
                detail::for_each_face<<<num_patches, threads, 0, stream>>>(
                    num_patches, this->m_d_patches_info, apply);
            } else {
                RXMESH_ERROR(
                    "RXMeshStatic::for_each_face() Input lambda function "
                    "should be annotated with  __device__ for execution on "
                    "device");
            }
        }
    }

    /**
     * @brief populate the launch_box with grid size and dynamic shared memory
     * needed for kernel launch
     * @param op List of query operations done inside this the kernel
     * @param launch_box input launch box to be populated
     * @param kernel The kernel to be launched
     * @param oriented if the query is oriented. Valid only for Op::VV and
     * Op::VE queries
     */
    template <uint32_t blockThreads>
    void prepare_launch_box(const std::vector<Op>    op,
                            LaunchBox<blockThreads>& launch_box,
                            const void*              kernel,
                            const bool               oriented = false) const
    {

        launch_box.blocks         = this->m_num_patches;
        launch_box.smem_bytes_dyn = 0;

        for (auto o : op) {
            launch_box.smem_bytes_dyn = std::max(
                launch_box.smem_bytes_dyn,
                this->template calc_shared_memory<blockThreads>(o, oriented));
        }

        if (!this->m_quite) {
            RXMESH_TRACE(
                "RXMeshStatic::calc_shared_memory() launching {} blocks with "
                "{} threads on the device",
                launch_box.blocks,
                blockThreads);
        }

        check_shared_memory(launch_box.smem_bytes_dyn,
                            launch_box.smem_bytes_static,
                            launch_box.num_registers_per_thread,
                            blockThreads,
                            kernel);
    }


    /**
     * @brief Adding a new face attribute
     * @tparam T type of the attribute
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param num_attributes number of the attributes
     * @param location where to allocate the attributes
     * @param layout as SoA or AoS
     * operations
     * @return shared pointer to the created attribute
     */
    template <class T>
    std::shared_ptr<FaceAttribute<T>> add_face_attribute(
        const std::string& name,
        uint32_t           num_attributes,
        locationT          location = LOCATION_ALL,
        layoutT            layout   = SoA)
    {
        return m_attr_container->template add<FaceAttribute<T>>(
            name.c_str(), num_attributes, location, layout, this);
    }

    /**
     * @brief Adding a new face attribute by reading values from a host buffer
     * f_attributes where the order of faces is the same as the order of
     * faces given to the constructor.The attributes are populated on device
     * and host
     * @tparam T type of the attribute
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param layout as SoA or AoS
     * operations
     * @return shared pointer to the created attribute
     */
    template <class T>
    std::shared_ptr<FaceAttribute<T>> add_face_attribute(
        const std::vector<std::vector<T>>& f_attributes,
        const std::string&                 name,
        layoutT                            layout = SoA)
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

        uint32_t num_attributes = f_attributes[0].size();

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

    /**
     * @brief Adding a new face attribute by reading values from a host buffer
     * f_attributes where the order of faces is the same as the order of
     * faces given to the constructor.The attributes are populated on device
     * and host
     * @tparam T type of the attribute
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param layout as SoA or AoS
     * operations
     * @return shared pointer to the created attribute
     */
    template <class T>
    std::shared_ptr<FaceAttribute<T>> add_face_attribute(
        const std::vector<T>& f_attributes,
        const std::string&    name,
        layoutT               layout = SoA)
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

    /**
     * @brief Adding a new edge attribute
     * @tparam T type of the attribute
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param num_attributes number of the attributes
     * @param location where to allocate the attributes
     * @param layout as SoA or AoS
     * operations
     * @return shared pointer to the created attribute
     */
    template <class T>
    std::shared_ptr<EdgeAttribute<T>> add_edge_attribute(
        const std::string& name,
        uint32_t           num_attributes,
        locationT          location = LOCATION_ALL,
        layoutT            layout   = SoA)
    {
        return m_attr_container->template add<EdgeAttribute<T>>(
            name.c_str(), num_attributes, location, layout, this);
    }

    /**
     * @brief Adding a new vertex attribute
     * @tparam T type of the attribute
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param num_attributes number of the attributes
     * @param location where to allocate the attributes
     * @param layout as SoA or AoS
     * operations
     * @return shared pointer to the created attribute
     */
    template <class T>
    std::shared_ptr<VertexAttribute<T>> add_vertex_attribute(
        const std::string& name,
        uint32_t           num_attributes,
        locationT          location = LOCATION_ALL,
        layoutT            layout   = SoA)
    {
        return m_attr_container->template add<VertexAttribute<T>>(
            name.c_str(), num_attributes, location, layout, this);
    }

    /**
     * @brief Adding a new vertex attribute by reading values from a host buffer
     * v_attributes where the order of vertices is the same as the order of
     * vertices given to the constructor. The attributes are populated on device
     * and host
     * @tparam T type of the attribute
     * @param v_attributes attributes to read
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param layout as SoA or AoS
     * operations
     * @return shared pointer to the created attribute
     */
    template <class T>
    std::shared_ptr<VertexAttribute<T>> add_vertex_attribute(
        const std::vector<std::vector<T>>& v_attributes,
        const std::string&                 name,
        layoutT                            layout = SoA)
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

        uint32_t num_attributes = v_attributes[0].size();

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

    /**
     * @brief Adding a new vertex attribute by reading values from a host buffer
     * v_attributes where the order of vertices is the same as the order of
     * vertices given to the constructor. The attributes are populated on device
     * and host
     * @tparam T type of the attribute
     * @param v_attributes attributes to read
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param layout as SoA or AoS
     * operations
     * @return shared pointer to the created attribute
     */
    template <class T>
    std::shared_ptr<VertexAttribute<T>> add_vertex_attribute(
        const std::vector<T>& v_attributes,
        const std::string&    name,
        layoutT               layout = SoA)
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

                uint32_t global_v = m_h_patches_ltog_v[p][v];

                (*ret)(v_handle, 0) = v_attributes[global_v];
            }
        }

        // move to device
        ret->move(rxmesh::HOST, rxmesh::DEVICE);
        return ret;
    }

    /**
     * @brief Checks if an attribute exists given its name
     * @param name the attribute name
     * @return True if the attribute exists. False otherwise.
     */
    bool does_attribute_exist(const std::string& name)
    {
        return m_attr_container->does_exist(name.c_str());
    }

    /**
     * @brief Remove an attribute. Could be vertex, edge, or face attribute
     * @param name the attribute name
     */
    void remove_attribute(const std::string& name)
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


    /**
     * @brief return a shared pointer the input vertex position
     */
    std::shared_ptr<VertexAttribute<float>> get_input_vertex_coordinates()
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

    /**
     * @brief Map a vertex handle into a global index as seen in the input
     * to RXMeshStatic
     * @param vh input vertex handle
     * @return the global index of vh
     */
    uint32_t map_to_global(const VertexHandle vh) const
    {
        auto pl = vh.unpack();
        return m_h_patches_ltog_v[pl.first][pl.second];
    }

    /**
     * @brief Map an edge handle into a global index
     * @param eh input edge handle
     * @return the global index of eh
     */
    uint32_t map_to_global(const EdgeHandle eh) const
    {
        auto pl = eh.unpack();
        return m_h_patches_ltog_e[pl.first][pl.second];
    }

    /**
     * @brief Map a face handle into a global index as seen in the input
     * to RXMeshStatic
     * @param vh input face handle
     * @return the global index of fh
     */
    uint32_t map_to_global(const FaceHandle fh) const
    {
        auto pl = fh.unpack();
        return m_h_patches_ltog_f[pl.first][pl.second];
    }

    /**
     * @brief compute a linear compact index for a give vertex/edge/face handle
     * @tparam HandleT the type of the input handle
     * @param input handle
     */
    template <typename HandleT>
    uint32_t linear_id(HandleT input)
    {
        using LocalT = typename HandleT::LocalT;

        if (!input.is_valid()) {
            RXMESH_ERROR("RXMeshStatic::linear_id() input handle is not valid");
        }


        if (input.patch_id() >= get_num_patches()) {
            RXMESH_ERROR(
                "RXMeshStatic::linear_id() patch index ({}) is out-of-bound",
                input.patch_id());
        }

        const HandleT owner_handle =
            Context::get_owner_handle(input, nullptr, m_h_patches_info);

        uint32_t p_id = owner_handle.unpack().first;
        uint16_t ret  = owner_handle.unpack().second;

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

    /**
     * @brief Export the mesh to obj file
     * @tparam T type of vertices coordinates
     * @param filename the output file
     * @param coords vertices coordinates
     */
    template <typename T>
    void export_obj(const std::string&        filename,
                    const VertexAttribute<T>& coords)
    {
        std::string  fn = filename;
        std::fstream file(fn, std::ios::out);
        file.precision(30);

        uint32_t num_v = 0;
        for (uint32_t p = 0; p < this->m_num_patches; ++p) {

            const uint32_t p_num_vertices =
                this->m_h_patches_info[p].num_vertices[0];

            for (uint16_t v = 0; v < p_num_vertices; ++v) {

                const VertexHandle vh = Context::get_owner_handle<VertexHandle>(
                    {p, {v}}, nullptr, m_h_patches_info);

                file << "v " << coords(vh, 0) << " " << coords(vh, 1) << " "
                     << coords(vh, 2) << std::endl;
            }

            const uint32_t p_num_faces = this->m_h_patches_info[p].num_faces[0];

            for (uint32_t f = 0; f < p_num_faces; ++f) {
                if (!detail::is_deleted(
                        f, this->m_h_patches_info[p].active_mask_f) &&
                    detail::is_owned(f,
                                     this->m_h_patches_info[p].owned_mask_f)) {
                    file << "f ";
                    for (uint32_t e = 0; e < 3; ++e) {
                        uint16_t edge =
                            this->m_h_patches_info[p].fe[3 * f + e].id;
                        flag_t dir(0);
                        Context::unpack_edge_dir(edge, edge, dir);
                        uint16_t e_id = (2 * edge) + dir;
                        uint16_t v    = this->m_h_patches_info[p].ev[e_id].id;
                        file << v + num_v + 1 << " ";
                    }
                    file << std::endl;
                }
            }

            num_v += p_num_vertices;
        }
    }


   protected:
    template <uint32_t blockThreads>
    size_t calc_shared_memory(const Op op, const bool oriented) const
    {
        // Operations that uses matrix transpose needs a template parameter
        // that is by default TRANSPOSE_ITEM_PER_THREAD. Here we check if
        // this default parameter is valid otherwise, it needs to be increased.
        if (op == Op::VV || op == Op::VE) {
            if (2 * this->m_max_edges_per_patch >
                blockThreads * TRANSPOSE_ITEM_PER_THREAD) {
                RXMESH_ERROR(
                    "RXMeshStatic::calc_shared_memory() "
                    "TRANSPOSE_ITEM_PER_THREAD = {} needs "
                    "to be increased for op = {}",
                    TRANSPOSE_ITEM_PER_THREAD,
                    op_to_string(op));
            }
        } else if (op == Op::VE || op == Op::EF || op == Op::FF) {
            if (3 * this->m_max_faces_per_patch >
                blockThreads * TRANSPOSE_ITEM_PER_THREAD) {
                RXMESH_ERROR(
                    "RXMeshStatic::calc_shared_memory() "
                    "TRANSPOSE_ITEM_PER_THREAD = {} needs "
                    "to be increased for op = {}",
                    TRANSPOSE_ITEM_PER_THREAD,
                    op_to_string(op));
            }
        }


        if (oriented && !(op == Op::VV || op == Op::VE)) {
            RXMESH_ERROR(
                "RXMeshStatic::calc_shared_memory() Oriented is only "
                "allowed on VV and VE. The input op is {}",
                op_to_string(op));
        }

        if (op == Op::EVDiamond && !m_is_input_edge_manifold) {
            RXMESH_ERROR(
                "RXMeshStatic::calc_shared_memory() Op::EVDiamond only works "
                "on edge manifold mesh. The input mesh is not edge manifold");
        }

        size_t dynamic_smem = 0;


        if (op == Op::FE) {
            // only FE will be loaded
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::FACE);

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::EDGE);

            // stores edges LP hashtable
            dynamic_smem +=
                sizeof(LPPair) * max_lp_hashtable_size(ELEMENT::EDGE);

            // for possible padding for alignment
            // 4 since there are 4 calls for ShmemAllocator.alloc
            dynamic_smem += ShmemAllocator::default_alignment * 4;

        } else if (op == Op::EV) {
            // only EV will be loaded
            dynamic_smem = 2 * this->m_max_edges_per_patch * sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::EDGE);

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::VERTEX);

            // stores vertex LP hashtable
            dynamic_smem +=
                sizeof(LPPair) * max_lp_hashtable_size(ELEMENT::VERTEX);

            // for possible padding for alignment
            // 4 since there are 4 calls for ShmemAllocator.alloc
            dynamic_smem += ShmemAllocator::default_alignment * 4;

        } else if (op == Op::FV) {
            // We load both FE and EV. We don't change EV.
            // FE are updated to contain FV instead of FE by reading from
            // EV. After that, we can throw EV away so we can load hashtable
            dynamic_smem += 3 * this->m_max_faces_per_patch * sizeof(uint16_t);
            dynamic_smem += 2 * this->m_max_edges_per_patch * sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::FACE);

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::VERTEX);

            //  stores vertex LP hashtable
            uint32_t table_bytes =
                sizeof(LPPair) * max_lp_hashtable_size(ELEMENT::VERTEX);
            if (table_bytes >
                2 * this->m_max_edges_per_patch * sizeof(uint16_t)) {

                dynamic_smem += table_bytes - 2 * this->m_max_edges_per_patch *
                                                  sizeof(uint16_t);
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
            // the EV (i.e., 2*#edges) since this output buffer will stored the
            // nnz and the nnz of a matrix the same before/after transpose
            dynamic_smem =
                (2 * 2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::VERTEX);

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::EDGE);

            // stores edge LP hashtable
            uint32_t lp_smem =
                sizeof(LPPair) * max_lp_hashtable_size(ELEMENT::EDGE);


            // For oriented VE, we additionally need to store FE and EF
            // along with the (transposed) VE. FE needs 3*max_num_faces. Since
            // oriented is only done on manifold, EF needs only 2*max_num_edges
            // since every edge is neighbor to maximum of two faces (which we
            // write on the same place as the extra EV)

            uint32_t fe_ef_smem =
                (2 * this->m_max_edges_per_patch) * sizeof(uint16_t) +
                (3 * this->m_max_faces_per_patch) * sizeof(uint16_t);

            if (oriented) {
                dynamic_smem += std::max(lp_smem, fe_ef_smem);
            } else {
                dynamic_smem += lp_smem;
            }

            // for possible padding for alignment
            // 4 since there are 4 calls for ShmemAllocator.alloc
            dynamic_smem += ShmemAllocator::default_alignment * 6;

        } else if (op == Op::EF) {
            // same as Op::VE but with faces
            dynamic_smem =
                (2 * 3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
                sizeof(uint16_t) + sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::EDGE);

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::FACE);

            // stores the face LP hashtable
            dynamic_smem +=
                sizeof(LPPair) * max_lp_hashtable_size(ELEMENT::FACE);

            // for possible padding for alignment
            // 4 since there are 4 calls for ShmemAllocator.alloc
            dynamic_smem += ShmemAllocator::default_alignment * 4;

        } else if (op == Op::VF) {
            // load EV and FE simultaneously. changes FE to FV using EV. Then
            // transpose FV in place and use EV to store the values/output while
            // using FV to store the prefix sum. Thus, the space used to store
            // EV should be max(3*#faces, 2*#edges)
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t);
            dynamic_smem += std::max(3 * this->m_max_faces_per_patch,
                                     2 * this->m_max_edges_per_patch) *
                                sizeof(uint16_t) +
                            sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::VERTEX);

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::FACE);

            // stores the face LP hashtable
            dynamic_smem +=
                sizeof(LPPair) * max_lp_hashtable_size(ELEMENT::FACE);

            // for possible padding for alignment
            // 5 since there are 5 calls for ShmemAllocator.alloc
            dynamic_smem += ShmemAllocator::default_alignment * 5;

        } else if (op == Op::VV) {
            // similar to VE but we also need to store the EV even after
            // we do the transpose. After that, we can throw EV away and load
            // the hash table
            dynamic_smem =
                (2 * 2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::VERTEX);

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::VERTEX);

            // duplicate EV
            uint32_t ev_smem =
                (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

            // stores the vertex LP hashtable
            uint32_t lp_smem =
                sizeof(LPPair) * max_lp_hashtable_size(ELEMENT::VERTEX);

            if (oriented) {
                // For oriented VV, we additionally need to store FE and EF
                // along with the (transposed) VE. FE needs 3*max_num_faces.
                // Since oriented is only done on manifold, EF needs only
                // 2*max_num_edges since every edge is neighbor to maximum of
                // two faces (which we write on the same place as the extra EV).
                // With VV, we need to reload EV again (since it is overwritten)

                uint32_t fe_smem =
                    (3 * this->m_max_faces_per_patch) * sizeof(uint16_t);

                dynamic_smem += std::max(lp_smem, ev_smem + fe_smem);

            } else {
                dynamic_smem += std::max(lp_smem, ev_smem);
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

            dynamic_smem = (3 * this->m_max_faces_per_patch +        // FE
                            2 * (3 * this->m_max_faces_per_patch) +  // EF
                            4 * this->m_max_faces_per_patch) *       // FF
                           sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::FACE);

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::FACE);

            // for possible padding for alignment
            // 6 since there are 6 calls for ShmemAllocator.alloc
            dynamic_smem += ShmemAllocator::default_alignment * 6;
        } else if (op == Op::EVDiamond) {
            // to load EV and also store the results which contains 4 vertices
            // for each edge
            dynamic_smem = 4 * this->m_max_edges_per_patch * sizeof(uint16_t);

            // to store FE
            uint32_t fe_smem =
                3 * this->m_max_faces_per_patch * sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::EDGE);

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size(ELEMENT::VERTEX);

            // stores vertex LP hashtable
            uint32_t lp_smem =
                sizeof(LPPair) * max_lp_hashtable_size(ELEMENT::VERTEX);

            dynamic_smem += std::max(fe_smem, lp_smem);

            // for possible padding for alignment
            // 4 since there are 4 calls for ShmemAllocator.alloc
            dynamic_smem += ShmemAllocator::default_alignment * 5;
        }

        if (oriented) {
            if (op == Op::VE) {
                // For VE, we need to add the extra memory we needed for VV that
                // load EV beside the VE
                dynamic_smem +=
                    (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);
            }
            // For oriented VV or VE, we additionally need to store FE and EF
            // along with the (transposed) VE. FE needs 3*max_num_faces. Since
            // oriented is only done on manifold, EF needs only 2*max_num_edges
            // since every edge is neighbor to maximum of two faces (which we
            // write on the same place as the extra EV). With VV, we need
            // to reload EV again (since it is overwritten) but we don't need to
            // do this for VE
            dynamic_smem +=
                (3 * this->m_max_faces_per_patch) * sizeof(uint16_t);
        }

        return dynamic_smem;
    }

    void check_shared_memory(const uint32_t smem_bytes_dyn,
                             size_t&        smem_bytes_static,
                             uint32_t&      num_reg_per_thread,
                             const uint32_t num_threads_per_block,
                             const void*    kernel) const
    {
        // check if total shared memory (static + dynamic) consumed by
        // k_base_query are less than the max shared per block
        cudaFuncAttributes func_attr = cudaFuncAttributes();
        CUDA_ERROR(cudaFuncGetAttributes(&func_attr, kernel));

        smem_bytes_static  = func_attr.sharedSizeBytes;
        num_reg_per_thread = static_cast<uint32_t>(func_attr.numRegs);
        int device_id;
        CUDA_ERROR(cudaGetDevice(&device_id));
        cudaDeviceProp devProp;
        CUDA_ERROR(cudaGetDeviceProperties(&devProp, device_id));

        if (!this->m_quite) {
            int num_blocks_per_sm = 0;
            CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks_per_sm,
                kernel,
                num_threads_per_block,
                smem_bytes_dyn));

            RXMESH_TRACE(
                "RXMeshStatic::check_shared_memory() user function requires "
                "shared memory = {} (dynamic) + {} (static) = {} (bytes) and "
                "{} registers per thread with occupancy of {} blocks/SM",
                smem_bytes_dyn,
                smem_bytes_static,
                smem_bytes_dyn + smem_bytes_static,
                num_reg_per_thread,
                num_blocks_per_sm);

            RXMESH_TRACE(
                "RXMeshStatic::check_shared_memory() available total shared "
                "memory per block = {} (bytes) = {} (Kb)",
                devProp.sharedMemPerBlock,
                float(devProp.sharedMemPerBlock) / 1024.0f);
        }

        if (smem_bytes_static + smem_bytes_dyn > devProp.sharedMemPerBlock) {
            RXMESH_ERROR(
                " RXMeshStatic::check_shared_memory() shared memory needed for"
                " input function ({} bytes) exceeds the max shared memory "
                "per block on the current device ({} bytes)",
                smem_bytes_static + smem_bytes_dyn,
                devProp.sharedMemPerBlock);
            exit(EXIT_FAILURE);
        }
    }

#if USE_POLYSCOPE
    void add_patch_to_polyscope(const uint32_t                        p,
                                std::vector<std::array<uint32_t, 3>>& fv,
                                bool with_ribbon)
    {
        for (uint16_t f = 0; f < this->m_h_patches_info[p].num_faces[0]; ++f) {
            if (!detail::is_deleted(f,
                                    this->m_h_patches_info[p].active_mask_f)) {
                if (!with_ribbon) {
                    if (!detail::is_owned(
                            f, this->m_h_patches_info[p].owned_mask_f)) {
                        // fv.push_back({0, 0, 0});
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
                }
                fv.push_back(face);
            }
        }
    }


    void update_polyscope_edge_map()
    {
        m_polyscope_edges_map.clear();

        for (uint32_t p = 0; p < this->m_num_patches; ++p) {

            for (uint16_t e = 0; e < this->m_h_patches_info[p].num_edges[0];
                 ++e) {
                LocalEdgeT local_e(e);
                if (!this->m_h_patches_info[p].is_deleted(local_e) &&
                    this->m_h_patches_info[p].is_owned(local_e)) {

                    VertexHandle v0(
                        p, {this->m_h_patches_info[p].ev[2 * e + 0].id});
                    VertexHandle v1(
                        p, {this->m_h_patches_info[p].ev[2 * e + 1].id});

                    uint32_t v_org0 = linear_id(v0);
                    uint32_t v_org1 = linear_id(v1);

                    auto key    = detail::edge_key(v_org0, v_org1);
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
    void register_polyscope()
    {
        update_polyscope_edge_map();

        std::vector<std::array<uint32_t, 3>> fv;
        fv.reserve(m_num_faces);

        for (uint32_t p = 0; p < this->m_num_patches; ++p) {
            add_patch_to_polyscope(p, fv, false);
        }

        m_polyscope_mesh =
            polyscope::registerSurfaceMesh(m_polyscope_mesh_name,
                                           *m_input_vertex_coordinates,
                                           fv,
                                           m_polyscope_edges_map);
    }

    std::string             m_polyscope_mesh_name;
    polyscope::SurfaceMesh* m_polyscope_mesh;
    EdgeMapT                m_polyscope_edges_map;
#endif

    std::shared_ptr<AttributeContainer>     m_attr_container;
    std::shared_ptr<VertexAttribute<float>> m_input_vertex_coordinates;
};
}  // namespace rxmesh