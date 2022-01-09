#pragma once
#include <assert.h>
#include <fstream>
#include <memory>

#include <cuda_profiler_api.h>

#include "rxmesh/attribute.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/for_each.cuh"
#include "rxmesh/launch_box.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/types.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/timer.h"

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
     * @brief Main constructor used to initialize internal member variables
     * @param fv Face incident vertices as read from an obj file
     * @param quite run in quite mode
     */
    RXMeshStatic(std::vector<std::vector<uint32_t>>& fv,
                 const bool                          quite = false)
        : RXMesh(fv, quite)
    {
        m_attr_container = std::make_shared<AttributeContainer>();
    };

    virtual ~RXMeshStatic()
    {
    }


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
                     v < this->m_h_patches_info[p].num_owned_vertices;
                     ++v) {
                    const VertexHandle v_handle(static_cast<uint32_t>(p), v);
                    apply(v_handle);
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
                for (uint16_t e = 0;
                     e < this->m_h_patches_info[p].num_owned_edges;
                     ++e) {
                    const EdgeHandle e_handle(static_cast<uint32_t>(p), e);
                    apply(e_handle);
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
                for (int f = 0; f < this->m_h_patches_info[p].num_owned_faces;
                     ++f) {
                    const FaceHandle f_handle(static_cast<uint32_t>(p), f);
                    apply(f_handle);
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
     * @param oriented if the query is oriented. Valid only for Op::VV queries
     */
    template <uint32_t blockThreads>
    void prepare_launch_box(const std::vector<Op>    op,
                            LaunchBox<blockThreads>& launch_box,
                            const void*              kernel,
                            const bool               oriented = false) const
    {
        static_assert(
            blockThreads && ((blockThreads & (blockThreads - 1)) == 0),
            " RXMeshStatic::prepare_launch_box() CUDA block size should be of "
            "power 2");

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
            name.c_str(),
            this->m_h_num_owned_f,
            num_attributes,
            location,
            layout);
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
            name.c_str(),
            this->m_h_num_owned_f,
            num_attributes,
            LOCATION_ALL,
            layout);

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
            name.c_str(),
            this->m_h_num_owned_f,
            num_attributes,
            LOCATION_ALL,
            layout);

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
            name.c_str(),
            this->m_h_num_owned_e,
            num_attributes,
            location,
            layout);
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
            name.c_str(),
            this->m_h_num_owned_v,
            num_attributes,
            location,
            layout);
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
            name.c_str(),
            this->m_h_num_owned_v,
            num_attributes,
            LOCATION_ALL,
            layout);

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
            name.c_str(),
            this->m_h_num_owned_v,
            num_attributes,
            LOCATION_ALL,
            layout);

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
                this->m_h_patches_info[p].num_vertices;

            for (uint16_t v = 0; v < p_num_vertices; ++v) {
                uint16_t v_id = v;
                uint32_t p_id = p;
                if (v >= this->m_h_patches_info[p].num_owned_vertices) {
                    uint16_t l =
                        v - this->m_h_patches_info[p].num_owned_vertices;
                    v_id = this->m_h_patches_info[p].not_owned_id_v[l].id;
                    p_id = this->m_h_patches_info[p].not_owned_patch_v[l];
                }
                VertexHandle vh(p_id, {v_id});
                file << "v " << coords(vh, 0) << " " << coords(vh, 1) << " "
                     << coords(vh, 2) << std::endl;
            }

            const uint32_t p_num_faces =
                this->m_h_patches_info[p].num_owned_faces;

            for (uint32_t f = 0; f < p_num_faces; ++f) {

                file << "f ";
                for (uint32_t e = 0; e < 3; ++e) {
                    uint16_t edge = this->m_h_patches_info[p].fe[3 * f + e].id;
                    flag_t   dir(0);
                    Context::unpack_edge_dir(edge, edge, dir);
                    uint16_t e_id = (2 * edge) + dir;
                    uint16_t v    = this->m_h_patches_info[p].ev[e_id].id;
                    file << v + num_v + 1 << " ";
                }
                file << std::endl;
            }

            num_v += p_num_vertices;
        }
    }

   protected:
    template <uint32_t blockThreads>
    size_t calc_shared_memory(const Op op, const bool oriented = false) const
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


        if (oriented && op != Op::VV) {
            RXMESH_ERROR(
                "RXMeshStatic::calc_shared_memory() Oriented is only "
                "allowed on VV. The input op is {}",
                op_to_string(op));
        }

        if (oriented && op == Op::VV && !this->m_is_input_closed) {
            RXMESH_ERROR(
                "RXMeshStatic::calc_shared_memory() Can't generate oriented "
                "output (VV) for input with boundaries");
        }

        size_t dynamic_smem = 0;

        if (op == Op::FE) {
            // only FE will be loaded
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t);
            // to load not-owned edges local and patch id
            dynamic_smem += this->m_max_not_owned_edges *
                                (sizeof(uint16_t) + sizeof(uint32_t)) +
                            sizeof(uint16_t);
        } else if (op == Op::EV) {
            // only EV will be loaded
            dynamic_smem = 2 * this->m_max_edges_per_patch * sizeof(uint16_t);
            // to load not-owned vertices local and patch id
            dynamic_smem += this->m_max_not_owned_vertices *
                            (sizeof(uint16_t) + sizeof(uint32_t));
        } else if (op == Op::FV) {
            // We load both FE and EV. We don't change EV.
            // FE are updated to contain FV instead of FE by reading from
            // EV
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t) +
                           2 * this->m_max_edges_per_patch * sizeof(uint16_t);
            // no need for extra memory to load not-owned vertices local and
            // patch id. We load them and overwrite EV.
            const uint32_t not_owned_v_bytes =
                this->m_max_not_owned_vertices *
                (sizeof(uint16_t) + sizeof(uint32_t));
            const uint32_t edges_bytes =
                2 * this->m_max_edges_per_patch * sizeof(uint16_t);
            if (not_owned_v_bytes > edges_bytes) {
                // dynamic_smem += not_owned_v_bytes - edges_bytes;
                RXMESH_ERROR(
                    "RXMeshStatic::calc_shared_memory() FV query might fail!");
            }
        } else if (op == Op::VE) {
            // load EV and then transpose it in place
            // The transpose needs two buffer; one for prefix sum and another
            // for the actual output
            // The prefix sum will be stored in place (where EV are loaded)
            // The output will be stored in another buffer with size equal to
            // the EV (i.e., 2*#edges) since this output buffer will stored the
            // nnz and the nnz of a matrix the same before/after transpose
            dynamic_smem =
                (2 * 2 * this->m_max_edges_per_patch) * sizeof(uint16_t) +
                sizeof(uint16_t);

            // to load the not-owned edges local and patch id
            dynamic_smem += this->m_max_not_owned_edges *
                            (sizeof(uint16_t) + sizeof(uint32_t));
        } else if (op == Op::EF) {
            // same as Op::VE but with faces
            dynamic_smem =
                (2 * 3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
                sizeof(uint16_t) + sizeof(uint16_t);

            // to load the not-owned faces local and patch id
            dynamic_smem += this->m_max_not_owned_faces *
                            (sizeof(uint16_t) + sizeof(uint32_t));
        } else if (op == Op::VF) {
            // load EV and FE simultaneously. changes FE to FV using EV. Then
            // transpose FV in place and use EV to store the values/output while
            // using FV to store the prefix sum. Thus, the space used to store
            // EV should be max(3*#faces, 2*#edges)
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t) +
                           std::max(3 * this->m_max_faces_per_patch,
                                    2 * this->m_max_edges_per_patch) *
                               sizeof(uint16_t) +
                           sizeof(uint16_t);

            // to load the not-owned faces local and patch id
            dynamic_smem += this->m_max_not_owned_faces *
                            (sizeof(uint16_t) + sizeof(uint32_t));
        } else if (op == Op::VV) {
            // similar to VE but we also need to store the EV even after
            // we do the transpose
            dynamic_smem =
                (3 * 2 * this->m_max_edges_per_patch) * sizeof(uint16_t);
            // no need for extra memory to load not-owned local and patch id.
            // We load them and overwrite the extra EV
            if (this->m_max_not_owned_vertices *
                    (sizeof(uint16_t) + sizeof(uint32_t)) >
                (2 * this->m_max_edges_per_patch) * sizeof(uint16_t)) {
                RXMESH_ERROR(
                    "RXMeshStatic::calc_shared_memory() VV query might fail!");
            }
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
            // no need for extra memory to load not-owned faces local and
            // patch id. We load them and overwrite FE.
        }

        if (op == Op::VV && oriented) {
            // For oriented VV, we additionally need to store FE and EF along
            // with the (transposed) VE
            // FE needs 3*max_num_faces
            // Since oriented is only done on manifold, EF needs only
            // 2*max_num_edges since every edge is neighbor to maximum of two
            // faces (which we write on the same place as the extra EV)
            dynamic_smem +=
                (3 * this->m_max_faces_per_patch) * sizeof(uint16_t);
        }

        return dynamic_smem;
    }

    void check_shared_memory(const uint32_t smem_bytes_dyn,
                             size_t&        smem_bytes_static,
                             uint32_t&      num_reg_per_thread,
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
            RXMESH_TRACE(
                "RXMeshStatic::check_shared_memory() user function requires "
                "shared memory = {} (dynamic) + {} (static) = {} (bytes) and "
                "{} registers per thread",
                smem_bytes_dyn,
                smem_bytes_static,
                smem_bytes_dyn + smem_bytes_static,
                num_reg_per_thread);

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


    std::shared_ptr<AttributeContainer> m_attr_container;
};
}  // namespace rxmesh