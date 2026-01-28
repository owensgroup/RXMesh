#pragma once
#include <assert.h>
#include <fstream>
#include <functional>
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

#include "rxmesh/kernels/boundary.cuh"
#include "rxmesh/kernels/query_kernel.cuh"

#if USE_POLYSCOPE
#include "polyscope/surface_mesh.h"
#endif

#include <glm/fwd.hpp>

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
     */
    explicit RXMeshStatic(const std::string file_path,
                          const std::string patcher_file             = "",
                          const uint32_t    patch_size               = 512,
                          const float       capacity_factor          = 1.0,
                          const float       patch_alloc_factor       = 1.0,
                          const float       lp_hashtable_load_factor = 0.8);

    /**
     * @brief Constructor using triangles and vertices
     * @param fv Face incident vertices as read from an obj file
     */
    explicit RXMeshStatic(std::vector<std::vector<uint32_t>>& fv,
                          const std::string                   patcher_file = "",
                          const uint32_t                      patch_size = 512,
                          const float capacity_factor                    = 1.0,
                          const float patch_alloc_factor                 = 1.0,
                          const float lp_hashtable_load_factor           = 0.8);

    /**
     * @brief Constructor using path to multiple meshes
     */
    explicit RXMeshStatic(const std::vector<std::string> files_path,
                          const uint32_t                 patch_size = 512);

    /**
     * @brief Add vertex coordinates to the input mesh. When calling
     * RXMeshStatic constructor that takes the face's vertices, this function
     * can be called to then add vertex coordinates and also add the mesh to
     * polyscope if it is active. You don't need to call this function if you
     * are constructing RXMeshStatic with the constructor that takes the path to
     * mesh file
     */
    void add_vertex_coordinates(std::vector<std::vector<float>>& vertices,
                                std::string mesh_name = "");

    virtual ~RXMeshStatic() = default;

#if USE_POLYSCOPE
    /**
     * @brief return a pointer to polyscope surface which has been registered
     * with this instance
     */
    polyscope::SurfaceMesh* get_polyscope_mesh();


    /**
     * @brief add a patch as a separate SurfaceMesh to polyscope renderer. The
     * patch is added along with its ribbon which could be helpful for debugging
     * @param p the patch id which will be added
     * @param with_vertex_patch add a vertex color quantity that show the vertex
     * patch and local ID
     * @param with_edge_patch add an edge color quantity that show the edge
     * patch and local ID
     * @param with_face_patch add a face color quantity that show the face
     * patch and local ID
     */
    polyscope::SurfaceMesh* render_patch(const uint32_t p,
                                         bool with_vertex_patch = true,
                                         bool with_edge_patch   = true,
                                         bool with_face_patch   = true);

    /**
     * @brief add the face's patch and local ID scalar quantities to a polyscope
     * instance (polyscope_mesh) for specific patch. polyscope_mesh should be
     * the one returned from render_patch call with the same input patch (p)
     * @param p patch id for which the face patch will be added
     * @param polyscope_mesh the SurfaceMesh pointer returned by calling
     * render_patch with the same input patch
     */
    void render_face_patch_and_local_id(const uint32_t          p,
                                        polyscope::SurfaceMesh* polyscope_mesh);

    /**
     * @brief add the edge's patch and local ID scalar quantities to a polyscope
     * instance (polyscope_mesh) for specific patch. polyscope_mesh should be
     * the one returned from render_patch call with the same input patch (p)
     * @param p patch id for which the face patch will be added
     * @param polyscope_mesh the SurfaceMesh pointer returned by calling
     * render_patch with the same input patch
     */
    void render_edge_patch_and_local_id(const uint32_t          p,
                                        polyscope::SurfaceMesh* polyscope_mesh);

    /**
     * @brief add the vertex's patch and local ID scalar quantities to a
     * polyscope instance (polyscope_mesh) for specific patch. polyscope_mesh
     * should be the one returned from render_patch call with the same input
     * patch (p)
     * @param p patch id for which the face patch will be added
     * @param polyscope_mesh the SurfaceMesh pointer returned by calling
     * render_patch with the same input patch
     */
    void render_vertex_patch_and_local_id(
        const uint32_t          p,
        polyscope::SurfaceMesh* polyscope_mesh);


    /**
     * @brief add the face's patch scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's face scalar quantity
     */
    polyscope::SurfaceFaceScalarQuantity* render_face_patch();

    /**
     * @brief add the edge's patch scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's edge scalar quantity
     */
    polyscope::SurfaceEdgeScalarQuantity* render_edge_patch();


    /**
     * @brief add the vertex's patch scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's vertex scalar quantity
     */
    polyscope::SurfaceVertexScalarQuantity* render_vertex_patch();


    /**
     * @brief add the face's local ID scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's face scalar quantity
     */
    polyscope::SurfaceFaceScalarQuantity* render_face_local_id();


    /**
     * @brief add the edge's local ID scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's edge scalar quantity
     */
    polyscope::SurfaceEdgeScalarQuantity* render_edge_local_id();

    /**
     * @brief add the vertex's local ID scalar quantity to the polyscope
     * instance associated RXMeshStatic
     * @return pointer to polyscope's vertex scalar quantity
     */
    polyscope::SurfaceVertexScalarQuantity* render_vertex_local_id();
#endif

    /**
     * @brief Apply a lambda function on all vertices in the mesh
     * @tparam LambdaT type of the lambda function (inferred)
     * @param location the execution location
     * @param apply lambda function to be applied on all vertices. The lambda
     * function signature takes a VertexHandle
     * @param stream the stream used to run the kernel in case of DEVICE
     * execution location
     * @param with_omp for HOST execution, use OpenMP where each patch is
     * assigned to a thread
     */
    template <typename LambdaT>
    void for_each_vertex(locationT    location,
                         LambdaT      apply,
                         cudaStream_t stream   = NULL,
                         bool         with_omp = true) const
    {
        if ((location & HOST) == HOST) {
            const int num_patches = this->get_num_patches();

            auto run = [&](int p) {
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
            };

            if (!with_omp) {
                for (int p = 0; p < num_patches; ++p) {
                    run(p);
                }
            } else {
#pragma omp parallel for
                for (int p = 0; p < num_patches; ++p) {
                    run(p);
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
     * @param with_omp for HOST execution, use OpenMP where each patch is
     * assigned to a thread
     */
    template <typename LambdaT>
    void for_each_edge(locationT    location,
                       LambdaT      apply,
                       cudaStream_t stream   = NULL,
                       bool         with_omp = true) const
    {
        if ((location & HOST) == HOST) {
            const int num_patches = this->get_num_patches();

            auto run = [&](int p) {
                for (uint16_t e = 0; e < this->m_h_patches_info[p].num_edges[0];
                     ++e) {

                    if (detail::is_owned(e, m_h_patches_info[p].owned_mask_e) &&
                        !detail::is_deleted(
                            e, m_h_patches_info[p].active_mask_e)) {

                        const EdgeHandle e_handle(static_cast<uint32_t>(p), e);
                        apply(e_handle);
                    }
                }
            };

            if (!with_omp) {
                for (int p = 0; p < num_patches; ++p) {
                    run(p);
                }
            } else {
#pragma omp parallel for
                for (int p = 0; p < num_patches; ++p) {
                    run(p);
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
     * @param with_omp for HOST execution, use OpenMP where each patch is
     * assigned to a thread
     */
    template <typename LambdaT>
    void for_each_face(locationT    location,
                       LambdaT      apply,
                       cudaStream_t stream   = NULL,
                       bool         with_omp = true) const
    {
        if ((location & HOST) == HOST) {
            const int num_patches = this->get_num_patches();

            auto run = [&](int p) {
                for (int f = 0; f < this->m_h_patches_info[p].num_faces[0];
                     ++f) {

                    if (detail::is_owned(f, m_h_patches_info[p].owned_mask_f) &&
                        !detail::is_deleted(
                            f, m_h_patches_info[p].active_mask_f)) {
                        const FaceHandle f_handle(static_cast<uint32_t>(p), f);
                        apply(f_handle);
                    }
                }
            };


            if (!with_omp) {
                for (int p = 0; p < num_patches; ++p) {
                    run(p);
                }
            } else {
#pragma omp parallel for
                for (int p = 0; p < num_patches; ++p) {
                    run(p);
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
     * @brief same as for_each_vertex/edge/face where the type is defined via
     * template parameter
     */
    template <typename HandleT, typename LambdaT>
    void for_each(locationT    location,
                  LambdaT      apply,
                  cudaStream_t stream   = NULL,
                  bool         with_omp = true)
    {
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            for_each_vertex(location, apply, stream, with_omp);
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            for_each_edge(location, apply, stream, with_omp);
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            for_each_face(location, apply, stream, with_omp);
        }
    }


    /**
     * @brief Launching a kernel knowing its launch box
     * @tparam ...ArgsT inferred
     * @tparam blockThreads the block size
     * @param lb launch box populated via prepare_launch_box
     * @param kernel the kernel to launch
     * @param stream to launch the kernel on
     * @param ...args input parameters to the kernel
     */
    template <uint32_t blockThreads, typename KernelT, typename... ArgsT>
    void run_kernel(const LaunchBox<blockThreads>& lb,
                    const KernelT                  kernel,
                    cudaStream_t                   stream,
                    ArgsT... args) const
    {
        kernel<<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn, stream>>>(
            get_context(), args...);
    }

    /**
     * @brief Launching a kernel knowing its launch box on the default stream
     * @tparam ...ArgsT infered
     * @tparam blockThreads the block size
     * @param lb launch box populated via prepare_launch_box
     * @param kernel the kernel to launch
     * @param ...args input parameters to the kernel
     */
    template <uint32_t blockThreads, typename KernelT, typename... ArgsT>
    void run_kernel(const LaunchBox<blockThreads>& lb,
                    const KernelT                  kernel,
                    ArgsT... args) const
    {
        kernel<<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(get_context(),
                                                                 args...);
    }

    /**
     * @brief run a kernel that will require a query operation
     * @tparam ...ArgsT infered
     * @tparam blockThreads the block size
     * @param op list of query operations used inside the kernel
     * @param kernel the kernel to run
     * @param ...args the inputs to the kernel
     */
    template <uint32_t blockThreads, typename KernelT, typename... ArgsT>
    void run_kernel(const std::vector<Op> op,
                    KernelT               kernel,
                    ArgsT... args) const
    {
        run_kernel<blockThreads>(
            kernel,
            op,
            false,
            false,
            false,
            [](uint32_t v, uint32_t e, uint32_t f) -> size_t { return 0; },
            NULL,
            args...);
    }

    /**
     * @brief run a kernel that will require a query operation
     * @tparam ...ArgsT infered
     * @tparam blockThreads the block size
     * @param op list of query operations used inside the kernel
     * @param kernel the kernel to run
     * @param oriented are the query operation required to be oriented
     * @param with_vertex_valence if vertex valence is requested to be
     * pre-computed and stored in shared memory
     * @param is_concurrent in case of multiple queries (i.e., op.size() > 1),
     * this parameter indicates if queries needs to be access at the same time
     * @param user_shmem a (lambda) function that takes the number of vertices,
     * edges, and faces and returns additional user-desired shared memory in
     * bytes. In case no extra shared memory needed, it can be
     * [](uint32_t v, uint32_t e, uint32_t f) { return 0; }
     * @param ...args the inputs to the kernel
     */
    template <uint32_t blockThreads, typename KernelT, typename... ArgsT>
    void run_kernel(
        KernelT                                             kernel,
        const std::vector<Op>                               op,
        const bool                                          oriented,
        const bool                                          with_vertex_valence,
        const bool                                          is_concurrent,
        std::function<size_t(uint32_t, uint32_t, uint32_t)> user_shmem,
        cudaStream_t                                        stream,
        ArgsT... args) const
    {
        LaunchBox<blockThreads> lb;

        prepare_launch_box(op,
                           lb,
                           (void*)kernel,
                           oriented,
                           with_vertex_valence,
                           is_concurrent,
                           user_shmem);

        run_kernel(lb, kernel, args...);
    }

    /**
     * @brief launch a kernel that require a query operation. This is limited to
     * one query only.
     * @tparam LambdaT inferred
     * @tparam blockThreads the size of cuda block
     * @tparam op the type of query operation
     * @param lb the launch box as initialized by prepare_launch_box
     * @param user_lambda the user lambda function which has the signature
     *      [=]__device__(InputHandle h, OutputIterator iter) {
     *      }
     * The InputHandle is a vertex, edge, or face handle depending on the input
     * to the query operation op. The OutputIterator is an vertex, edge, or face
     * iterator depending on the output of the query operation op.
     *
     * @param oriented if the query operation op is oriented
     * @param stream the stream to launch the kernel on
     */
    template <Op op, uint32_t blockThreads, typename LambdaT>
    void run_query_kernel(const LambdaT user_lambda,
                          const bool    oriented = false,
                          cudaStream_t  stream   = NULL) const
    {
        LaunchBox<blockThreads> lb;

        prepare_launch_box(
            {op},
            lb,
            (void*)detail::query_kernel<blockThreads, op, LambdaT>,
            oriented);

        run_query_kernel<op>(lb, user_lambda, oriented, stream);
    }

    /**
     * @brief launch a kernel that require a query operation. This is limited to
     * one query only.
     * @tparam LambdaT inferred
     * @tparam blockThreads the size of cuda block
     * @tparam op the type of query operation
     * @param lb the launch box as initialized by prepare_launch_box
     * @param user_lambda the user lambda function which has the signature
     *      [=]__device__(InputHandle h, OutputIterator iter) {
     *      }
     * The InputHandle is a vertex, edge, or face handle depending on the input
     * to the query operation op. The OutputIterator is an vertex, edge, or face
     * iterator depending on the output of the query operation op.
     *
     * @param oriented if the query operation op is oriented
     * @param stream the stream to launch the kernel on
     */
    template <Op op, uint32_t blockThreads, typename LambdaT>
    void run_query_kernel(LaunchBox<blockThreads> lb,
                          const LambdaT           user_lambda,
                          const bool              oriented = false,
                          cudaStream_t            stream   = NULL) const
    {
        detail::query_kernel<blockThreads, op>
            <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn, stream>>>(
                get_context(), oriented, user_lambda);
    }


    /**
     * @brief populate the launch_box with grid size and dynamic shared memory
     * needed for kernel launch
     * @param op List of query operations done inside this the kernel
     * @param launch_box input launch box to be populated
     * @param kernel The kernel to be launched
     * @param oriented if the query is oriented. Valid only for Op::VV and
     * Op::VE queries
     * @param with_vertex_valence if vertex valence is requested to be
     * pre-computed and stored in shared memory
     * @param is_concurrent: in case of multiple queries (i.e., op.size() > 1),
     * this parameter indicates if queries needs to be access at the same time
     * @param user_shmem a (lambda) function that takes the number of vertices,
     * edges, and faces and returns additional user-desired shared memory in
     * bytes
     */
    template <uint32_t blockThreads>
    void prepare_launch_box(
        const std::vector<Op>    op,
        LaunchBox<blockThreads>& launch_box,
        const void*              kernel,
        const bool               oriented            = false,
        const bool               with_vertex_valence = false,
        const bool               is_concurrent       = false,
        std::function<size_t(uint32_t, uint32_t, uint32_t)> user_shmem =
            [](uint32_t v, uint32_t e, uint32_t f) { return 0; }) const
    {

        launch_box.blocks         = this->m_num_patches;
        launch_box.smem_bytes_dyn = 0;

        for (auto o : op) {
            size_t sh = this->template calc_shared_memory<blockThreads>(
                o, oriented, false);
            if (is_concurrent) {
                launch_box.smem_bytes_dyn += sh;
            } else {
                launch_box.smem_bytes_dyn =
                    std::max(launch_box.smem_bytes_dyn, sh);
            }
        }

        launch_box.smem_bytes_dyn += user_shmem(m_max_vertices_per_patch,
                                                m_max_edges_per_patch,
                                                m_max_faces_per_patch);

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
        layoutT            layout   = SoA);

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
        layoutT                            layout = SoA);

    /**
     * @brief Adding a new face attribute similar to another face attribute
     * in allocation, number of attributes, and layout
     * @tparam T type of the returned attribute
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param other the other face attribute
     * @return shared pointer to the created face attribute
     */
    template <class T>
    std::shared_ptr<FaceAttribute<T>> add_face_attribute_like(
        const std::string&      name,
        const FaceAttribute<T>& other)
    {
        return add_face_attribute<T>(name,
                                     other.get_num_attributes(),
                                     other.get_allocated(),
                                     other.get_layout());
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
        layoutT               layout = SoA);

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
        layoutT            layout   = SoA);

    /**
     * @brief Adding a new edge attribute similar to another edge attribute
     * in allocation, number of attributes, and layout
     * @tparam T type of the returned attribute
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param other the other edge attribute
     * @return shared pointer to the created edge attribute
     */
    template <class T>
    std::shared_ptr<EdgeAttribute<T>> add_edge_attribute_like(
        const std::string&      name,
        const EdgeAttribute<T>& other);

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
        layoutT            layout   = SoA);

    /**
     * @brief Adding a new vertex attribute similar to another vertex attribute
     * in allocation, number of attributes, and layout
     * @tparam T type of the returned attribute
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param other the other vertex attribute
     * @return shared pointer to the created vertex attribute
     */
    template <class T>
    std::shared_ptr<VertexAttribute<T>> add_vertex_attribute_like(
        const std::string&        name,
        const VertexAttribute<T>& other);

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
        layoutT                            layout = SoA);

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
        layoutT               layout = SoA);

    /**
     * @brief similar to add_vertex/edge/face_attribute where the mesh element
     * type is defined via template parameter
     * @return
     */
    template <class T, class HandleT>
    std::shared_ptr<Attribute<T, HandleT>> add_attribute(
        const std::string& name,
        uint32_t           num_attributes,
        locationT          location = LOCATION_ALL,
        layoutT            layout   = SoA);

    /**
     * @brief Adding a new attribute similar to another attribute in allocation,
     * number of attributes, and layout. The type of the attribute (vertex,edge,
     * or face) is derived automatically from the input attribute (other)
     * @tparam T type of the returned attribute
     * @tparam HandleT handle type of the returned attribute
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param other the other attribute
     * @return shared pointer to the created attribute
     */
    template <class T, class HandleT>
    std::shared_ptr<Attribute<T, HandleT>> add_attribute_like(
        const std::string&           name,
        const Attribute<T, HandleT>& other);

    /**
     * @brief Checks if an attribute exists given its name
     * @param name the attribute name
     * @return True if the attribute exists. False otherwise.
     */
    bool does_attribute_exist(const std::string& name);

    /**
     * @brief Remove an attribute. Could be vertex, edge, or face attribute
     * @param name the attribute name
     */
    void remove_attribute(const std::string& name);

    /**
     * @brief populate boundary_v with 1 if the vertex is a boundary vertex and
     * 0 otherwise. Only the first attribute (i.e., boundary_v(vh, 0)) will be
     * populated. Possible types of T is bool or int (and maybe float). The
     * results will be first calculated on device and then move to the host is
     * boundary_v is allocated on the host.
     */
    template <typename T>
    void get_boundary_vertices(VertexAttribute<T>& boundary_v,
                               bool                move_to_host = true,
                               cudaStream_t        stream       = NULL) const;

    /**
     * @brief return a shared pointer the input vertex position
     */
    std::shared_ptr<VertexAttribute<float>> get_input_vertex_coordinates();

    /**
     * @brief return the number of regions (labels) in the mesh.
     */
    int get_num_regions() const;

    /**
     * @brief return a shared pointer of the face region label
     */
    std::shared_ptr<FaceAttribute<int>> get_face_region_label();

    /**
     * @brief return a shared pointer of the edge region label
     */
    std::shared_ptr<EdgeAttribute<int>> get_edge_region_label();

    /**
     * @brief return a shared pointer of the vertex region label
     */
    std::shared_ptr<VertexAttribute<int>> get_vertex_region_label();

    /**
     * @brief return a shared pointer of region label based on the template type
     */
    template <typename HandleT>
    std::shared_ptr<Attribute<int, HandleT>> get_region_label()
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

    /**
     * @brief scale the mesh so that it fits inside a bounding box defined by
     * the box lower and upper. Results are reflected on the coordinates
     * returned by get_input_vertex_coordinates()
     * @param lower bounding box lower corner
     * @param upper bounding box upper corner
     */
    void scale(glm::fvec3 lower, glm::fvec3 upper);

    /**
     * @brief compute the mesh bounding box using coordinates returned by
     * get_input_vertex_coordinates()
     * @param lower
     * @param upper
     */
    void bounding_box(glm::vec3& lower, glm::vec3& upper);

    /**
     * @brief Map a vertex handle into a global index as seen in the input
     * to RXMeshStatic
     * @param vh input vertex handle
     * @return the global index of vh
     */
    uint32_t map_to_global(const VertexHandle vh) const;

    /**
     * @brief Map an edge handle into a global index
     * @param eh input edge handle
     * @return the global index of eh
     */
    uint32_t map_to_global(const EdgeHandle eh) const;

    /**
     * @brief Map a face handle into a global index as seen in the input
     * to RXMeshStatic
     * @param vh input face handle
     * @return the global index of fh
     */
    uint32_t map_to_global(const FaceHandle fh) const;

    /**
     * @brief compute a linear compact index for a give vertex/edge/face handle
     * @tparam HandleT the type of the input handle
     * @param input handle
     */
    template <typename HandleT>
    uint32_t linear_id(HandleT input) const;

    /**
     * @brief get the owner handle of a given mesh element handle
     * @param handle the mesh element handle
     * memory
     */
    template <typename HandleT>
    HandleT get_owner_handle(const HandleT input) const;

    /**
     * @brief Export the mesh to obj file
     * @tparam T type of vertices coordinates
     * @param filename the output file
     * @param coords vertices coordinates
     */
    template <typename T>
    void export_obj(const std::string&        filename,
                    const VertexAttribute<T>& coords) const;

    /**
     * @brief export the mesh to a VTK file which can be visualized using
     * Paraview. The VTK supports visualizing attributes on vertices and faces.
     * Edge attributes are NOT supported. This function uses parameter pack such
     * that the user can call it with zero, one or move attributes (again should
     * be either VertexAttribute or FaceAttribute).
     */
    template <typename T, typename... AttributesT>
    void export_vtk(const std::string&        filename,
                    const VertexAttribute<T>& coords,
                    AttributesT... attributes) const
    {
        std::string  fn = filename;
        std::fstream file(fn, std::ios::out);
        file.precision(30);

        file << "# vtk DataFile Version 3.0\n";
        file << extract_file_name(filename) << "\n";
        file << "ASCII\n";
        file << "DATASET POLYDATA\n";
        file << "POINTS " << get_num_vertices() << " float\n ";

        std::vector<glm::vec3> v_list;
        create_vertex_list(v_list, coords);

        assert(get_num_vertices() == v_list.size());

        for (uint32_t v = 0; v < v_list.size(); ++v) {
            file << v_list[v][0] << " " << v_list[v][1] << " " << v_list[v][2]
                 << " \n";
        }

        std::vector<glm::uvec3> f_list;
        create_face_list(f_list);

        assert(f_list.size() == get_num_faces());

        file << "POLYGONS 3 " << 4 * f_list.size() << "\n";

        for (uint32_t f = 0; f < f_list.size(); ++f) {
            file << "3 ";
            for (uint32_t i = 0; i < 3; ++i) {
                file << f_list[f][i] << " ";
            }
            file << "\n";
        }
        bool first_v_attr = true;
        bool first_f_attr = true;


        ([&] { export_vtk(file, first_v_attr, first_f_attr, attributes); }(),
         ...);

        file.close();
    }

    /**
     * @brief convert given vertex attributes representing the coordinates into
     * std vector
     */
    template <typename T>
    void create_vertex_list(std::vector<glm::vec3>&   v_list,
                            const VertexAttribute<T>& coords) const;

    /**
     * @brief convert the mesh connectivity to face list
     */
    void create_face_list(std::vector<glm::uvec3>& f_list) const;

   protected:
    template <typename AttributeT>
    void export_vtk(std::fstream&     file,
                    bool&             first_v_attr,
                    bool&             first_f_attr,
                    const AttributeT& attribute) const
    {
        using HandleT = typename AttributeT::HandleType;

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            if (first_f_attr) {
                file << "CELL_DATA " << get_num_faces() << "\n";
                first_f_attr = false;
            }
            uint32_t num_attr = attribute.get_num_attributes();
            if (num_attr == 1) {
                file << "SCALARS " << attribute.get_name() << " float 1\n";
                file << "LOOKUP_TABLE default\n";
            } else if (num_attr == 2) {
                file << "COLOR_SCALARS " << attribute.get_name() << " 2\n";
            } else if (num_attr == 3) {
                file << "VECTORS " << attribute.get_name() << " float \n";
            } else {
                RXMESH_ERROR(
                    "RXMeshStatic::export_vtk() The number of attributes ({}) "
                    "is not support. Only 1, 2, or 3 attributes are supported",
                    num_attr);
                return;
            }

            for_each_face(
                HOST,
                [&](const FaceHandle& fh) {
                    for (uint32_t i = 0; i < attribute.get_num_attributes();
                         ++i) {
                        file << attribute(fh, i) << " ";
                    }
                    file << "\n";
                },
                NULL,
                false);
        }


        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            if (first_v_attr) {
                file << "POINT_DATA " << get_num_vertices() << "\n";
                first_v_attr = false;
            }
            uint32_t num_attr = attribute.get_num_attributes();
            if (num_attr == 1) {
                file << "SCALARS " << attribute.get_name() << " float 1\n";
                file << "LOOKUP_TABLE default\n";
            } else if (num_attr == 2) {
                file << "COLOR_SCALARS " << attribute.get_name() << " 2\n";
            } else if (num_attr == 3) {
                file << "VECTORS " << attribute.get_name() << " float \n";
            } else {
                RXMESH_ERROR(
                    "RXMeshStatic::export_vtk() The number of attributes ({}) "
                    "is not support. Only 1, 2, or 3 attributes are supported",
                    num_attr);
                return;
            }

            for_each_vertex(
                HOST,
                [&](const VertexHandle& vh) {
                    for (uint32_t i = 0; i < attribute.get_num_attributes();
                         ++i) {
                        file << attribute(vh, i) << " ";
                    }
                    file << "\n";
                },
                NULL,
                false);
        }
    }

    template <uint32_t blockThreads>
    size_t calc_shared_memory(const Op   op,
                              const bool oriented,
                              bool       use_capacity) const
    {
        uint32_t max_v(this->m_max_vertices_per_patch),
            max_e(this->m_max_edges_per_patch),
            max_f(this->m_max_faces_per_patch);

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

        if ((op == Op::EVDiamond || op == Op::EE) &&
            !m_is_input_edge_manifold) {
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
            uint32_t ef_smem = (std::max(max_e + 1, 3 * max_f) + (3 * max_f)) *
                               sizeof(uint16_t);


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

    void check_shared_memory(const uint32_t smem_bytes_dyn,
                             size_t&        smem_bytes_static,
                             uint32_t&      num_reg_per_thread,
                             size_t&        local_mem_per_thread,
                             const uint32_t num_threads_per_block,
                             const void*    kernel,
                             bool           print = true) const
    {
        // TODO this has to be customized for different GPU arch
        // int max_shmem_bytes = 89 * 1024;
        // CUDA_ERROR(
        //    cudaFuncSetAttribute(kernel,
        //                         cudaFuncAttributeMaxDynamicSharedMemorySize,
        //                         max_shmem_bytes));

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


        int num_blocks_per_sm = 0;
        CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads_per_block, smem_bytes_dyn));

        local_mem_per_thread = func_attr.localSizeBytes;

        if (print) {
            RXMESH_TRACE(
                "RXMeshStatic::check_shared_memory() user function requires "
                "shared memory = {} (dynamic) + {} (static) = {} (bytes) and "
                "{} registers/thread with occupancy of {} blocks/SM, {} local "
                "mem/thread (bytes)",
                smem_bytes_dyn,
                smem_bytes_static,
                smem_bytes_dyn + smem_bytes_static,
                num_reg_per_thread,
                num_blocks_per_sm,
                local_mem_per_thread);

            RXMESH_TRACE(
                "RXMeshStatic::check_shared_memory() max dynamic shared "
                "memory per block for this function = {} (bytes) = {} "
                "(Kb)",
                func_attr.maxDynamicSharedSizeBytes,
                float(func_attr.maxDynamicSharedSizeBytes) / 1024.0f);

            RXMESH_TRACE(
                "RXMeshStatic::check_shared_memory() max total shared "
                "memory per block for the current device = {} (bytes) = {} "
                "(Kb)",
                devProp.sharedMemPerBlockOptin,
                float(devProp.sharedMemPerBlockOptin) / 1024.0f);
        }

        if (int(smem_bytes_dyn) > func_attr.maxDynamicSharedSizeBytes) {
            RXMESH_ERROR(
                " RXMeshStatic::check_shared_memory() dynamic shared memory "
                "needed for input function ({} bytes) exceeds the max dynamic "
                "shared memory per block for this function ({} bytes)",
                smem_bytes_dyn,
                func_attr.maxDynamicSharedSizeBytes);
            // exit(EXIT_FAILURE);
        }


        if (smem_bytes_static + smem_bytes_dyn >
            devProp.sharedMemPerBlockOptin) {
            RXMESH_ERROR(
                " RXMeshStatic::check_shared_memory() total shared memory "
                "needed for input function ({} bytes) exceeds the max total "
                "shared memory per block (opt-in) on the current device ({} "
                "bytes)",
                smem_bytes_static + smem_bytes_dyn,
                devProp.sharedMemPerBlockOptin);
            // exit(EXIT_FAILURE);
        }

        if (num_blocks_per_sm == 0) {
            RXMESH_ERROR(
                "RXMeshStatic::check_shared_memory() This kernel will not run "
                "since it asks for too many resources i.e., shared memory "
                "and/or registers. If you are in Debug mode, try to switch to "
                "Release. Otherwise, you may try to change the block size "
                "and/or break the kernel into such that the number of "
                "registers is less. You may also try reducing the amount of "
                "additional shared memory you requested");
            // exit(EXIT_FAILURE);
        }
    }

#if USE_POLYSCOPE
    void add_patch_to_polyscope(const uint32_t                        p,
                                std::vector<std::array<uint32_t, 3>>& fv,
                                bool with_ribbon);


    void update_polyscope_edge_map();

    void register_polyscope();

    std::string             m_polyscope_mesh_name;
    polyscope::SurfaceMesh* m_polyscope_mesh;
    EdgeMapT                m_polyscope_edges_map;
#endif

   public:
    void add_edge_labels(FaceAttribute<int>& face_label,
                         EdgeAttribute<int>& edge_label);


   protected:
    std::shared_ptr<AttributeContainer>     m_attr_container;
    std::shared_ptr<VertexAttribute<float>> m_input_vertex_coordinates;

    std::shared_ptr<FaceAttribute<int>>   m_face_label;
    std::shared_ptr<EdgeAttribute<int>>   m_edge_label;
    std::shared_ptr<VertexAttribute<int>> m_vertex_label;
    int                                   m_num_regions;
};


}  // namespace rxmesh

#include "rxmesh/rxmesh_static.inl"