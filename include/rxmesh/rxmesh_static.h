#pragma once
#include <assert.h>
#include <fstream>
#include <functional>
#include <memory>

#include <cuda_profiler_api.h>

#include "rxmesh/attribute.h"
#include "rxmesh/diff/diff_attribute.h"
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
                          const float       lp_hashtable_load_factor = 0.8)
        : RXMesh(patch_size)
    {
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
    };

    /**
     * @brief Constructor using triangles and vertices
     * @param fv Face incident vertices as read from an obj file
     */
    explicit RXMeshStatic(std::vector<std::vector<uint32_t>>& fv,
                          const std::string                   patcher_file = "",
                          const uint32_t                      patch_size = 512,
                          const float capacity_factor                    = 1.0,
                          const float patch_alloc_factor                 = 1.0,
                          const float lp_hashtable_load_factor           = 0.8)
        : RXMesh(patch_size), m_input_vertex_coordinates(nullptr)
    {
        this->init(fv,
                   patcher_file,
                   capacity_factor,
                   patch_alloc_factor,
                   lp_hashtable_load_factor);
        m_attr_container = std::make_shared<AttributeContainer>();
    };

    /**
     * @brief Add vertex coordinates to the input mesh. When calling
     * RXMeshStatic constructor that takes the face's vertices, this function
     * can be called to then add vertex coordinates and also add the mesh to
     * polyscope if it is active. You don't need to call this function if you
     * are constructing RXMeshStatic with the constructor that takes the path to
     * mesh file
     */
    void add_vertex_coordinates(std::vector<std::vector<float>>& vertices,
                                std::string                      mesh_name = "")
    {
        if (m_input_vertex_coordinates == nullptr) {

            m_input_vertex_coordinates =
                this->add_vertex_attribute<float>(vertices, "rx:vertices");

#if USE_POLYSCOPE
            // polyscope::options::autocenterStructures = true;
            // polyscope::options::autoscaleStructures  = true;
            // polyscope::options::automaticallyComputeSceneExtents = true;
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
                                         bool with_face_patch   = true)
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

    /**
     * @brief add the face's patch and local ID scalar quantities to a polyscope
     * instance (polyscope_mesh) for specific patch. polyscope_mesh should be
     * the one returned from render_patch call with the same input patch (p)
     * @param p patch id for which the face patch will be added
     * @param polyscope_mesh the SurfaceMesh pointer returned by calling
     * render_patch with the same input patch
     */
    void render_face_patch_and_local_id(const uint32_t          p,
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
            0.0,
            double(*std::max_element(local_id.begin(), local_id.end()) - 1));
        polyscope_mesh->addFaceScalarQuantity(l_name, local_id)
            ->setMapRange(l_range);
    }

    /**
     * @brief add the edge's patch and local ID scalar quantities to a polyscope
     * instance (polyscope_mesh) for specific patch. polyscope_mesh should be
     * the one returned from render_patch call with the same input patch (p)
     * @param p patch id for which the face patch will be added
     * @param polyscope_mesh the SurfaceMesh pointer returned by calling
     * render_patch with the same input patch
     */
    void render_edge_patch_and_local_id(const uint32_t          p,
                                        polyscope::SurfaceMesh* polyscope_mesh)
    {
        std::string p_name = "rx:EPatch" + std::to_string(p);
        std::string l_name = "rx:ELocal" + std::to_string(p);
        // unlike render_face_patch and  where the size of this
        // std::vector is the size of number faces in patch, here we
        // use the total number of edges since we pass to polyscope the edge map
        // for the whole mesh (not just for this patch) (see render_patch) and
        // thus it expects the size of this quantity to be the same size i.e.,
        // total number of edges
        std::vector<int> patch_id(get_num_edges(), -(p + 1));
        std::vector<int> local_id(get_num_edges(), -(p + 1));

        for_each_edge(HOST, [&](EdgeHandle eh) {
            patch_id[linear_id(eh)] = eh.patch_id();
            local_id[linear_id(eh)] = eh.local_id();
        });

        std::pair<double, double> p_range(0.0, double(get_num_patches() - 1));
        polyscope_mesh->addEdgeScalarQuantity(p_name, patch_id)
            ->setMapRange(p_range);

        std::pair<double, double> l_range(
            0.0,
            double(*std::max_element(local_id.begin(), local_id.end()) - 1));
        polyscope_mesh->addEdgeScalarQuantity(l_name, local_id)
            ->setMapRange(l_range);
    }

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
        polyscope::SurfaceMesh* polyscope_mesh)
    {
        std::string p_name = "rx:VPatch" + std::to_string(p);
        std::string l_name = "rx:VLocal" + std::to_string(p);
        // unlike render_face_patch and  where the size of this
        // std::vector is the size of number faces in patch, here we
        // use the total number of vertices since we pass to polyscope the
        // vertex position for the whole mesh (not just for this patch) (see
        // render_patch) and thus it expects the size of this quantity to be the
        // same size i.e., total number of vertices
        std::vector<int> patch_id(get_num_vertices(), -(p + 1));
        std::vector<int> local_id(get_num_vertices(), -(p + 1));

        for_each_vertex(HOST, [&](VertexHandle vh) {
            patch_id[linear_id(vh)] = vh.patch_id();
            local_id[linear_id(vh)] = vh.local_id();
        });

        std::pair<double, double> p_range(0.0, double(get_num_patches() - 1));
        polyscope_mesh->addVertexScalarQuantity(p_name, patch_id)
            ->setMapRange(p_range);

        std::pair<double, double> l_range(
            0.0,
            double(*std::max_element(local_id.begin(), local_id.end()) - 1));
        polyscope_mesh->addVertexScalarQuantity(l_name, local_id)
            ->setMapRange(l_range);
    }


    /**
     * @brief add the face's patch scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's face scalar quantity
     */
    polyscope::SurfaceFaceScalarQuantity* render_face_patch()
    {
        std::string name = "rx:FPatch";
        auto face_patch  = this->add_face_attribute<uint32_t>(name, 1, HOST);
        for_each_face(
            HOST, [&](FaceHandle fh) { (*face_patch)(fh) = fh.patch_id(); });
        auto ret = m_polyscope_mesh->addFaceScalarQuantity(name, *face_patch);
        remove_attribute(name);

        std::pair<double, double> range(0.0, double(get_num_patches() - 1));
        ret->setMapRange(range);

        return ret;
    }

    /**
     * @brief add the edge's patch scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's edge scalar quantity
     */
    polyscope::SurfaceEdgeScalarQuantity* render_edge_patch()
    {
        std::string name = "rx:EPatch";
        auto edge_patch  = this->add_edge_attribute<uint32_t>(name, 1, HOST);
        for_each_edge(
            HOST, [&](EdgeHandle eh) { (*edge_patch)(eh) = eh.patch_id(); });
        auto ret = m_polyscope_mesh->addEdgeScalarQuantity(name, *edge_patch);
        remove_attribute(name);

        std::pair<double, double> range(0.0, double(get_num_patches() - 1));
        ret->setMapRange(range);

        return ret;
    }


    /**
     * @brief add the vertex's patch scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's vertex scalar quantity
     */
    polyscope::SurfaceVertexScalarQuantity* render_vertex_patch()
    {
        std::string name  = "rx:VPatch";
        auto vertex_patch = this->add_vertex_attribute<uint32_t>(name, 1, HOST);
        for_each_vertex(HOST, [&](VertexHandle vh) {
            (*vertex_patch)(vh) = vh.patch_id();
        });
        auto ret =
            m_polyscope_mesh->addVertexScalarQuantity(name, *vertex_patch);
        remove_attribute(name);
        return ret;
    }


    /**
     * @brief add the face's local ID scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's face scalar quantity
     */
    polyscope::SurfaceFaceScalarQuantity* render_face_local_id()
    {
        std::string name = "rx:FLocal";

        auto f_local = add_face_attribute<uint16_t>(name, 1);

        for_each_face(
            HOST, [&](const FaceHandle fh) { (*f_local)(fh) = fh.local_id(); });

        auto ret = m_polyscope_mesh->addFaceScalarQuantity(name, *f_local);

        remove_attribute(name);

        return ret;
    }


    /**
     * @brief add the edge's local ID scalar quantity to the polyscope instance
     * associated RXMeshStatic
     * @return pointer to polyscope's edge scalar quantity
     */
    polyscope::SurfaceEdgeScalarQuantity* render_edge_local_id()
    {
        std::string name = "rx:ELocal";

        auto e_local = add_edge_attribute<uint16_t>(name, 1);

        for_each_edge(
            HOST, [&](const EdgeHandle eh) { (*e_local)(eh) = eh.local_id(); });

        auto ret = m_polyscope_mesh->addEdgeScalarQuantity(name, *e_local);

        remove_attribute(name);

        return ret;
    }

    /**
     * @brief add the vertex's local ID scalar quantity to the polyscope
     * instance associated RXMeshStatic
     * @return pointer to polyscope's vertex scalar quantity
     */
    polyscope::SurfaceVertexScalarQuantity* render_vertex_local_id()
    {
        std::string name = "rx:VLocal";

        auto v_local = add_vertex_attribute<uint16_t>(name, 1);

        for_each_vertex(HOST, [&](const VertexHandle vh) {
            (*v_local)(vh) = vh.local_id();
        });

        auto ret = m_polyscope_mesh->addVertexScalarQuantity(name, *v_local);

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
     * @tparam ...ArgsT infered
     * @tparam blockThreads the block size
     * @param lb launch box populated via prepare_launch_box
     * @param kernel the kernel to launch
     * @param stream to launch the kerenl on
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
    void run_kernel(const std::vector<Op> op, KernelT kernel, ArgsT... args)
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
                          cudaStream_t  stream   = NULL)
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
                          cudaStream_t            stream   = NULL)
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
            size_t sh =
                this->template calc_shared_memory<blockThreads>(o, oriented);
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
     * @brief Adding a new differentiable face attribute
     * @tparam T the underlying type of the attribute
     * @tparam Size the number of components per face
     * @tparam WithHessian if hessian is required
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param location where to allocate the attributes
     */
    template <class T, int Size, bool WithHessian>
    std::shared_ptr<DiffFaceAttribute<T, Size, WithHessian>>
    add_diff_face_attribute(const std::string& name,
                            uint32_t           num_attributes = 1,
                            locationT          location       = LOCATION_ALL,
                            layoutT            layout         = SoA)
    {
        return m_attr_container
            ->template add<DiffFaceAttribute<T, Size, WithHessian>>(
                name.c_str(), num_attributes, location, layout, this);
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
        const EdgeAttribute<T>& other)
    {
        return add_edge_attribute<T>(name,
                                     other.get_num_attributes(),
                                     other.get_allocated(),
                                     other.get_layout());
    }

    /**
     * @brief Adding a new differentiable edge attribute
     * @tparam T the underlying type of the attribute
     * @tparam Size the number of components per edge
     * @tparam WithHessian if hessian is required
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param location where to allocate the attributes
     */
    template <class T, int Size, bool WithHessian>
    std::shared_ptr<DiffEdgeAttribute<T, Size, WithHessian>>
    add_diff_edge_attribute(const std::string& name,
                            uint32_t           num_attributes = 1,
                            locationT          location       = LOCATION_ALL,
                            layoutT            layout         = SoA)
    {
        return m_attr_container
            ->template add<DiffEdgeAttribute<T, Size, WithHessian>>(
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
        const VertexAttribute<T>& other)
    {
        return add_vertex_attribute<T>(name,
                                       other.get_num_attributes(),
                                       other.get_allocated(),
                                       other.get_layout());
    }

    /**
     * @brief Adding a new differentiable vertex attribute
     * @tparam T the underlying type of the attribute
     * @tparam Size the number of components per vertex, e.g., 3 for vertex
     * coordinates
     * @tparam WithHessian if hessian is required
     * @param name of the attribute. Should not collide with other attributes
     * names
     * @param location where to allocate the attributes
     */
    template <class T, int Size, bool WithHessian>
    std::shared_ptr<DiffVertexAttribute<T, Size, WithHessian>>
    add_diff_vertex_attribute(const std::string& name,
                              uint32_t           num_attributes = 1,
                              locationT          location       = LOCATION_ALL,
                              layoutT            layout         = SoA)
    {
        return m_attr_container
            ->template add<DiffVertexAttribute<T, Size, WithHessian>>(
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
     * @brief similar to add_vertex/edge/face_attribute where the mesh element
     * type is defined via template parameter
     * @return
     */
    template <class T, class HandleT>
    std::shared_ptr<Attribute<T, HandleT>> add_attribute(
        const std::string& name,
        uint32_t           num_attributes,
        locationT          location = LOCATION_ALL,
        layoutT            layout   = SoA)
    {
        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            return add_vertex_attribute<T>(
                name, num_attributes, location, layout);
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            return add_edge_attribute<T>(
                name, num_attributes, location, layout);
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            return add_face_attribute<T>(
                name, num_attributes, location, layout);
        }
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
     * @brief populate boundary_v with 1 if the vertex is a boundary vertex and
     * 0 otherwise. Only the first attribute (i.e., boundary_v(vh, 0)) will be
     * populated. Possible types of T is bool or int (and maybe float). The
     * results will be first calculated on device and then move to the host is
     * boundary_v is allocated on the host.
     */
    template <typename T>
    void get_boundary_vertices(VertexAttribute<T>& boundary_v,
                               bool                move_to_host = true,
                               cudaStream_t        stream       = NULL) const
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
     * @brief scale the mesh so that it fits inside a bounding box defined by
     * the box lower and upper. Results are reflected on the coordinates
     * returned by get_input_vertex_coordinates()
     * @param lower bounding box lower corner
     * @param upper bounding box upper corner
     */
    void scale(glm::fvec3 lower, glm::fvec3 upper)
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
    }

    /**
     * @brief compute the mesh bounding box using coordinates returned by
     * get_input_vertex_coordinates()
     * @param lower
     * @param upper
     */
    void bounding_box(glm::vec3& lower, glm::vec3& upper)
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
    uint32_t linear_id(HandleT input) const
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

    /**
     * @brief get the owner handle of a given mesh element handle
     * @param handle the mesh element handle
     * memory
     */
    template <typename HandleT>
    HandleT get_owner_handle(const HandleT input) const
    {
        return get_context().get_owner_handle(input, m_h_patches_info);
    }

    /**
     * @brief Export the mesh to obj file
     * @tparam T type of vertices coordinates
     * @param filename the output file
     * @param coords vertices coordinates
     */
    template <typename T>
    void export_obj(const std::string&        filename,
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

    /**
     * @brief convert the mesh connectivity to face list
     */
    void create_face_list(std::vector<glm::uvec3>& f_list) const
    {
        f_list.reserve(get_num_faces());

        for (uint32_t p = 0; p < this->m_num_patches; ++p) {
            const uint32_t p_num_faces = this->m_h_patches_info[p].num_faces[0];
            for (uint32_t f = 0; f < p_num_faces; ++f) {
                if (!detail::is_deleted(
                        f, this->m_h_patches_info[p].active_mask_f) &&
                    detail::is_owned(f,
                                     this->m_h_patches_info[p].owned_mask_f)) {

                    glm::uvec3 face;

                    for (uint32_t e = 0; e < 3; ++e) {
                        uint16_t edge =
                            this->m_h_patches_info[p].fe[3 * f + e].id;
                        flag_t dir(0);
                        Context::unpack_edge_dir(edge, edge, dir);
                        uint16_t     e_id = (2 * edge) + dir;
                        uint16_t     v = this->m_h_patches_info[p].ev[e_id].id;
                        VertexHandle vh(p, v);
                        uint32_t     vid = linear_id(vh);
                        face[e]          = vid;
                    }
                    f_list.push_back(face);
                }
            }
        }
    }

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
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t);

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
            dynamic_smem = 2 * this->m_max_edges_per_patch * sizeof(uint16_t);

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
            dynamic_smem += 3 * this->m_max_faces_per_patch * sizeof(uint16_t);
            dynamic_smem += 2 * this->m_max_edges_per_patch * sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size<LocalFaceT>();

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size<LocalVertexT>();

            //  stores vertex LP hashtable
            uint32_t table_bytes =
                sizeof(LPPair) * max_lp_hashtable_capacity<LocalVertexT>();
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
            // Normally, the number of vertices is way less than 2*#E but in
            // dynamic mesh, we can not predicate these numbers since some
            // of these edges could be deleted (marked deleted in the bitmask)
            // so, we allocate the buffer that hold EV (which will also hold the
            // offset) to be the max of #V and 2#E
            dynamic_smem = std::max(this->m_max_vertices_per_patch,
                                    2 * this->m_max_edges_per_patch) *
                           sizeof(uint16_t);
            dynamic_smem +=
                (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size<LocalVertexT>();

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size<LocalEdgeT>();

            // stores edge LP hashtable
            uint32_t lp_smem =
                sizeof(LPPair) * max_lp_hashtable_capacity<LocalEdgeT>();


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
            dynamic_smem += max_bitmask_size<LocalEdgeT>();

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size<LocalFaceT>();

            // stores the face LP hashtable
            dynamic_smem +=
                sizeof(LPPair) * max_lp_hashtable_capacity<LocalFaceT>();

            // for possible padding for alignment
            // 4 since there are 4 calls for ShmemAllocator.alloc
            dynamic_smem += ShmemAllocator::default_alignment * 4;

        } else if (op == Op::VF) {
            // load EV and FE simultaneously. Changes FE to FV using EV. Then
            // transpose FV in place and use EV to store the values/output while
            // using FV to store the prefix sum. Thus, the space used to store
            // EV should be max(3*#faces, 2*#edges)
            dynamic_smem = std::max(3 * this->m_max_faces_per_patch,
                                    1 + this->m_max_vertices_per_patch) *
                           sizeof(uint16_t);
            dynamic_smem += std::max(3 * this->m_max_faces_per_patch,
                                     2 * this->m_max_edges_per_patch) *
                                sizeof(uint16_t) +
                            sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size<LocalVertexT>();

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size<LocalFaceT>();

            // stores the face LP hashtable
            dynamic_smem +=
                sizeof(LPPair) * max_lp_hashtable_capacity<LocalFaceT>();

            // for possible padding for alignment
            // 5 since there are 5 calls for ShmemAllocator.alloc
            dynamic_smem += ShmemAllocator::default_alignment * 5;

        } else if (op == Op::VV) {
            // similar to VE but we also need to store the EV even after
            // we do the transpose. After that, we can throw EV away and load
            // the hash table
            dynamic_smem = std::max(this->m_max_vertices_per_patch + 1,
                                    2 * this->m_max_edges_per_patch) *
                           sizeof(uint16_t);
            dynamic_smem +=
                (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

            // store participant bitmask
            dynamic_smem += max_bitmask_size<LocalVertexT>();

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size<LocalVertexT>();

            // duplicate EV
            uint32_t ev_smem =
                (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

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
            dynamic_smem += max_bitmask_size<LocalFaceT>();

            // store not-owned bitmask
            dynamic_smem += max_bitmask_size<LocalFaceT>();

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
            dynamic_smem = 4 * this->m_max_edges_per_patch * sizeof(uint16_t);

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
                             size_t&        local_mem_per_thread,
                             const uint32_t num_threads_per_block,
                             const void*    kernel,
                             bool           print = true) const
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
                    assert(v_org0 < get_num_vertices());
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

    void register_polyscope()
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

    std::string             m_polyscope_mesh_name;
    polyscope::SurfaceMesh* m_polyscope_mesh;
    EdgeMapT                m_polyscope_edges_map;
#endif

    std::shared_ptr<AttributeContainer>     m_attr_container;
    std::shared_ptr<VertexAttribute<float>> m_input_vertex_coordinates;
};
}  // namespace rxmesh