#pragma once

#include "rxmesh/diff/candidate_pairs.h"
#include "rxmesh/rxmesh_static.h"

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/traversal/shrinkingRadiusQuery.h"

using namespace rxmesh;

template <typename ProblemT,
          typename VAttrT,
          typename VAttrI,
          typename T = typename VAttrT::Type>
void floor_barrier_energy(ProblemT&      problem,
                          VAttrT&        contact_area,
                          const T        h,  // time_step
                          const VAttrI&  is_dbc,
                          const vec3<T>& ground_n,
                          const vec3<T>& ground_o,
                          const T        dhat,
                          const T        kappa)
{

    const T h_sq = h * h;

    const Eigen::Vector3<T> o(ground_o[0], ground_o[1], ground_o[2]);
    const Eigen::Vector3<T> n(ground_n[0], ground_n[1], ground_n[2]);

    const Eigen::Vector3<T> normal(0.0, -1.0, 0.0);

    problem.template add_term<Op::V, true>(
        [=] __device__(const auto& vh, auto& obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            const Eigen::Vector3<ActiveT> xi = iter_val<ActiveT, 3>(vh, obj);

            ActiveT E(T(0));


            if (!is_dbc(vh)) {
                // floor
                ActiveT d = (xi - o).dot(n);
                if (d < dhat) {
                    ActiveT s = d / dhat;

                    if (s <= T(0)) {
                        using PassiveT = PassiveType<ActiveT>;
                        return ActiveT(std::numeric_limits<PassiveT>::max());
                    }

                    E = h_sq * contact_area(vh) * dhat * T(0.5) * kappa *
                        (s - 1) * log(s);
                }
            }

            return E;
        });
}

template <typename ProblemT,
          typename VAttrT,
          typename VAttrB,
          typename PairT,
          typename T = typename VAttrT::Type>
void vv_contact(ProblemT&          problem,
                RXMeshStatic&      rx,
                PairT&             contact_pairs,
                const VertexHandle dbc_vertex,
                const VertexHandle dbc_vertex1,
                const VertexHandle dbc_vertex2,
                const VAttrB&      is_dbc,
                const VAttrT&      x,
                VAttrT&            contact_area,
                const T            h,
                const T            dhat,
                const T            kappa,
                const VertexAttribute<int>& region_label)
{
    contact_pairs.reset();

    // Step 1: Get vertex count and context
    uint32_t num_vertices = rx.get_num_elements<VertexHandle>();
    auto     ctx          = rx.get_context();

    // Step 2: Allocate and populate bounding boxes for BVH
    // For point data, each box is degenerate (min == max)
    using box_t = cuBQL::box_t<T, 3>;
    box_t* d_boxes = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_boxes, sizeof(box_t) * num_vertices));

    // Populate boxes from vertex positions
    const int threads = 256;
    const int blocks  = DIVIDE_UP(num_vertices, threads);

    for_each_item<<<blocks, threads>>>(
        num_vertices, [=] __device__(int i) mutable {
            VertexHandle      vh  = ctx.template get_handle<VertexHandle>(i);
            Eigen::Vector3<T> pos = x.template to_eigen<3>(vh);

            // Create bounding box at vertex position
            cuBQL::vec_t<T, 3> point;
            point.x = pos[0];
            point.y = pos[1];
            point.z = pos[2];

            d_boxes[i] = box_t().including(point);
        });

    CUDA_ERROR(cudaDeviceSynchronize());

    // Step 3: Build BVH
    cuBQL::BinaryBVH<T, 3> bvh;
    cuBQL::BuildConfig     build_config;
    cuBQL::gpuBuilder(bvh, d_boxes, num_vertices, build_config);

    // Calculate and print BVH memory usage
    size_t nodes_memory = bvh.numNodes * sizeof(typename cuBQL::BinaryBVH<T, 3>::Node);
    size_t primIDs_memory = bvh.numPrims * sizeof(uint32_t);
    size_t total_bvh_memory = nodes_memory + primIDs_memory;

    // Step 4: Query BVH for each vertex to find nearby vertices
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        if (is_dbc(vh)) {
            return;  // Skip DBC vertices
        }

        const Eigen::Vector3<T> xi = x.template to_eigen<3>(vh);
        const uint32_t          vh_id = ctx.template linear_id<VertexHandle>(vh);

        // Query point for BVH traversal
        cuBQL::vec_t<T, 3> query_point;
        query_point.x = xi[0];
        query_point.y = xi[1];
        query_point.z = xi[2];

        // Fixed-radius query using shrinking radius approach
        auto query_lambda = [&](uint32_t prim_id) -> float {
            if (prim_id == vh_id) {
                return dhat * dhat;  // Return SQUARED radius, skip self
            }

            VertexHandle other_vh = ctx.template get_handle<VertexHandle>(prim_id);

            // Only add contact pairs between vertices from different meshes
            int region_vh = region_label(vh);
            int region_other = region_label(other_vh);

            if (region_vh == region_other) {
                return dhat * dhat;  // Skip vertices from same mesh
            }

            Eigen::Vector3<T> xj = x.template to_eigen<3>(other_vh);

            T dist = (xi - xj).norm();

            if (dist < dhat) {
                contact_pairs.insert(vh, other_vh);
            }

            return dhat * dhat;  // Return SQUARED radius for fixed-radius query
        };

        cuBQL::shrinkingRadiusQuery::forEachPrim<T, 3>(
            query_lambda, bvh, query_point, dhat * dhat);
    });

    // printf("Traversed the BVH and executed the contact pair addition.\n");
    // printf("Contact Pairs Size After Traversal: %d\n", contact_pairs.num_pairs());

    // Step 5: Cleanup
    cuBQL::cuda::free(bvh);  // Free BVH memory (nodes and primIDs)
    GPU_FREE(d_boxes);  // Free bounding boxes
    // printf("Released BVH and bounding box memory.\n");
}

template <typename ProblemT,
          typename VAttrT,
          typename T = typename VAttrT::Type>
void vv_contact_energy(ProblemT&     problem,
                       const VAttrT& contact_area,
                       const T       h,
                       const T       dhat,
                       const T       kappa)
{
    const T h_sq = h * h;

    problem.template add_term<true>([=] __device__(const auto& id,
                                                   const auto& iter,
                                                   const auto& obj) mutable {
        using ActiveT = ACTIVE_TYPE(id);

        const VertexHandle v0 = iter[0];  // First vertex in contact pair
        const VertexHandle v1 = iter[1];  // Second vertex in contact pair

        // Get vertex positions
        const Eigen::Vector3<ActiveT> xi = iter_val<ActiveT, 3>(id, iter, obj, 0);
        const Eigen::Vector3<ActiveT> xj = iter_val<ActiveT, 3>(id, iter, obj, 1);

        // Compute distance between vertices
        ActiveT d = (xi - xj).norm();

        ActiveT E(T(0));

        if (d < dhat) {
            ActiveT s = d / dhat;

            if (s <= T(0)) {
                using PassiveT = PassiveType<ActiveT>;
                return ActiveT(std::numeric_limits<PassiveT>::max());
            }

            // Use average contact area of both vertices
            T avg_contact_area = (contact_area(v0) + contact_area(v1)) * T(0.5);

            // Barrier energy: E = h^2 * A * dhat * 0.5 * kappa * (s - 1) * log(s)
            // E = h_sq * avg_contact_area * dhat * T(0.5) * kappa * (s - 1) * log(s);
            E = avg_contact_area * dhat * T(0.5) * kappa * (s - 1) * log(s);
        }

        return E;
    });
}

template <uint32_t blockThreads, typename T>
__global__ static void build_triangle_boxes_kernel(
    const Context            context,
    const VertexAttribute<T> x,
    cuBQL::box_t<T, 3>*      d_boxes)
{
    using box_t = cuBQL::box_t<T, 3>;
    auto compute_box = [&](FaceHandle fh, VertexIterator& fv) {
        // Get the three vertices of the triangle
        // Don't use iter_val because this is bvh creation oepration.
        Eigen::Vector3<T> v0 = x.template to_eigen<3>(fv[0]);
        Eigen::Vector3<T> v1 = x.template to_eigen<3>(fv[1]);
        Eigen::Vector3<T> v2 = x.template to_eigen<3>(fv[2]);

        // Create bounding box that encompasses all three vertices
        cuBQL::vec_t<T, 3> p0{v0[0], v0[1], v0[2]};
        cuBQL::vec_t<T, 3> p1{v1[0], v1[1], v1[2]};
        cuBQL::vec_t<T, 3> p2{v2[0], v2[1], v2[2]};

        uint32_t face_id = context.linear_id(fh);
        d_boxes[face_id] = box_t().including(p0).including(p1).including(p2);
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, compute_box);
}

template <typename ProblemT,
          typename VAttrT,
          typename VAttrB,
          typename T = typename VAttrT::Type>
void vf_contact(ProblemT&     problem,
                RXMeshStatic& rx,
                const VAttrB& is_dbc,
                const VAttrT& x,
                VAttrT&       contact_area,
                const T       h,
                const T       dhat,
                const T       kappa)
{
    // Step 1: Get face count
    uint32_t num_faces = rx.get_num_elements<FaceHandle>();

    // Step 2: Allocate bounding boxes for BVH
    using box_t = cuBQL::box_t<T, 3>;
    box_t* d_boxes = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_boxes, sizeof(box_t) * num_faces));

    // Step 3: Populate boxes using Query::dispatch with FV
    constexpr uint32_t blockThreads = 256;
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::FV},
                          launch_box,
                          (void*)build_triangle_boxes_kernel<blockThreads, T>);

    build_triangle_boxes_kernel<blockThreads, T>
        <<<launch_box.blocks, launch_box.num_threads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), x, d_boxes);
    CUDA_ERROR(cudaDeviceSynchronize());

    // Step 4: Build BVH over triangles
    cuBQL::BinaryBVH<T, 3> bvh;
    cuBQL::BuildConfig     build_config;
    cuBQL::gpuBuilder(bvh, d_boxes, num_faces, build_config);

    // Calculate and print BVH memory usage
    size_t nodes_memory     = bvh.numNodes * sizeof(typename cuBQL::BinaryBVH<T, 3>::Node);
    size_t primIDs_memory   = bvh.numPrims * sizeof(uint32_t);
    size_t total_bvh_memory = nodes_memory + primIDs_memory;

    // printf("Built triangle BVH.\n");
    // printf("  Number of nodes: %u\n", bvh.numNodes);
    // printf("  Number of primitives: %u\n", bvh.numPrims);
    // printf("  Total BVH memory: %zu bytes (%.2f MB)\n",
    //        total_bvh_memory,
    //        total_bvh_memory / (1024.0f * 1024.0f));

    // Step 5: Cleanup
    cuBQL::cuda::free(bvh);
    GPU_FREE(d_boxes);
    // printf("Released triangle BVH and bounding box memory.\n");
}

template <typename ProblemT,
          typename VAttrT,
          typename VAttrB,
          typename PairT,
          typename T = typename VAttrT::Type>
void add_contact(ProblemT&          problem,
                 RXMeshStatic&      rx,
                 PairT&             contact_pairs,
                 const VertexHandle dbc_vertex,
                 const VertexHandle dbc_vertex1,
                 const VertexHandle dbc_vertex2,
                 const VAttrB&      is_dbc,
                 const VAttrT&      x,
                 VAttrT&            contact_area,
                 const T            h,
                 const T            dhat,
                 const T            kappa,
                 const VertexAttribute<int>& region_label)
{
    // Call VV contact handler
    vv_contact(problem,
               rx,
               contact_pairs,
               dbc_vertex,
               dbc_vertex1,
               dbc_vertex2,
               is_dbc,
               x,
               contact_area,
               h,
               dhat,
               kappa,
               region_label);

    // Call VF contact handler
    vf_contact(problem, rx, is_dbc, x, contact_area, h, dhat, kappa);
}

template <typename ProblemT,
          typename VAttrT,
          typename T = typename VAttrT::Type>
void ceiling_barrier_energy(ProblemT&      problem,
                            VAttrT&        contact_area,
                            const T        h,  // time_step
                            const vec3<T>& ground_n,
                            const vec3<T>& ground_o,
                            const T        dhat,
                            const T        kappa)
{
    const T h_sq = h * h;

    const Eigen::Vector3<T> normal(0.0, -1.0, 0.0);

    problem.template add_term<true>([=] __device__(const auto& id,
                                                   const auto& iter,
                                                   const auto& obj) mutable {
        using ActiveT = ACTIVE_TYPE(id);

        const VertexHandle c0 = iter[0];

        const VertexHandle c1 = iter[1];

        const Eigen::Vector3<ActiveT> xi =
            iter_val<ActiveT, 3>(id, iter, obj, 0);

        const Eigen::Vector3<T> x_dbc = obj.template to_eigen<3>(c1);


        // ceiling
        ActiveT d = (xi - x_dbc).dot(normal);

        ActiveT E(T(0));

        if (d < dhat) {
            ActiveT s = d / dhat;

            if (s <= T(0)) {
                using PassiveT = PassiveType<ActiveT>;
                return ActiveT(std::numeric_limits<PassiveT>::max());
            }

            E = h_sq * contact_area(c0) * dhat * T(0.5) * kappa * (s - 1) *
                log(s);
        }


        return E;
    });
}

template <typename VAttrT,
          typename VAttrI,
          typename DenseMatT,
          typename T = typename VAttrT::Type>
T barrier_step_size(RXMeshStatic&      rx,
                    const DenseMatT&   search_dir,
                    DenseMatT&         alpha,
                    const VertexHandle dbc_vertex,
                    const VAttrT&      x,
                    const VAttrI&      is_dbc,
                    const vec3<T>&     ground_n,
                    const vec3<T>&     ground_o)
{
    alpha.reset(T(1), DEVICE);

    const vec3<T> n(0.0, -1.0, 0.0);

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        const vec3<T> p_dbc(search_dir(dbc_vertex, 0),
                            search_dir(dbc_vertex, 1),
                            search_dir(dbc_vertex, 2));

        const vec3<T> x_dbc = x.to_glm<3>(dbc_vertex);


        const vec3<T> pi(
            search_dir(vh, 0), search_dir(vh, 1), search_dir(vh, 2));

        const vec3<T> xi = x.to_glm<3>(vh);

        // floor
        T p_n = glm::dot(pi, ground_n);
        if (p_n < 0) {
            alpha(vh) = std::min(
                alpha(vh), T(0.9) * glm::dot(ground_n, (xi - ground_o)) / -p_n);
        }

        // ceiling
        // TODO this should be generalized
        if (!is_dbc(vh)) {
            p_n = glm::dot(n, (pi - p_dbc));
            if (p_n < 0) {
                alpha(vh) = std::min(alpha(vh),
                                     T(0.9) * glm::dot(n, (xi - x_dbc)) / -p_n);
            }
        }
    });

    // we want the min here but since the min value is greater than 1 (y_ground
    // is less than 0, and search_dir is also less than zero)
    return alpha.abs_min();
}
