#pragma once

#include "rxmesh/diff/candidate_pairs.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/timer.h"

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/traversal/shrinkingRadiusQuery.h"

using namespace rxmesh;

/**
 * Pre-allocated GPU buffer for BVH bounding boxes.
 * Allocated once and reused across iterations to avoid repeated malloc/free.
 */
template <typename T>
struct BVHBuffers {
    using box_t = cuBQL::box_t<T, 3>;

    box_t* d_boxes;
    size_t capacity;
    bool initialized;

    BVHBuffers() : d_boxes(nullptr), capacity(0), initialized(false) {}

    BVHBuffers(size_t max_vertices)
        : capacity(max_vertices), initialized(false) {
        allocate(max_vertices);
    }

    void allocate(size_t max_vertices) {
        if (initialized) {
            return;  // Already allocated
        }
        capacity = max_vertices;
        CUDA_ERROR(cudaMalloc((void**)&d_boxes, sizeof(box_t) * capacity));
        initialized = true;
    }

    ~BVHBuffers() {
        if (initialized && d_boxes) {
            GPU_FREE(d_boxes);
            d_boxes = nullptr;
            initialized = false;
        }
    }
};

template <typename ProblemT,
          typename VAttrT,
          typename T = typename VAttrT::Type>
void floor_barrier_energy(ProblemT&      problem,
                          VAttrT&        contact_area,
                          const T        h,  // time_step
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

            return E;
        });
}

template <typename ProblemT,
          typename VAttrT,
          typename PairT,
          typename BVHBufferT,
          typename T = typename VAttrT::Type>
void vv_contact(ProblemT&          problem,
                RXMeshStatic&      rx,
                PairT&             contact_pairs,
                BVHBufferT&        bvh_buffers,
                const VAttrT&      x,
                VAttrT&            contact_area,
                const T            h,
                const T            dhat,
                const T            kappa,
                const VertexAttribute<int>& region_label)
{
    GPUTimer timer_total, timer_build, timer_query;
    timer_total.start();

    contact_pairs.reset();

    // Step 1: Get vertex count and context
    uint32_t num_vertices = rx.get_num_elements<VertexHandle>();
    auto     ctx          = rx.get_context();

    // Step 2: Use pre-allocated bounding boxes buffer
    // For point data, each box is degenerate (min == max)
    using box_t = typename BVHBufferT::box_t;
    box_t* d_boxes = bvh_buffers.d_boxes;

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
    timer_build.start();
    cuBQL::BinaryBVH<T, 3> bvh;
    cuBQL::BuildConfig     build_config;
    cuBQL::gpuBuilder(bvh, d_boxes, num_vertices, build_config);
    timer_build.stop();

    // Calculate and print BVH memory usage
    size_t nodes_memory = bvh.numNodes * sizeof(typename cuBQL::BinaryBVH<T, 3>::Node);
    size_t primIDs_memory = bvh.numPrims * sizeof(uint32_t);
    size_t total_bvh_memory = nodes_memory + primIDs_memory;

    // Step 4: Query BVH for each vertex to find nearby vertices
    timer_query.start();
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {

        const Eigen::Vector3<T> xi = x.template to_eigen<3>(vh);
        const uint32_t          vh_id = ctx.template linear_id<VertexHandle>(vh);

        // Query point for BVH traversal
        cuBQL::vec_t<T, 3> query_point;
        query_point.x = xi[0];
        query_point.y = xi[1];
        query_point.z = xi[2];

        // remains invariant
        const int region_vh = region_label(vh);
        const T dhat_sq = dhat * dhat;

        // Fixed-radius query using shrinking radius approach
        auto query_lambda = [&](uint32_t prim_id) -> float {
            if (prim_id == vh_id) {
                return dhat_sq;  // Return SQUARED radius, skip self
            }

            VertexHandle other_vh = ctx.template get_handle<VertexHandle>(prim_id);

            // Only add contact pairs between vertices from different meshes
            int region_other = region_label(other_vh);

            if (region_vh == region_other) {
                return dhat_sq;  // Skip vertices from same mesh
            }

            Eigen::Vector3<T> xj = x.template to_eigen<3>(other_vh);

            T dist_sq = (xi - xj).squaredNorm();

            if (dist_sq < dhat_sq) {
                contact_pairs.insert(vh, other_vh);
            }

            return dhat_sq;  // Return SQUARED radius for fixed-radius query
        };

        cuBQL::shrinkingRadiusQuery::forEachPrim<T, 3>(
            query_lambda, bvh, query_point, dhat_sq);
    });
    timer_query.stop();

    timer_total.stop();

    // Print timing information
    // RXMESH_INFO("VV Contact Detection:");
    // RXMESH_INFO("  BVH Build time: {:.3f} ms", timer_build.elapsed_millis());
    // RXMESH_INFO("  BVH Query time: {:.3f} ms", timer_query.elapsed_millis());
    // RXMESH_INFO("  Total time: {:.3f} ms", timer_total.elapsed_millis());
    // RXMESH_INFO("  Contact pairs found: {}", contact_pairs.num_pairs());
    // RXMESH_INFO("  BVH memory: {:.2f} MB", total_bvh_memory / (1024.0f * 1024.0f));

    // Step 5: Cleanup
    cuBQL::cuda::free(bvh);  // Free BVH memory (nodes and primIDs go to pool)
    // Note: d_boxes is from pre-allocated buffer, NOT freed here
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

    problem.template add_interaction_term<Op::VV, true>([=] __device__(const auto& id,
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
            E = h_sq * avg_contact_area * dhat * T(0.5) * kappa * (s - 1) * log(s);
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
          typename PairT,
          typename BVHBufferT,
          typename T = typename VAttrT::Type>
void add_contact(ProblemT&          problem,
                 RXMeshStatic&      rx,
                 PairT&             contact_pairs,
                 BVHBufferT&        bvh_buffers,
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
               bvh_buffers,
               x,
               contact_area,
               h,
               dhat,
               kappa,
               region_label);

    // Call VF contact handler
    // vf_contact(problem, rx, is_dbc, x, contact_area, h, dhat, kappa);
}

template <typename VAttrT,
          typename DenseMatT,
          typename T = typename VAttrT::Type>
T barrier_step_size(RXMeshStatic&      rx,
                    const DenseMatT&   search_dir,
                    DenseMatT&         alpha,
                    const VAttrT&      x,
                    const vec3<T>&     ground_n,
                    const vec3<T>&     ground_o)
{
    alpha.reset(T(1), DEVICE);

    const vec3<T> n(0.0, -1.0, 0.0);

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {

        const vec3<T> pi(
            search_dir(vh, 0), search_dir(vh, 1), search_dir(vh, 2));

        const vec3<T> xi = x.to_glm<3>(vh);

        // floor
        T p_n = glm::dot(pi, ground_n);
        if (p_n < 0) {
            alpha(vh) = std::min(
                alpha(vh), T(0.9) * glm::dot(ground_n, (xi - ground_o)) / -p_n);
        }

    });

    // we want the min here but since the min value is greater than 1 (y_ground
    // is less than 0, and search_dir is also less than zero)
    return alpha.abs_min();
}
