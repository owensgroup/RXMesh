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


template <typename ProblemT, typename VAttrT, typename T = typename VAttrT::Type>
void box_barrier_energy(ProblemT& problem,
                        VAttrT& contact_area,
                        const T h,
                        const T box_min_x, const T box_max_x,
                        const T box_min_y,
                        const T box_min_z, const T box_max_z,
                        const T dhat,
                        const T kappa)
{
    const T h_sq = h * h;

    // Define 5 planes (inward-pointing normals) - no ceiling
    vec3<T> planes[5] = {
        vec3<T>(0, 1, 0),   // bottom: normal up
        vec3<T>(1, 0, 0),   // left (min_x): normal right
        vec3<T>(-1, 0, 0),  // right (max_x): normal left
        vec3<T>(0, 0, 1),   // front (min_z): normal back
        vec3<T>(0, 0, -1)   // back (max_z): normal forward
    };

    vec3<T> offsets[5] = {
        vec3<T>(0, box_min_y, 0),
        vec3<T>(box_min_x, 0, 0),
        vec3<T>(box_max_x, 0, 0),
        vec3<T>(0, 0, box_min_z),
        vec3<T>(0, 0, box_max_z)
    };

    // Apply barrier energy for each of the 5 faces
    for (int face_idx = 0; face_idx < 5; face_idx++) {
        vec3<T> n = planes[face_idx];
        vec3<T> o = offsets[face_idx];

        // Precompute Eigen vectors outside the device lambda
        const Eigen::Vector3<T> o_eigen(o[0], o[1], o[2]);
        const Eigen::Vector3<T> n_eigen(n[0], n[1], n[2]);
                
        problem.template add_term<Op::V, true>(
            [=] __device__(const auto& vh, auto& obj) mutable {
                using ActiveT = ACTIVE_TYPE(vh);

                const Eigen::Vector3<ActiveT> xi = iter_val<ActiveT, 3>(vh, obj);

                ActiveT d = (xi - o_eigen).dot(n_eigen);

                ActiveT E(T(0));

                if (d < dhat) {
                    ActiveT s = d / dhat;
                    if (s <= T(0)) {
                        using PassiveT = PassiveType<ActiveT>;
                        return ActiveT(std::numeric_limits<PassiveT>::max());
                    }
                    E = h_sq * contact_area(vh) * dhat * T(0.5) * kappa *
                        (s - T(1)) * log(s);
                }

                return E;
            });
    }
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
    // GPUTimer timer_total, timer_build, timer_query;
    // timer_total.start();

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
    // timer_build.start();
    cuBQL::BinaryBVH<T, 3> bvh;
    cuBQL::BuildConfig     build_config;
    cuBQL::gpuBuilder(bvh, d_boxes, num_vertices, build_config);
    // timer_build.stop();

    // Calculate and print BVH memory usage
    size_t nodes_memory = bvh.numNodes * sizeof(typename cuBQL::BinaryBVH<T, 3>::Node);
    size_t primIDs_memory = bvh.numPrims * sizeof(uint32_t);
    size_t total_bvh_memory = nodes_memory + primIDs_memory;

    // Step 4: Query BVH for each vertex to find nearby vertices
    // timer_query.start();
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
    // timer_query.stop();

    // timer_total.stop();

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

// Point-to-triangle distance computation
template <typename T>
__device__ __host__ inline T point_triangle_distance_squared(
    const Eigen::Vector3<T>& p,
    const Eigen::Vector3<T>& a,
    const Eigen::Vector3<T>& b,
    const Eigen::Vector3<T>& c)
{
    // Compute triangle edges
    Eigen::Vector3<T> ab = b - a;
    Eigen::Vector3<T> ac = c - a;
    Eigen::Vector3<T> ap = p - a;

    // Barycentric coordinates
    T d1 = ab.dot(ap);
    T d2 = ac.dot(ap);

    // Check if p projects outside triangle near vertex a
    if (d1 <= T(0) && d2 <= T(0)) {
        return ap.squaredNorm();
    }

    // Check if p projects outside triangle near vertex b
    Eigen::Vector3<T> bp = p - b;
    T d3 = ab.dot(bp);
    T d4 = ac.dot(bp);
    if (d3 >= T(0) && d4 <= d3) {
        return bp.squaredNorm();
    }

    // Check if p projects on edge ab
    T vc = d1 * d4 - d3 * d2;
    if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {
        T v = d1 / (d1 - d3);
        Eigen::Vector3<T> closest = a + v * ab;
        return (p - closest).squaredNorm();
    }

    // Check if p projects outside triangle near vertex c
    Eigen::Vector3<T> cp = p - c;
    T d5 = ab.dot(cp);
    T d6 = ac.dot(cp);
    if (d6 >= T(0) && d5 <= d6) {
        return cp.squaredNorm();
    }

    // Check if p projects on edge ac
    T vb = d5 * d2 - d1 * d6;
    if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {
        T w = d2 / (d2 - d6);
        Eigen::Vector3<T> closest = a + w * ac;
        return (p - closest).squaredNorm();
    }

    // Check if p projects on edge bc
    T va = d3 * d6 - d5 * d4;
    if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {
        T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        Eigen::Vector3<T> closest = b + w * (c - b);
        return (p - closest).squaredNorm();
    }

    // p projects inside triangle
    T denom = T(1) / (va + vb + vc);
    T v = vb * denom;
    T w = vc * denom;
    Eigen::Vector3<T> closest = a + ab * v + ac * w;
    return (p - closest).squaredNorm();
}

template <typename ProblemT,
          typename VAttrT,
          typename T = typename VAttrT::Type>
void vf_contact_energy(ProblemT&     problem,
                       const VAttrT& contact_area,
                       const T       h,
                       const T       dhat,
                       const T       kappa)
{
    const T h_sq = h * h;

    problem.template add_interaction_term<Op::VF, true>(
        [=] __device__(const auto& fh,
                       const auto& vh,
                       const auto& iter,
                       const auto& obj) mutable {

            using ActiveT = ACTIVE_TYPE(fh);

            // Get vertex and face vertices positions
            const Eigen::Vector3<ActiveT> xi = iter_val<ActiveT, 3>(fh, vh, iter, obj, 0);
            const Eigen::Vector3<ActiveT> p0 = iter_val<ActiveT, 3>(fh, vh, iter, obj, 1);
            const Eigen::Vector3<ActiveT> p1 = iter_val<ActiveT, 3>(fh, vh, iter, obj, 2);
            const Eigen::Vector3<ActiveT> p2 = iter_val<ActiveT, 3>(fh, vh, iter, obj, 3);

            // Compute point-to-triangle distance
            ActiveT d_sq = point_triangle_distance_squared(xi, p0, p1, p2);
            ActiveT d = sqrt(d_sq);

            ActiveT E(T(0));

            if (d < dhat) {
                ActiveT s = d / dhat;

                if (s <= T(0)) {
                    using PassiveT = PassiveType<ActiveT>;
                    return ActiveT(std::numeric_limits<PassiveT>::max());
                }

                // Compute triangle area using cross product: Area = 0.5 * ||(p1-p0) × (p2-p0)||
                Eigen::Vector3<ActiveT> edge1 = p1 - p0;
                Eigen::Vector3<ActiveT> edge2 = p2 - p0;
                ActiveT triangle_area = T(0.5) * edge1.cross(edge2).norm();

                // Barrier energy: E = h^2 * A * dhat * 0.5 * kappa * (s - 1) * log(s)
                E = h_sq * triangle_area * dhat * T(0.5) * kappa * (s - 1) * log(s);
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
          typename PairT,
          typename BVHBufferT,
          typename T = typename VAttrT::Type>
void vf_contact(ProblemT&     problem,
                RXMeshStatic& rx,
                PairT&        vf_contact_pairs,
                BVHBufferT&   face_bvh_buffers,
                const VAttrT& x,
                VAttrT&       contact_area,
                const T       h,
                const T       dhat,
                const T       kappa,
                const VertexAttribute<int>& vertex_region_label,
                const FaceAttribute<int>& face_region_label,
                const FaceAttribute<uint64_t>& face_vertices)
{
    // GPUTimer timer_total, timer_build, timer_query;
    // timer_total.start();

    vf_contact_pairs.reset();

    // Step 1: Get face count and context
    uint32_t num_faces = rx.get_num_elements<FaceHandle>();
    auto     ctx       = rx.get_context();

    // Step 2: Use pre-allocated bounding boxes buffer
    using box_t = typename BVHBufferT::box_t;
    box_t* d_boxes = face_bvh_buffers.d_boxes;

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
    // timer_build.start();
    cuBQL::BinaryBVH<T, 3> bvh;
    cuBQL::BuildConfig     build_config;
    cuBQL::gpuBuilder(bvh, d_boxes, num_faces, build_config);
    // timer_build.stop();

    // Calculate and print BVH memory usage
    size_t nodes_memory     = bvh.numNodes * sizeof(typename cuBQL::BinaryBVH<T, 3>::Node);
    size_t primIDs_memory   = bvh.numPrims * sizeof(uint32_t);
    size_t total_bvh_memory = nodes_memory + primIDs_memory;

    // Step 5: Query BVH for VF contact detection
    // Note: this is a more broader/lenient query as we're not testing for exact closeness here
    // This is because right now there isn't a way to obtain the vertex positions from the face handle
    // timer_query.start();
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {

        const Eigen::Vector3<T> xi = x.template to_eigen<3>(vh);

        // Query point for BVH traversal
        cuBQL::vec_t<T, 3> query_point;
        query_point.x = xi[0];
        query_point.y = xi[1];
        query_point.z = xi[2];

        // remains invariant
        const int region_vh = vertex_region_label(vh);
        const T dhat_sq = dhat * dhat;

        // Fixed-radius query to find nearby triangles
        auto query_lambda = [&](uint32_t prim_id) -> float {
            FaceHandle fh = ctx.template get_handle<FaceHandle>(prim_id);

            const int region_fh = face_region_label(fh);
            if (region_fh != region_vh) {
                // Get the face vertices using face_vertices attribute
                uint64_t v0_id = face_vertices(fh, 0);
                uint64_t v1_id = face_vertices(fh, 1);
                uint64_t v2_id = face_vertices(fh, 2);

                VertexHandle v0(v0_id);
                VertexHandle v1(v1_id);
                VertexHandle v2(v2_id);

                // Get vertex positions
                Eigen::Vector3<T> p0 = x.template to_eigen<3>(v0);
                Eigen::Vector3<T> p1 = x.template to_eigen<3>(v1);
                Eigen::Vector3<T> p2 = x.template to_eigen<3>(v2);

                // Compute point-to-triangle distance
                T d_sq = point_triangle_distance_squared(xi, p0, p1, p2);

                // Only add contact if within dhat
                if (d_sq < dhat_sq) {
                    vf_contact_pairs.insert(vh, fh);
                }
            }
            return dhat_sq;  // Return SQUARED radius for fixed-radius query
        };

        cuBQL::shrinkingRadiusQuery::forEachPrim<T, 3>(
            query_lambda, bvh, query_point, dhat_sq);
    });
    // timer_query.stop();

    // timer_total.stop();

    // Print timing information
    // RXMESH_INFO("VF Contact Detection:");
    // RXMESH_INFO("  BVH Build time: {:.3f} ms", timer_build.elapsed_millis());
    // RXMESH_INFO("  BVH Query time: {:.3f} ms", timer_query.elapsed_millis());
    // RXMESH_INFO("  Total time: {:.3f} ms", timer_total.elapsed_millis());
    // RXMESH_INFO("  Contact pairs found: {}", vf_contact_pairs.num_pairs());
    // RXMESH_INFO("  BVH memory: {:.2f} MB", total_bvh_memory / (1024.0f * 1024.0f));

    // Step 6: Cleanup
    cuBQL::cuda::free(bvh);  // Free BVH memory (nodes and primIDs go to pool)
    // Note: d_boxes is from pre-allocated buffer, NOT freed here
}

template <typename ProblemT,
          typename VAttrT,
          typename VVPairT,
          typename VFPairT,
          typename BVHBufferT,
          typename T = typename VAttrT::Type>
void add_contact(ProblemT&          problem,
                 RXMeshStatic&      rx,
                 VVPairT&           vv_contact_pairs,
                 VFPairT&           vf_contact_pairs,
                 BVHBufferT&        vertex_bvh_buffers,
                 BVHBufferT&        face_bvh_buffers,
                 const VAttrT&      x,
                 VAttrT&            contact_area,
                 const T            h,
                 const T            dhat,
                 const T            kappa,
                 const VertexAttribute<int>& vertex_region_label,
                 const FaceAttribute<int>& face_region_label,
                 const FaceAttribute<uint64_t>& face_vertices)
{
    // Call VV contact handler
    vv_contact(problem,
               rx,
               vv_contact_pairs,
               vertex_bvh_buffers,
               x,
               contact_area,
               h,
               dhat,
               kappa,
               vertex_region_label);

    // Call VF contact handler
    vf_contact(problem,
               rx,
               vf_contact_pairs,
               face_bvh_buffers,
               x,
               contact_area,
               h,
               dhat,
               kappa,
               vertex_region_label,
               face_region_label,
               face_vertices);
}

template <typename VAttrT,
          typename DenseMatT,
          typename T = typename VAttrT::Type>
T box_barrier_step_size(RXMeshStatic& rx,
                        const DenseMatT& search_dir,
                        DenseMatT& alpha,
                        const VAttrT& x,
                        const T box_min_x, const T box_max_x,
                        const T box_min_y,
                        const T box_min_z, const T box_max_z)
{
    alpha.reset(T(1), DEVICE);

    // Define 5 planes (same as in box_barrier_energy) - no ceiling
    vec3<T> planes[5] = {
        vec3<T>(0, 1, 0),   vec3<T>(1, 0, 0),   vec3<T>(-1, 0, 0),
        vec3<T>(0, 0, 1),   vec3<T>(0, 0, -1)
    };

    vec3<T> offsets[5] = {
        vec3<T>(0, box_min_y, 0), vec3<T>(box_min_x, 0, 0), vec3<T>(box_max_x, 0, 0),
        vec3<T>(0, 0, box_min_z), vec3<T>(0, 0, box_max_z)
    };

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        const vec3<T> pi(search_dir(vh, 0), search_dir(vh, 1), search_dir(vh, 2));
        const vec3<T> xi = x.to_glm<3>(vh);

        // Check all 5 faces
        for (int face = 0; face < 5; face++) {
            vec3<T> n = planes[face];
            vec3<T> o = offsets[face];

            T p_n = glm::dot(pi, n);
            if (p_n < 0) {  // moving toward this wall
                alpha(vh) = std::min(alpha(vh),
                                     T(0.9) * glm::dot(n, (xi - o)) / -p_n);
            }
        }
    });

    return alpha.abs_min();
}
