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

template <typename VAttrT,
          typename VAttrB,
          typename PairT,
          typename T = typename VAttrT::Type>
void add_contact(RXMeshStatic&      rx,
                 PairT&             contact_pairs,
                 const VertexHandle dbc_vertex,
                 const VertexHandle dbc_vertex1,
                 const VertexHandle dbc_vertex2,
                 const VAttrB&      is_dbc,
                 const VAttrT&      x,
                 const T            dhat)
{
    printf("Inside Contact Pairs\n");
    printf("Contact Pairs Size Before Reset: %d\n", contact_pairs.num_pairs());
    contact_pairs.reset();
    printf("Contact Pairs Size After Reset: %d\n", contact_pairs.num_pairs());

    // Step 1: Get vertex count and context
    uint32_t num_vertices = rx.get_num_elements<VertexHandle>();
    auto     ctx          = rx.get_context();
    printf("Number of vertices: %u\n", num_vertices);

    // Step 2: Allocate and populate bounding boxes for BVH
    // For point data, each box is degenerate (min == max)
    using box_t = cuBQL::box_t<T, 3>;
    box_t* d_boxes = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_boxes, sizeof(box_t) * num_vertices));
    printf("Allocated memory for the bvh boxes: %u bytes.\n", sizeof(box_t) * num_vertices);

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

            // Print first few boxes for debugging
            // if (i < 5) {
            //     printf("Box[%d]: lower=(%f, %f, %f), upper=(%f, %f, %f)\n",
            //            i,
            //            d_boxes[i].lower.x, d_boxes[i].lower.y, d_boxes[i].lower.z,
            //            d_boxes[i].upper.x, d_boxes[i].upper.y, d_boxes[i].upper.z);
            // }
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

    printf("Built the BVH.\n");
    printf("  Number of nodes: %u\n", bvh.numNodes);
    printf("  Number of primitives: %u\n", bvh.numPrims);
    printf("  Nodes memory: %zu bytes (%.2f MB)\n", nodes_memory, nodes_memory / (1024.0f * 1024.0f));
    printf("  PrimIDs memory: %zu bytes (%.2f MB)\n", primIDs_memory, primIDs_memory / (1024.0f * 1024.0f));
    printf("  Total BVH memory: %zu bytes (%.2f MB)\n", total_bvh_memory, total_bvh_memory / (1024.0f * 1024.0f));

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

            VertexHandle      other_vh = ctx.template get_handle<VertexHandle>(prim_id);
            Eigen::Vector3<T> xj       = x.template to_eigen<3>(other_vh);

            T dist = (xi - xj).norm();

            if (dist < dhat) {
                contact_pairs.insert(vh, other_vh);
            }

            return dhat * dhat;  // Return SQUARED radius for fixed-radius query
        };

        cuBQL::shrinkingRadiusQuery::forEachPrim<T, 3>(
            query_lambda, bvh, query_point, dhat * dhat);
    });

    printf("Traversed the BVH and executed the contact pair addition.\n");
    printf("Contact Pairs Size After Traversal: %d\n", contact_pairs.num_pairs());

    // Step 5: Cleanup
    cuBQL::cuda::free(bvh);  // Free BVH memory (nodes and primIDs)
    GPU_FREE(d_boxes);  // Free bounding boxes
    printf("Released BVH and bounding box memory.\n");
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
