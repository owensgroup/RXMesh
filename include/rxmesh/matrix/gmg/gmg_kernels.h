#pragma once

#include "rxmesh/context.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/query.cuh"

#include "rxmesh/matrix/gmg/hashtable.h"

namespace rxmesh {
namespace detail {

template <uint32_t blockThreads>
__global__ static void populate_edge_hashtable_1st_level(
    const Context          context,
    const DenseMatrix<int> vertex_cluster,
    GPUStorage<Edge>       edge_storage)
{

    auto count = [&](EdgeHandle eh, VertexIterator& iter) {
        assert(iter.size() == 2);

        VertexHandle v0 = iter[0];
        VertexHandle v1 = iter[1];

        int v0_sample = vertex_cluster(v0);
        int v1_sample = vertex_cluster(v1);

        if (v0_sample != v1_sample) {
            Edge e(v0_sample, v1_sample);
            bool inserted = edge_storage.insert(e);
            assert(inserted);
        }
    };


    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, count);
}


__device__ __inline__ void compute_barycentric(const Eigen::Vector3f& v0,
                                               const Eigen::Vector3f& v1,
                                               const Eigen::Vector3f& v2,
                                               const Eigen::Vector3f& p,
                                               float&                 a,
                                               float&                 b,
                                               float&                 c)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1    = v1 - v0;
    Eigen::Vector3f edge2    = v2 - v0;
    Eigen::Vector3f pointVec = p - v0;

    // Compute normal of the triangle
    Eigen::Vector3f normal = edge1.cross(edge2);
    float area2 = normal.squaredNorm();  // Area of the triangle multiplied by 2

    // Compute barycentric coordinates
    float lambda0 = 0, lambda1 = 0, lambda2 = 0;

    lambda0 = (v1 - p).cross(v2 - p).dot(normal) / area2;
    lambda1 = (v2 - p).cross(v0 - p).dot(normal) / area2;
    lambda2 = (v0 - p).cross(v1 - p).dot(normal) / area2;

    a = lambda0;
    b = lambda1;
    c = lambda2;
}

// Compute barycentric coordinates for closest point on triangle
__device__ __inline__ void compute_positive_barycentric_coords(
    const Eigen::Vector3f& p,
    const Eigen::Vector3f& a,
    const Eigen::Vector3f& b,
    const Eigen::Vector3f& c,
    Eigen::Vector3f&       barycentricCoords)
{
    // Edge vectors
    Eigen::Vector3f ab = b - a;
    Eigen::Vector3f ac = c - a;
    Eigen::Vector3f ap = p - a;

    float d1 = ab.dot(ap);
    float d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        barycentricCoords = Eigen::Vector3f(1, 0, 0);  // a
        return;
    }

    // Check if P in vertex region outside B
    Eigen::Vector3f bp = p - b;
    float           d3 = ab.dot(bp);
    float           d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        barycentricCoords = Eigen::Vector3f(0, 1, 0);  // b
        return;
    }

    // Check if P in edge region of AB
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v           = d1 / (d1 - d3);
        barycentricCoords = Eigen::Vector3f(1 - v, v, 0);
        return;
    }

    // Check if P in vertex region outside C
    Eigen::Vector3f cp = p - c;
    float           d5 = ab.dot(cp);
    float           d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        barycentricCoords = Eigen::Vector3f(0, 0, 1);  // c
        return;
    }

    // Check if P in edge region of AC
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w           = d2 / (d2 - d6);
        barycentricCoords = Eigen::Vector3f(1 - w, 0, w);
        return;
    }

    // Check if P in edge region of BC
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w           = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        barycentricCoords = Eigen::Vector3f(0, 1 - w, w);
        return;
    }

    // P inside face region � compute barycentrics via projection
    float denom = ab.dot(ab) * ac.dot(ac) - ab.dot(ac) * ab.dot(ac);
    float v     = (ap.dot(ab) * ac.dot(ac) - ap.dot(ac) * ab.dot(ac)) / denom;
    float w     = (ap.dot(ac) * ab.dot(ab) - ap.dot(ab) * ab.dot(ac)) / denom;
    float u     = 1.0f - v - w;
    barycentricCoords = Eigen::Vector3f(u, v, w);
}


__device__ __inline__ float projected_distance(const Eigen::Vector3f& v0,
                                               const Eigen::Vector3f& v1,
                                               const Eigen::Vector3f& v2,
                                               const Eigen::Vector3f& p)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1 = v1 - v0;
    Eigen::Vector3f edge2 = v2 - v0;

    // Compute the triangle normal
    Eigen::Vector3f normal        = edge1.cross(edge2);
    float           normal_length = normal.norm();

    if (normal_length < 1e-6f) {
        return -1.0f;  // Return -1 to indicate an error
    }

    // Normalize the normal
    normal.normalize();

    // Compute vector from point to the triangle vertex
    Eigen::Vector3f point_to_vertex = p - v0;

    // Project the vector onto the normal
    float distance = point_to_vertex.dot(normal);

    // Return the absolute distance
    return std::fabs(distance);
}

__global__ void build_next_ring_kernel(int               num_vertices,
                                       const int*        csr_offsets,
                                       const int*        csr_neighbors,
                                       GPUStorage<Edge>& next_edge_storage)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices)
        return;

    int start = csr_offsets[v];
    int end   = csr_offsets[v + 1];

    for (int j = start; j < end; ++j) {
        int nbr     = csr_neighbors[j];
        int n_start = csr_offsets[nbr];
        int n_end   = csr_offsets[nbr + 1];
        for (int k = n_start; k < n_end; ++k) {
            int n2 = csr_neighbors[k];
            printf("\n%d is a neighbor of %d", n2, nbr);
            if (v != n2) {
                next_edge_storage.insert(Edge(v, n2));
            }
        }
    }
}

__global__ void expand_from_frontier_kernel(const int        num_vertices,
                                            const int*       csr_offsets,
                                            const int*       csr_neighbors,
                                            const Edge*      frontier_edges,
                                            int              frontier_size,
                                            GPUStorage<Edge> next_frontier)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size)
        return;

    Edge edge   = frontier_edges[idx];
    auto [v, u] = edge.unpack();

    // Expand from u: generate (v, w)
    int start_u = csr_offsets[u];
    int end_u   = csr_offsets[u + 1];
    for (int i = start_u; i < end_u; ++i) {
        int w = csr_neighbors[i];
        if (w != v) {
            next_frontier.insert(Edge(v, w));
        }
    }

    // Expand from v: generate (u, w)
    int start_v = csr_offsets[v];
    int end_v   = csr_offsets[v + 1];
    for (int i = start_v; i < end_v; ++i) {
        int w = csr_neighbors[i];
        if (w != u) {
            next_frontier.insert(Edge(u, w));
        }
    }
}

void build_n_ring_on_gpu_compute(GPUStorage<Edge>  input_edges,
                                 GPUStorage<Edge>& out_nring_edges,
                                 int               num_vertices,
                                 int               max_ring)
{
    constexpr int blockThreads = 256;
    input_edges.uniquify();

    // 1. Build CSR from input_edges
    int* degrees;
    cudaMalloc(&degrees, sizeof(int) * num_vertices);
    cudaMemset(degrees, 0, sizeof(int) * num_vertices);

    
    input_edges.for_each([] __device__(const Edge& e) {
        auto [a, b] = e.unpack();
    });

    input_edges.for_each([degrees] __device__(const Edge& e) {
        auto [a, b] = e.unpack();
        ::atomicAdd(&degrees[a], 1);
        ::atomicAdd(&degrees[b], 1);
    });

    int* csr_offsets;
    cudaMalloc(&csr_offsets, sizeof(int) * (num_vertices + 1));
    cudaMemset(csr_offsets, 0, sizeof(int) * (num_vertices + 1));

    thrust::exclusive_scan(thrust::device, degrees, degrees + num_vertices+1, csr_offsets);

    int total_neighbors;
    cudaMemcpy(&total_neighbors,
               &csr_offsets[num_vertices],
               sizeof(int),
               cudaMemcpyDeviceToHost);


    int* csr_neighbors;
    int* csr_insert_ptrs;
    cudaMalloc(&csr_neighbors, sizeof(int) * total_neighbors);
    cudaMalloc(&csr_insert_ptrs, sizeof(int) * (num_vertices + 1));
    cudaMemcpy(csr_insert_ptrs,
               csr_offsets,
               sizeof(int) * (num_vertices + 1),
               cudaMemcpyDeviceToDevice);


    input_edges.for_each(
        [csr_insert_ptrs, csr_neighbors] __device__(const Edge& e) {
            auto [a, b] = e.unpack();
            int ia      = ::atomicAdd(&csr_insert_ptrs[a], 1);
            int ib      = ::atomicAdd(&csr_insert_ptrs[b], 1);
            csr_neighbors[ia] = b;
            csr_neighbors[ib] = a;
        });

    cudaFree(degrees);
    cudaFree(csr_insert_ptrs);
    
    

    // 2. Initialize visited_edges and frontier
    GPUStorage<Edge> visited_edges(input_edges.get_capacity() * max_ring);
    GPUStorage<Edge> frontier_edges(input_edges.get_capacity() * 2);

    {
        int   init_count = input_edges.count();
        Edge* init_data  = input_edges.m_storage;

        int blocks = (init_count + blockThreads - 1) / blockThreads;

        input_edges.for_each(
            [visited_edges, frontier_edges] __device__(const Edge& e) mutable {
                auto [a, b] = e.unpack();
                if (a > b) {
                    int tmp = a;
                    a       = b;
                    b       = tmp;
                }
                Edge canon_edge(a, b);
                visited_edges.insert(canon_edge);
                frontier_edges.insert(canon_edge);
            });
        cudaDeviceSynchronize();
    }

    GPUStorage<Edge> next_frontier(frontier_edges.get_capacity() * 2);

    // 3. Ring Expansion
    for (int ring = 2; ring <= max_ring; ++ring) {
        int frontier_size = frontier_edges.count();


        if (frontier_size == 0) 
        {
            break;
        }

        Edge* frontier_data = frontier_edges.m_storage;
        int   blocks        = (frontier_size + blockThreads - 1) / blockThreads;

        expand_from_frontier_kernel<<<blocks, blockThreads>>>(num_vertices,
                                                              csr_offsets,
                                                              csr_neighbors,
                                                              frontier_data,
                                                              frontier_size,
                                                              next_frontier);
        cudaDeviceSynchronize();
        // Deduplicate before merging
        next_frontier.uniquify();
        // Merge into visited
        next_frontier.for_each(
            [visited_edges] __device__(const Edge& e) mutable {
                auto [a, b] = e.unpack();
                if (a > b) {
                    int tmp = a;
                    a       = b;
                    b       = tmp;
                }
                visited_edges.insert(Edge(a, b));
            });


        cudaDeviceSynchronize();
        frontier_edges.free();
        frontier_edges = std::move(next_frontier);
        next_frontier  = GPUStorage<Edge>(frontier_edges.get_capacity() * 2);
    }

    cudaFree(csr_offsets);
    cudaFree(csr_neighbors);

    // Final result
    out_nring_edges = std::move(visited_edges);

    int final_count = out_nring_edges.count();
    out_nring_edges.for_each([] __device__(const Edge& e) {
        auto [a, b] = e.unpack();
    });
    out_nring_edges.uniquify();
}


}  // namespace detail
}  // namespace rxmesh