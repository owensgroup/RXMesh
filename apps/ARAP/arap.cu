#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

using namespace rxmesh;

template <typename T>
__device__ __forceinline__ T
edge_cotan_weight(const rxmesh::VertexHandle&       p_id,
                  const rxmesh::VertexHandle&       r_id,
                  const rxmesh::VertexHandle&       q_id,
                  const rxmesh::VertexHandle&       s_id,
                  const rxmesh::VertexAttribute<T>& X)
{
    // Get the edge weight between the two vertices p-r where
    // q and s composes the diamond around p-r

    const vec3<T> p(X(p_id, 0), X(p_id, 1), X(p_id, 2));
    const vec3<T> r(X(r_id, 0), X(r_id, 1), X(r_id, 2));
    const vec3<T> q(X(q_id, 0), X(q_id, 1), X(q_id, 2));
    const vec3<T> s(X(s_id, 0), X(s_id, 1), X(s_id, 2));

    //cotans[(v1, v2)] =np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))

    T weight = 0;
    if (q_id.is_valid())
        weight   += dot((p - q), (r - q)) / length(cross(p - q, r - q));
    if (s_id.is_valid())
        weight   += dot((p - s), (r - s)) / length(cross(p - s, r - s));
    weight /= 2;
    return weight;
}



template <typename T, uint32_t blockThreads>
__global__ static void compute_edge_weights(const rxmesh::Context      context,
                                             rxmesh::VertexAttribute<T> coords,
                                            rxmesh::SparseMatrix<T> A_mat)
{

    auto vn_lambda = [&](VertexHandle vertex_id, VertexIterator& vv)
    {
        VertexHandle q_id = vv.back();

        for (uint32_t v = 0; v < vv.size(); ++v) 
        {
            VertexHandle r_id = vv[v];
            T e_weight = 0;
            VertexHandle s_id = (v == vv.size() - 1) ? vv[0] : vv[v + 1];
            e_weight = edge_cotan_weight(vertex_id, r_id, q_id, s_id, coords);
            A_mat(vertex_id, vv[v]) = e_weight;
        }

    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, vn_lambda);
}

template <typename T, uint32_t blockThreads>
__global__ static void compute_edge_weights_evd(const rxmesh::Context      context,
                                            rxmesh::VertexAttribute<T> coords,
                                            rxmesh::SparseMatrix<T>    A_mat)
{

    auto vn_lambda = [&](EdgeHandle edge_id, VertexIterator& vv) {
            T e_weight = 0;
            e_weight = edge_cotan_weight(vv[0], vv[2], vv[1], vv[3], coords);
            A_mat(vv[0], vv[2]) = e_weight;
        
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, vn_lambda);
}

template <typename T, uint32_t blockThreads>
__global__ static void edge_weight_values(
    const rxmesh::Context      context,
    rxmesh::EdgeAttribute<T> edge_weights,
    rxmesh::SparseMatrix<T>    A_mat)
{

    auto vn_lambda = [&](EdgeHandle edge_id, VertexIterator& vv) {
        edge_weights(edge_id, 0) = A_mat(vv[0], vv[1]);
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, vn_lambda);
}





int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    //compute wij
    //auto weights = rx.add_edge_attribute<float>("edgeWeights", 1);
    auto vertex_pos = *rx.get_input_vertex_coordinates();
    SparseMatrix<float> weights(rx);
    
    constexpr uint32_t               CUDABlockSize = 256;
    rxmesh::LaunchBox<CUDABlockSize> launch_box;
    rx.prepare_launch_box({rxmesh::Op::EVDiamond},
                          launch_box,
                          (void*)compute_edge_weights_evd<float, CUDABlockSize>);

     compute_edge_weights_evd<float, CUDABlockSize>
        <<<launch_box.blocks,
                                                  launch_box.num_threads,
                                                  launch_box.smem_bytes_dyn>>>(
                                                  rx.get_context(), vertex_pos, weights);
                                                  



#if USE_POLYSCOPE
    polyscope::show();
#endif
}