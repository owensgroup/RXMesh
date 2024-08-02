#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

using namespace rxmesh;


template <typename T, uint32_t blockThreads>
__global__ static void compute_area_matrix(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<int> boundaryVertices,
    rxmesh::SparseMatrix<cuComplex>    AreaMatrix)
{

    auto vn_lambda = [&](EdgeHandle edge_id, VertexIterator& vv) {
        if (boundaryVertices(vv[0], 0) == 1 && boundaryVertices(vv[1], 0) == 1) {
            AreaMatrix(vv[0], vv[1]) = make_cuComplex(0, -0.25);  // modify later
            AreaMatrix(vv[1], vv[0]) = make_cuComplex(0, 0.25);
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, vn_lambda);
}

template <typename T>
__device__ __forceinline__ T
edge_cotan_weight(const rxmesh::VertexHandle&       p_id,
                  const rxmesh::VertexHandle&       r_id,
                  const rxmesh::VertexHandle&       q_id,
                  //const rxmesh::VertexHandle&       s_id,
                  const rxmesh::VertexAttribute<T>& X)
{
    // Get the edge weight between the two vertices p-r where
    // q and s composes the diamond around p-r

    const vec3<T> p(X(p_id, 0), X(p_id, 1), X(p_id, 2));
    const vec3<T> r(X(r_id, 0), X(r_id, 1), X(r_id, 2));
    const vec3<T> q(X(q_id, 0), X(q_id, 1), X(q_id, 2));
    //const vec3<T> s(X(s_id, 0), X(s_id, 1), X(s_id, 2));

    // cotans[(v1, v2)] =np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))

    float weight = 0;
    if (q_id.is_valid())
        weight += dot((p - q), (r - q)) / length(cross(p - q, r - q));
    
    //weight /= 2;
    return std::max(0.f, weight);
}




template <typename T, uint32_t blockThreads>
__global__ static void compute_edge_weights_evd(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> coords,
    rxmesh::SparseMatrix<T>    A_mat)
{

    auto vn_lambda = [&](EdgeHandle edge_id, VertexIterator& vv) {
        T e_weight = 0;

        if (vv[1].is_valid())
            e_weight += edge_cotan_weight(vv[0], vv[2], vv[1], coords);
        if (vv[3].is_valid())
            e_weight += edge_cotan_weight(vv[0], vv[2], vv[3], coords);
        //if (vv[1].is_valid() && vv[3].is_valid())
        e_weight /= 4;
        A_mat(vv[0], vv[2]) = e_weight;
        A_mat(vv[2], vv[0]) = e_weight;

    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, vn_lambda);
}

template <typename T, uint32_t blockThreads>
__global__ static void calculate_Ld_matrix(
    const rxmesh::Context   context,
    rxmesh::SparseMatrix<float> weight_mat,  // [num_coord, num_coord]
    rxmesh::SparseMatrix<cuComplex> Ld           // [num_coord, num_coord]
)
{
    auto init_lambda = [&](VertexHandle v_id, VertexIterator& vv) {
        Ld(v_id, v_id) = make_cuComplex(0, 0);

        for (int nei_index = 0; nei_index < vv.size(); nei_index++)
            Ld(v_id, vv[nei_index]) = make_cuComplex(0, 0);

        for (int nei_index = 0; nei_index < vv.size(); nei_index++) {
            Ld(v_id, v_id) =
                cuCaddf(Ld(v_id, v_id),
                        make_cuComplex(weight_mat(v_id, vv[nei_index]),
                                       weight_mat(v_id, vv[nei_index])));

            Ld(v_id, vv[nei_index]) =
                cuCsubf(Ld(v_id, vv[nei_index]),make_cuComplex(weight_mat(v_id, vv[nei_index]), 0));
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, init_lambda);
}

template <typename T, uint32_t blockThreads>
__global__ static void subtract_matrix(const rxmesh::Context   context,
                                       rxmesh::SparseMatrix<T> A_mat,
                                       rxmesh::SparseMatrix<T> B_mat,
                                       rxmesh::SparseMatrix<T> C_mat)
{

    auto subtract = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); ++i) {
            A_mat(v_id, vv[i]) = 
                cuCsubf(B_mat(v_id, vv[i]), C_mat(v_id, vv[i]));
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, subtract);
}

int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");

    SparseMatrix<cuComplex> Ld(rx);  // complex V x V

    SparseMatrix<cuComplex> A(rx);  // 2V x 2V

    auto boundaryVertices =
        *rx.add_vertex_attribute<int>("boundaryVertices", 1);

    auto coords = *rx.get_input_vertex_coordinates();

    rx.get_boundary_vertices(
        boundaryVertices);  // 0 or 1 value for boundary vertex

    // identify boundary edge (vv query)
    // v1 is central; v2 is on boundary



    //for matrix calls
    constexpr uint32_t CUDABlockSize = 256;

    rxmesh::LaunchBox<CUDABlockSize> launch_box_area;
    rx.prepare_launch_box({rxmesh::Op:: EV},
                          launch_box_area,
                          (void*)compute_area_matrix<float, CUDABlockSize>);

    compute_area_matrix<cuComplex, CUDABlockSize>
        <<<launch_box_area.blocks,
           launch_box_area.num_threads,
           launch_box_area.smem_bytes_dyn>>>(
        rx.get_context(), boundaryVertices, A);


    
 SparseMatrix<float> weight_matrix(rx);

    // obtain cotangent weight matrix
    rxmesh::LaunchBox<CUDABlockSize> launch_box;
    rx.prepare_launch_box(
        {rxmesh::Op::EVDiamond},
        launch_box,
        (void*)compute_edge_weights_evd<float, CUDABlockSize>);

    compute_edge_weights_evd<float, CUDABlockSize>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(
            rx.get_context(), coords, weight_matrix);
            
    
    rxmesh::LaunchBox<CUDABlockSize> launch_box_ld;
    rx.prepare_launch_box(
        {rxmesh::Op::VV},
        launch_box_ld,
        (void*)calculate_Ld_matrix<float, CUDABlockSize>);

    calculate_Ld_matrix<float, CUDABlockSize>
        <<<launch_box_ld.blocks,
           launch_box_ld.num_threads,
           launch_box_ld.smem_bytes_dyn>>>(
            rx.get_context(), weight_matrix, Ld);


     SparseMatrix<cuComplex> Lc(rx);
    rxmesh::LaunchBox<CUDABlockSize> launch_box_lc;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_ld,
                          (void*)subtract_matrix<cuComplex, CUDABlockSize>);

    subtract_matrix<cuComplex, CUDABlockSize>
        <<<launch_box_ld.blocks,
           launch_box_ld.num_threads,
           launch_box_ld.smem_bytes_dyn>>>
    (rx.get_context(), Lc, Ld, A);



    //vertex_normals->move(rxmesh::DEVICE, rxmesh::HOST);



    //Lc


#if USE_POLYSCOPE
    polyscope::show();
#endif
}