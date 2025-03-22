#include "rxmesh/query.cuh"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/geometry_util.cuh"

using namespace rxmesh;

template <typename T>
using MatVecOp = std::function<void(rxmesh::RXMeshStatic&,
                                    const rxmesh::VertexAttribute<T>&,
                                    rxmesh::VertexAttribute<T>&)>;

template <typename T>
__device__ __forceinline__ T
partial_voronoi_area(const rxmesh::VertexHandle&       p_id,  // center
                     const rxmesh::VertexHandle&       q_id,  // before center
                     const rxmesh::VertexHandle&       r_id,  // after center
                     const rxmesh::VertexAttribute<T>& X)
{
    // compute partial Voronoi area of the center vertex that is associated with
    // the triangle p->q->r (oriented ccw)
    using namespace rxmesh;

    const vec3<T> p = X.to_glm<3>(p_id);
    const vec3<T> q = X.to_glm<3>(q_id);
    const vec3<T> r = X.to_glm<3>(r_id);

    return partial_voronoi_area(p, q, r);
}

template <typename T>
void init_PR(rxmesh::RXMeshStatic&             rx,
             const rxmesh::VertexAttribute<T>& B,
             const rxmesh::VertexAttribute<T>& S,
             rxmesh::VertexAttribute<T>&       R,
             rxmesh::VertexAttribute<T>&       P)
{
    rx.for_each_vertex(rxmesh::DEVICE,
                       [B, S, R, P] __device__(const rxmesh::VertexHandle vh) {
                           R(vh, 0) = B(vh, 0) - S(vh, 0);
                           R(vh, 1) = B(vh, 1) - S(vh, 1);
                           R(vh, 2) = B(vh, 2) - S(vh, 2);

                           P(vh, 0) = R(vh, 0);
                           P(vh, 1) = R(vh, 1);
                           P(vh, 2) = R(vh, 2);
                       });
}
template <typename T>
void axpy(rxmesh::RXMeshStatic&             rx,
          rxmesh::VertexAttribute<T>&       y,
          const rxmesh::VertexAttribute<T>& x,
          const T                           alpha,
          const T                           beta,
          cudaStream_t                      stream = NULL)
{
    // Y = alpha*X + beta*Y
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [y, x, alpha, beta] __device__(const rxmesh::VertexHandle vh) {
            for (uint32_t i = 0; i < 3; ++i) {
                y(vh, i) = alpha * x(vh, i) + beta * y(vh, i);
            }
        });
}

template <typename T>
void cg_solver(rxmesh::RXMeshStatic&       rx,
               rxmesh::VertexAttribute<T>& X,
               rxmesh::VertexAttribute<T>& B,
                
               MatVecOp<T>                 matvec_op,
               float tolerance=0.00001,
    int max_iter=1000)
{
    rxmesh::VertexAttribute<T> S =
        *rx.add_vertex_attribute<T>("S", 3, rxmesh::DEVICE, rxmesh::SoA);
    rxmesh::VertexAttribute<T> R =
        *rx.add_vertex_attribute<T>("R", 3, rxmesh::DEVICE, rxmesh::SoA);
    rxmesh::VertexAttribute<T> P =
        *rx.add_vertex_attribute<T>("P", 3, rxmesh::DEVICE, rxmesh::SoA);

    // Initialize R = B - AX
    matvec_op(rx, X, S);
    init_PR(rx, B, S, R, P);

    VertexReduceHandle<float> reduce_handle(X);

    // Conjugate gradient iteration
    T delta_new = reduce_handle.norm2(R);
    delta_new *= delta_new;
    T delta_0 = delta_new;
    T delta_old = 0;
    uint32_t num_cg_iter_taken = 0;

    while (num_cg_iter_taken < max_iter) {
        // Compute S = A * P
        matvec_op(rx, P, S);

        // CG updates
        T alpha = delta_new / reduce_handle.dot(S, P);
        axpy<T>(rx, X, P, alpha, 1.0);
        axpy<T>(rx, R, S, -alpha, 1.0);

        CUDA_ERROR(cudaStreamSynchronize(0));
        delta_old = delta_new;


        // delta_new = <r,r>
        delta_new = reduce_handle.norm2(R);
        delta_new *= delta_new;

        CUDA_ERROR(cudaStreamSynchronize(0));

        if (delta_new < tolerance * tolerance * delta_0)
            break;

        T beta = delta_new / delta_old;
        axpy<T>(rx, P, R, 1.0, beta);

        num_cg_iter_taken++;
        CUDA_ERROR(cudaStreamSynchronize(0));

    }
}

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
    using namespace rxmesh;

    const vec3<T> p = X.to_glm<3>(p_id);
    const vec3<T> r = X.to_glm<3>(r_id);
    const vec3<T> q = X.to_glm<3>(q_id);
    const vec3<T> s = X.to_glm<3>(s_id);

    return edge_cotan_weight(p, r, q, s);
}
template <typename T, uint32_t blockThreads>
__global__ static void rxmesh_matvec(const rxmesh::Context            context,
                                     const rxmesh::VertexAttribute<T> coords,
                                     const rxmesh::VertexAttribute<T> in,
                                     rxmesh::VertexAttribute<T>       out,
                                     const bool use_uniform_laplace,
                                     const T    time_step)
{
    using namespace rxmesh;

    auto matvec_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        T sum_e_weight(0);

        vec3<T> x(T(0));

        // vertex weight
        T v_weight(0);

        // this is the last vertex in the one-ring (before r_id)
        VertexHandle q_id = iter.back();

        for (uint32_t v = 0; v < iter.size(); ++v) {
            // the current one ring vertex
            VertexHandle r_id = iter[v];

            T e_weight = 0;
            if (use_uniform_laplace) {
                e_weight = 1;
            } else {
                // the second vertex in the one ring (after r_id)
                VertexHandle s_id =
                    (v == iter.size() - 1) ? iter[0] : iter[v + 1];

                e_weight = edge_cotan_weight(p_id, r_id, q_id, s_id, coords);

                // e_weight = max(0, e_weight) but without branch divergence
                e_weight = (static_cast<T>(e_weight >= 0.0)) * e_weight;
            }

            e_weight *= time_step;
            sum_e_weight += e_weight;

            x[0] -= e_weight * in(r_id, 0);
            x[1] -= e_weight * in(r_id, 1);
            x[2] -= e_weight * in(r_id, 2);


            // compute vertex weight
            if (use_uniform_laplace) {
                ++v_weight;
            }
            /*else {
                T tri_area = partial_voronoi_area(p_id, q_id, r_id, coords);
                v_weight += (tri_area > 0) ? tri_area : 0;
                q_id = r_id;
            }*/
        }

        // Diagonal entry
        if (use_uniform_laplace) {
            v_weight = 1.0 / v_weight;
        } else {
            v_weight = 0.5 / v_weight;
        }

        assert(!isnan(v_weight));
        assert(!isinf(v_weight));

        T diag       = ((1.0 / v_weight) + sum_e_weight);
        out(p_id, 0) = x[0] + diag * in(p_id, 0);
        out(p_id, 1) = x[1] + diag * in(p_id, 1);
        out(p_id, 2) = x[2] + diag * in(p_id, 2);
    };

    // With uniform Laplacian, we just need the valence, thus we
    // call query and set oriented to false

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(
        block,
        shrd_alloc,
        matvec_lambda,
        [](VertexHandle) { return true; },
        !use_uniform_laplace);
}
template <typename T, uint32_t blockThreads>
__global__ static void init_B(const rxmesh::Context            context,
                              const rxmesh::VertexAttribute<T> X,
                              rxmesh::VertexAttribute<T>       B,
                              const bool use_uniform_laplace)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        if (use_uniform_laplace) {
            const T valence = static_cast<T>(iter.size());
            B(p_id, 0)      = X(p_id, 0) * valence;
            B(p_id, 1)      = X(p_id, 1) * valence;
            B(p_id, 2)      = X(p_id, 2) * valence;
        } else {

            // using Laplace weights
            T v_weight = 0;

            // this is the last vertex in the one-ring (before r_id)
            VertexHandle q_id = iter.back();

            for (uint32_t v = 0; v < iter.size(); ++v) {
                // the current one ring vertex
                VertexHandle r_id = iter[v];

                T tri_area = partial_voronoi_area(p_id, q_id, r_id, X);

                v_weight += (tri_area > 0) ? tri_area : 0.0;

                q_id = r_id;
            }
            v_weight = 0.5 / v_weight;

            B(p_id, 0) = X(p_id, 0) / v_weight;
            B(p_id, 1) = X(p_id, 1) / v_weight;
            B(p_id, 2) = X(p_id, 2) / v_weight;
        }
    };

    // With uniform Laplacian, we just need the valence, thus we
    // call query and set oriented to false
    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(
        block,
        shrd_alloc,
        init_lambda,
        [](VertexHandle) { return true; },
        !use_uniform_laplace);
}

int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");
    //RXMeshStatic rx2(STRINGIFY(INPUT_DIR) "sphere3.obj");
    constexpr uint32_t blockThreads = 256;


    // Different attributes used throughout the application
    auto input_coord = rx.get_input_vertex_coordinates();

    // B in CG
    auto B = rx.add_vertex_attribute<float>("B", 3, rxmesh::DEVICE, rxmesh::SoA);
    B->reset(0.0, rxmesh::DEVICE);

    // X in CG (the output)
    auto X = rx.add_vertex_attribute<float>("X", 3, rxmesh::LOCATION_ALL);
    X->copy_from(*input_coord, rxmesh::DEVICE, rxmesh::DEVICE);


    // RXMesh launch box
    LaunchBox<blockThreads> launch_box_init_B;
    LaunchBox<blockThreads> launch_box_matvec;
    
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_matvec,
                          (void*)rxmesh_matvec<float, blockThreads>,
                          false);



    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_init_B,
                          (void*)init_B<float, blockThreads>,
                          true);

    // init kernel to initialize RHS (B)
    init_B<float, blockThreads><<<launch_box_init_B.blocks,
                              launch_box_init_B.num_threads,
                              launch_box_init_B.smem_bytes_dyn>>>(
        rx.get_context(), *X, *B, true);

    
    // CG scalars
    float alpha(0), beta(0), delta_new(0), delta_old(0);


    auto mcf_matvec = [launch_box_matvec, input_coord](
                          rxmesh::RXMeshStatic&                 rx,
                          const rxmesh::VertexAttribute<float>& input,
                          rxmesh::VertexAttribute<float>&       output) {
        rxmesh_matvec<float, blockThreads>
            <<<launch_box_matvec.blocks,
               launch_box_matvec.num_threads,
               launch_box_matvec.smem_bytes_dyn>>>(
                rx.get_context(), *input_coord, input, output, true, 10);
    };


    cg_solver<float>(rx, *X, *B,mcf_matvec,0.000001);
    X->move(rxmesh::DEVICE, rxmesh::HOST);

    rx.get_polyscope_mesh()->updateVertexPositions(*X);


#if USE_POLYSCOPE
    polyscope::show();
#endif
}