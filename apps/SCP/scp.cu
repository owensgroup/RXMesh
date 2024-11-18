#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;


template <uint32_t blockThreads>
__global__ static void area_term(const Context              context,
                                 const VertexAttribute<int> v_bd,
                                 SparseMatrix<cuComplex>    E)
{
    using namespace rxmesh;

    auto compute = [&](FaceHandle& face_id, const VertexIterator& iter) {
        assert(iter.size() == 3);

        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            if (v_bd(iter[i]) == 1 && v_bd(iter[j]) == 1) {
                ::atomicAdd(&E(iter[i], iter[j]).y, 0.25f);
                ::atomicAdd(&E(iter[j], iter[i]).y, -0.25f);
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);

    ShmemAllocator shrd_alloc;

    query.dispatch<Op::FV>(block, shrd_alloc, compute);
}

template <uint32_t blockThreads>
__global__ static void conformal_energy(const Context                context,
                                        const VertexAttribute<float> coord,
                                        SparseMatrix<cuComplex>      E)
{
    using namespace rxmesh;

    auto compute = [&](EdgeHandle& p0, const VertexIterator& iter) {
        auto weight = [&](const vec3<float>& P,
                          const vec3<float>& Q,
                          const vec3<float>& O) {
            const vec3<float> l1 = O - Q;
            const vec3<float> l2 = O - P;

            float w = glm::dot(l1, l2) / glm::length(glm::cross(l1, l2));
            return std::max(0.f, w);
        };

        VertexHandle p = iter[0];
        VertexHandle q = iter[2];

        VertexHandle o0 = iter[1];
        VertexHandle o1 = iter[3];

        assert(p.is_valid() && q.is_valid());
        assert(o0.is_valid() || o1.is_valid());

        const vec3<float> P = coord.to_glm<3>(p);
        const vec3<float> Q = coord.to_glm<3>(q);

        float coef = 0;

        if (o0.is_valid()) {
            const vec3<float> O0 = coord.to_glm<3>(o0);
            coef += weight(P, Q, O0);
        }
        if (o1.is_valid()) {
            const vec3<float> O1 = coord.to_glm<3>(o1);
            coef += weight(P, Q, O1);
        }

        coef *= 0.25f;

        // off-diagonal
        E(p, q).x = -coef;
        E(q, p).x = -coef;

        // diagonal-diagonal
        ::atomicAdd(&E(p, p).x, coef);
        ::atomicAdd(&E(q, q).x, coef);
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);

    ShmemAllocator shrd_alloc;

    query.dispatch<Op::EVDiamond>(block, shrd_alloc, compute);
}

int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");

    constexpr uint32_t CUDABlockSize = 256;

    auto v_bd = *rx.add_vertex_attribute<int>("vBoundary", 1);
    rx.get_boundary_vertices(v_bd);

    auto coords = *rx.get_input_vertex_coordinates();

    auto uv = *rx.add_vertex_attribute<float>("uv", 3);

    DenseMatrix<cuComplex> uv_mat(rx, rx.get_num_vertices(), 1);


    // calc number of boundary vertices
    ReduceHandle rh(v_bd);
    int          num_bd_vertices = rh.reduce(v_bd, cub::Sum(), 0);

    // compute conformal energy matrix Lc
    SparseMatrix<cuComplex> Lc(rx);
    Lc.reset(make_cuComplex(0.f, 0.f), LOCATION_ALL);
    rxmesh::LaunchBox<CUDABlockSize> lb;
    rx.prepare_launch_box(
        {Op::EVDiamond}, lb, (void*)conformal_energy<CUDABlockSize>);

    conformal_energy<CUDABlockSize>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), coords, Lc);

    // area term
    rx.prepare_launch_box({Op::FV}, lb, (void*)area_term<CUDABlockSize>);

    area_term<CUDABlockSize><<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
        rx.get_context(), v_bd, Lc);

    // Compute B and eb matrix
    DenseMatrix<cuComplex> eb(rx, rx.get_num_vertices(), 1);
    eb.reset(make_cuComplex(0.f, 0.f), LOCATION_ALL);

    SparseMatrix<cuComplex> B(rx);
    B.reset(make_cuComplex(0.f, 0.f), LOCATION_ALL);

    float nb = 1.f / std::sqrt(float(num_bd_vertices));
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [B, eb, v_bd, nb] __device__(const rxmesh::VertexHandle vh) mutable {
            eb(vh, 0) = make_cuComplex((float)v_bd(vh, 0) * (float)nb, 0.0f);
            B(vh, vh) = make_cuComplex((float)v_bd(vh, 0), 0.0f);
        });

    // temp mat needed for the power method
    DenseMatrix<cuComplex> T1(rx, rx.get_num_vertices(), 1);
    T1.reset(make_cuComplex(0.f, 0.f), LOCATION_ALL);


    // fill-in the init solution of the eigen vector with random values
    uv_mat.fill_random();

    // factorize the matrix
    Lc.pre_solve(rx, Solver::CHOL, PermuteMethod::NSTDIS);

    // the power method
    int iterations = 32;

    float prv_norm = std::numeric_limits<float>::max();

    for (int i = 0; i < iterations; i++) {
        cuComplex T2 = eb.dot(uv_mat);
        rx.for_each_vertex(rxmesh::DEVICE,
                           [eb, T2, T1, uv_mat, B] __device__(
                               const rxmesh::VertexHandle vh) mutable {
                               T1(vh, 0) =
                                   cuCsubf(cuCmulf(B(vh, vh), uv_mat(vh, 0)),
                                           cuCmulf(eb(vh, 0), T2));
                           });

        Lc.solve(T1, uv_mat);

        float norm = uv_mat.norm2();

        uv_mat.multiply(1.0f / norm);

        if (std::abs(prv_norm - norm) < 0.0001) {
            break;
        }
        prv_norm = norm;
    }

    // convert from matrix format to attributes
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [uv_mat, uv] __device__(const rxmesh::VertexHandle vh) mutable {
            uv(vh, 0) = uv_mat(vh, 0).x;
            uv(vh, 1) = uv_mat(vh, 0).y;
        });

    uv.move(DEVICE, HOST);

    // normalize the uv coordinates
    glm::vec3 lower;
    glm::vec3 upper;

    ReduceHandle rrh(uv);

    lower[0] = rrh.reduce(uv, cub::Min(), std::numeric_limits<float>::max(), 0);
    lower[1] = rrh.reduce(uv, cub::Min(), std::numeric_limits<float>::max(), 1);
    lower[2] = rrh.reduce(uv, cub::Min(), std::numeric_limits<float>::max(), 2);

    upper[0] = rrh.reduce(uv, cub::Max(), std::numeric_limits<float>::min(), 0);
    upper[1] = rrh.reduce(uv, cub::Max(), std::numeric_limits<float>::min(), 1);
    upper[2] = rrh.reduce(uv, cub::Max(), std::numeric_limits<float>::min(), 2);

    upper -= lower;
    float s = std::max(upper[0], upper[1]);

    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        uv(vh, 0) -= upper[0];
        uv(vh, 1) -= upper[1];
        uv(vh, 0) /= s;
        uv(vh, 1) /= s;
    });

    // add uv to Polyscope
    rx.get_polyscope_mesh()->addVertexParameterizationQuantity("uv", uv);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("vBoundary", v_bd);


#if USE_POLYSCOPE
    polyscope::show();
#endif

    uv_mat.release();
    Lc.release();
    eb.release();
    B.release();
    T1.release();

    return 0;
}