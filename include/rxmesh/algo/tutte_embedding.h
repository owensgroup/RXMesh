#pragma once

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_util.cuh"

namespace rxmesh {

namespace detail {

template <typename T, typename BoundaryT, int blockThreads>
__global__ static void next_vertex(const Context                    context,
                                   const VertexAttribute<T>         coordinates,
                                   const VertexAttribute<BoundaryT> v_boundary,
                                   const VertexHandle               current_v,
                                   VertexHandle*                    next_v,
                                   VertexAttribute<T>               uv)
{
    if (current_v.patch_id() != blockIdx.x) {
        return;
    }
    auto func = [&](const VertexHandle& vh, const VertexIterator& iter) {
        if (vh == current_v) {
            for (int i = 0; i < iter.size(); ++i) {


                if (v_boundary(iter[i]) && uv(iter[i], 1) == T(0)) {

                    const vec3<T> c0 = coordinates.template to_glm<3>(iter[i]);
                    const vec3<T> c1 =
                        coordinates.template to_glm<3>(current_v);

                    T dist = glm::distance(c0, c1);

                    uv(iter[i], 0) = uv(current_v, 0) + dist;
                    uv(iter[i], 1) = T(1.0);

                    (*next_v) = iter[i];
                    break;
                }
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);

    ShmemAllocator shrd_alloc;

    query.dispatch<Op::VV>(block, shrd_alloc, func);
}


template <typename T, typename BoundaryT, int blockThreads>
__global__ static void setup_L(const Context                    context,
                               const VertexAttribute<T>         coordinates,
                               const VertexAttribute<BoundaryT> v_boundary,
                               SparseMatrix<T>                  L)
{

    auto func = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        // Edge: iter[0]-iter[2]
        // Opposite vertices: iter[1] and iter[3]

        VertexHandle p = iter[0];
        VertexHandle r = iter[2];
        VertexHandle q = iter[1];
        VertexHandle s = iter[3];

        assert(p.is_valid());
        assert(r.is_valid());

        // if boundary edge
        if (!q.is_valid() || !s.is_valid()) {
            // other edges/threads might be writting this as well!

            assert(v_boundary(p) && v_boundary(r));

            L(p, p) = T(1);
            L(r, r) = T(1);
        } else {

            // T cotan = edge_cotan_weight(coordinates.to_glm<3>(p),
            //                             coordinates.to_glm<3>(r),
            //                             coordinates.to_glm<3>(q),
            //                             coordinates.to_glm<3>(s));

            T cotan = T(1);

            if (!v_boundary(p)) {
                L(p, r) = -cotan;
                ::atomicAdd(&L(p, p), cotan);
            }

            if (!v_boundary(r)) {
                L(r, p) = -cotan;
                ::atomicAdd(&L(r, r), cotan);
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);

    ShmemAllocator shrd_alloc;

    query.dispatch<Op::EVDiamond>(block, shrd_alloc, func);
}

template <typename T, typename BoundaryT>
inline void map_vertices_to_circle(RXMeshStatic&             rx,
                                   const VertexAttribute<T>& coordinates,
                                   const VertexAttribute<BoundaryT>& v_boundary,
                                   VertexAttribute<T>&               uv)
{

    VertexHandle current_v;
    VertexHandle initial_v;
    VertexHandle last_v;

    uint32_t num_boundary_vertices = 0;

    rx.for_each_vertex(
        HOST,
        [&](const VertexHandle& vh) {
            if (v_boundary(vh)) {
                if (!initial_v.is_valid()) {
                    initial_v = vh;
                }
                num_boundary_vertices++;
            }
            uv(vh, 0) = T(0);
            uv(vh, 1) = T(0);
        },
        NULL,
        false);

    // repupose uv to temporarly store the length and if this vertex is visited
    uv(initial_v, 0) = T(0);
    uv(initial_v, 1) = T(1);
    uv.move(HOST, DEVICE);

    current_v = initial_v;

    uint32_t num = 1;

    constexpr uint32_t blockThreads = 256;

    VertexHandle*      d_next_v = nullptr;
    const VertexHandle h_next_v = VertexHandle();
    CUDA_ERROR(cudaMalloc((void**)&d_next_v, sizeof(VertexHandle)));
    CUDA_ERROR(cudaMemcpy(
        d_next_v, &h_next_v, sizeof(VertexHandle), cudaMemcpyHostToDevice));


    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box(
        {Op::VV}, lb, (void*)next_vertex<T, BoundaryT, blockThreads>);

    // TODO this is very inefficient. We only update one vertex per iteration,
    // i.e., only a single thread do usful work here.
    while (num < num_boundary_vertices) {
        next_vertex<T, BoundaryT, blockThreads>
            <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(rx.get_context(),
                                                               coordinates,
                                                               v_boundary,
                                                               current_v,
                                                               d_next_v,
                                                               uv);
        CUDA_ERROR(cudaMemcpy(&current_v,
                              d_next_v,
                              sizeof(VertexHandle),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(
            d_next_v, &h_next_v, sizeof(VertexHandle), cudaMemcpyHostToDevice));
        if (current_v.is_valid()) {
            last_v = current_v;
        }
        num++;
    }

    uv.move(DEVICE, HOST);

    const vec3<T> c0 = coordinates.template to_glm<3>(initial_v);
    const vec3<T> c1 = coordinates.template to_glm<3>(last_v);

    T total_len = uv(last_v, 0) + glm::distance(c0, c1);

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
        if (v_boundary(vh)) {
            T frac = uv(vh, 0) * T(2.0) * glm::pi<T>() / total_len;

            uv(vh, 0) = std::cos(frac);
            uv(vh, 1) = std::sin(frac);
        }
    });

    uv.move(DEVICE, HOST);


    GPU_FREE(d_next_v);
}

template <typename T, typename BoundaryT>
inline void harmonic(RXMeshStatic&                     rx,
                     const VertexAttribute<T>&         coordinates,
                     const VertexAttribute<BoundaryT>& v_boundary,
                     VertexAttribute<T>&               uv,
                     SparseMatrix<T>&                  L)
{
    auto rhs = *uv.to_matrix();
    auto sol =
        DenseMatrix<T>(rx, rx.get_num_vertices(), uv.get_num_attributes());

    sol.reset(T(0), DEVICE);
    L.reset(T(0), DEVICE);

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box(
        {Op::EVDiamond}, lb, (void*)setup_L<T, BoundaryT, blockThreads>);


    setup_L<T, BoundaryT, blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), coordinates, v_boundary, L);

    L.move(DEVICE, HOST);
    rhs.move(DEVICE, HOST);
    sol.move(DEVICE, HOST);

    L.solve(rhs, sol, Solver::LU, PermuteMethod::NSTDIS);

    // sol.move(DEVICE, HOST);

    uv.from_matrix(&sol);

    uv.move(HOST, DEVICE);

    sol.release();
    rhs.release();
}
}  // namespace detail

template <typename T, typename BoundaryT>
inline void tutte_embedding(RXMeshStatic&               rx,
                            const VertexAttribute<T>&   coordinates,
                            VertexAttribute<BoundaryT>& v_boundary,
                            VertexAttribute<T>&         uv,
                            SparseMatrix<T>&            L)
{
    detail::map_vertices_to_circle(rx, coordinates, v_boundary, uv);

    detail::harmonic(rx, coordinates, v_boundary, uv, L);
}

template <typename T, typename BoundaryT>
inline void tutte_embedding(RXMeshStatic&                     rx,
                            const VertexAttribute<T>&         coordinates,
                            const VertexAttribute<BoundaryT>& v_boundary,
                            VertexAttribute<T>&               uv)
{
    SparseMatrix<T> L(rx);

    detail::map_vertices_to_circle(rx, coordinates, v_boundary, uv);

    detail::harmonic(rx, coordinates, v_boundary, uv, L);

    L.release();
}

template <typename T>
inline void tutte_embedding(RXMeshStatic&             rx,
                            const VertexAttribute<T>& coordinates,
                            VertexAttribute<T>&       uv)
{
    auto v_boundary = *rx.add_vertex_attribute<bool>("rx:vBnd", 1);

    SparseMatrix<T> L(rx);

    rx.get_boundary_vertices(v_boundary);

    tutte_embedding(rx, coordinates, v_boundary, uv, L);

    L.release();
}

}  // namespace rxmesh