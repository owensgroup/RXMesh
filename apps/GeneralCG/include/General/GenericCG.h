#pragma once
#include "rxmesh/reduce_handle.h"


template <typename T>
using MatVecOp = std::function<void(rxmesh::RXMeshStatic&,
                                    const rxmesh::VertexAttribute<T>&,
                                    rxmesh::VertexAttribute<T>&)>;

template<typename T>
class CG
{
    MatVecOp<T> matvec_op;
    int         max_iter;
    float       time_step;
    float                      tolerance;
    rxmesh::VertexAttribute<T> X;
    rxmesh::VertexAttribute<T> B;

public:
    CG(rxmesh::VertexAttribute<T>& _X,
       rxmesh::VertexAttribute<T>& _B,
       MatVecOp<T>                 mat_vec_op,
       int                         iterations,
       float                       timestep, 
        float tol)
        :
          X(_X),
          B(_B),
          matvec_op(mat_vec_op),
          max_iter(iterations),
          time_step(timestep),
          tolerance(tol)
    {

    }

    void solve(RXMeshStatic& rx)
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
        T        delta_0           = delta_new;
        T        delta_old         = 0;
        uint32_t num_cg_iter_taken = 0;

        while (num_cg_iter_taken < max_iter) {
            // Compute S = A * P
            matvec_op(rx, P, S);

            // CG updates
            T alpha = delta_new / reduce_handle.dot(S, P);
            axpy(rx, X, P, alpha, 1.0);
            axpy(rx, R, S, -alpha, 1.0);

            CUDA_ERROR(cudaStreamSynchronize(0));
            delta_old = delta_new;


            // delta_new = <r,r>
            delta_new = reduce_handle.norm2(R);
            delta_new *= delta_new;

            CUDA_ERROR(cudaStreamSynchronize(0));

            if (delta_new < tolerance * tolerance * delta_0)
                break;

            T beta = delta_new / delta_old;
            axpy(rx, P, R, 1.0, beta);

            num_cg_iter_taken++;
            CUDA_ERROR(cudaStreamSynchronize(0));
        }
    }

    void init_PR(rxmesh::RXMeshStatic&             rx,
                 const rxmesh::VertexAttribute<T>& B,
                 const rxmesh::VertexAttribute<T>& S,
                 rxmesh::VertexAttribute<T>&       R,
                 rxmesh::VertexAttribute<T>&       P)
    {
        rx.for_each_vertex(
            rxmesh::DEVICE,
            [B, S, R, P] __device__(const rxmesh::VertexHandle vh) {
                R(vh, 0) = B(vh, 0) - S(vh, 0);
                R(vh, 1) = B(vh, 1) - S(vh, 1);
                R(vh, 2) = B(vh, 2) - S(vh, 2);

                P(vh, 0) = R(vh, 0);
                P(vh, 1) = R(vh, 1);
                P(vh, 2) = R(vh, 2);
            });
    }

    
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
};