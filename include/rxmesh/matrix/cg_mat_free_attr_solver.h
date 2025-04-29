#pragma once
#include "rxmesh/matrix/iterative_solver.h"

#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/reduce_handle.h"

namespace rxmesh {

/**
 * @brief (Un-preconditioned) matrix-free CG that works only when variables
 * are defined on Attributes
 */
template <typename T, typename HandleT>
struct CGMatFreeAttrSolver : public IterativeSolver<T, Attribute<T, HandleT>>
{
    using AttributeT = Attribute<T, HandleT>;

    using MatVecT =
        std::function<void(const AttributeT&, AttributeT&, cudaStream_t)>;

    CGMatFreeAttrSolver(
        RXMeshStatic& rx,
        MatVecT       mat_vec,
        int           unkown_dim,  // 1D, 2D, 3D
        int           max_iter,
        T             abs_tol             = 1e-6,
        T             rel_tol             = 1e-6,
        int           reset_residual_freq = std::numeric_limits<int>::max())
        : IterativeSolver<T, AttributeT>(max_iter, abs_tol, rel_tol),
          m_rx(&rx),
          m_mat_vec(mat_vec),
          alpha(0),
          beta(0),
          delta_new(0),
          delta_old(0),
          m_reset_residual_freq(reset_residual_freq),
          reduce_handle(ReduceHandle<T, HandleT>(rx)),
          S(*rx.add_attribute<T, HandleT>("CG:S", unkown_dim)),
          P(*rx.add_attribute<T, HandleT>("CG:P", unkown_dim)),
          R(*rx.add_attribute<T, HandleT>("CG:R", unkown_dim))
    {
    }

    virtual void pre_solve(const AttributeT& B,
                           AttributeT&       X,
                           cudaStream_t      stream = NULL) override
    {
        S.reset(0.0, rxmesh::DEVICE, stream);
        P.reset(0.0, rxmesh::DEVICE, stream);
        R.reset(0.0, rxmesh::DEVICE, stream);

        // init S
        m_mat_vec(X, S, stream);

        init_PR(B, S, R, P, stream);

        delta_new = reduce_handle.norm2(R, INVALID32, stream);
        delta_new *= delta_new;
    }

    virtual void solve(const AttributeT& B,
                       AttributeT&       X,
                       cudaStream_t      stream = NULL) override
    {
        m_start_residual = delta_new;

        m_iter_taken = 0;

        while (m_iter_taken < m_max_iter) {
            // s = Ap
            m_mat_vec(P, S, stream);

            // alpha = delta_new / <S,P>
            alpha = reduce_handle.dot(S, P, INVALID32, stream);
            alpha = delta_new / alpha;

            // X =  alpha*P + X
            axpy(X, P, alpha, T(1.), stream);

            // reset residual
            if (m_iter_taken > 0 && m_iter_taken % m_reset_residual_freq == 0) {
                // s= Ax
                m_mat_vec(X, S, stream);
                // r = b-s
                subtract(R, B, S, stream);
            } else {
                // r = - alpha*s + r
                axpy(R, S, -alpha, T(1.), stream);
            }

            // delta_old = delta_new
            CUDA_ERROR(cudaStreamSynchronize(stream));
            delta_old = delta_new;

            // delta_new = <r,r>
            delta_new = reduce_handle.norm2(R, INVALID32, stream);
            delta_new *= delta_new;

            // exit if error is getting too low across three coordinates
            if (is_converged(m_start_residual, delta_new)) {
                m_final_residual = delta_new;
                return;
            }

            // beta = delta_new/delta_old
            beta = delta_new / delta_old;

            // p = beta*p + r
            axpy(P, R, T(1.), beta, stream);

            m_iter_taken++;
        }
        m_final_residual = delta_new;
    }

    virtual std::string name() override
    {
        return std::string("CG Matrix Free Attr");
    }

    virtual ~CGMatFreeAttrSolver()
    {
    }


    /**
     * @brief implement Y = alpha*X + beta*Y
     */
    void axpy(AttributeT&       y,
              const AttributeT& x,
              const T           alpha,
              const T           beta,
              cudaStream_t      stream)
    {
        int num_attr = y.get_num_attributes();

        m_rx->for_each<HandleT>(
            DEVICE,
            [y, x, alpha, beta, num_attr] __device__(const HandleT vh) mutable {
                for (int i = 0; i < num_attr; ++i) {
                    y(vh, i) = alpha * x(vh, i) + beta * y(vh, i);
                }
            },
            stream);
    }

    /**
     * @brief implement r = b - s
     */
    void subtract(AttributeT&       r,
                  const AttributeT& b,
                  const AttributeT& s,
                  cudaStream_t      stream)
    {
        int num_attr = r.get_num_attributes();

        m_rx->for_each<HandleT>(
            DEVICE,
            [r, b, s, num_attr] __device__(const HandleT vh) mutable {
                for (int i = 0; i < num_attr; ++i) {
                    r(vh, i) = b(vh, i) - s(vh, i);
                }
            },
            stream);
    }


    /**
     * @brief initialize the residual and the direction in CG method
     */
    void init_PR(const AttributeT& B,
                 const AttributeT& S,
                 AttributeT&       R,
                 AttributeT&       P,
                 cudaStream_t      stream = NULL)
    {
        int num_attr = R.get_num_attributes();

        m_rx->for_each<HandleT>(
            DEVICE,
            [B, S, R, P, num_attr] __device__(const HandleT vh) mutable {
                for (int i = 0; i < num_attr; ++i) {
                    R(vh, i) = B(vh, i) - S(vh, i);
                    P(vh, i) = R(vh, i);
                }
            },
            stream);
    };

   protected:
    RXMeshStatic* m_rx;
    MatVecT       m_mat_vec;
    AttributeT    S, P, R;
    T             alpha, beta, delta_new, delta_old;
    int           m_reset_residual_freq;

    ReduceHandle<T, HandleT> reduce_handle;
};

}  // namespace rxmesh