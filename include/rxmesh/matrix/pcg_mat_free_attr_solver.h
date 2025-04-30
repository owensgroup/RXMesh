#pragma once
#include "rxmesh/matrix/cg_mat_free_solver.h"

namespace rxmesh {

/**
 * @brief Jacobi-preconditioned matrix-free CG
 */
template <typename T, typename HandleT>
struct PCGMatFreeAttrSolver : public CGMatFreeAttrSolver<T, HandleT>
{
    using AttributeT = Attribute<T, HandleT>;

    using MatVecT =
        std::function<void(const AttributeT&, AttributeT&, cudaStream_t)>;

    using PrecondMatVecT =
        std::function<void(const AttributeT&, AttributeT&, cudaStream_t)>;

    PCGMatFreeAttrSolver(
        RXMeshStatic&  rx,
        MatVecT        mat_vec,
        PrecondMatVecT precond_mat_vec,
        int            unkown_dim,  // 1D, 2D, 3D
        int            max_iter,
        T              abs_tol             = 1e-6,
        T              rel_tol             = 1e-6,
        int            reset_residual_freq = std::numeric_limits<int>::max())
        : CGMatFreeAttrSolver<T, HandleT>(rx,
                                          mat_vec,
                                          unkown_dim,
                                          max_iter,
                                          abs_tol,
                                          rel_tol,
                                          reset_residual_freq),
          m_precond_mat_vec(precond_mat_vec)
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
        // S = Ax
        m_mat_vec(X, S, stream);

        // init R
        // R = B - S
        init_R(B, S, R, stream);

        // init P
        // P = inv(M) * R
        m_precond_mat_vec(R, P, stream);


        delta_new = std::abs(reduce_handle.dot(R, P, INVALID32, stream));
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

            // S = inv(M) *R
            m_precond_mat_vec(R, S, stream);

            // delta_old = delta_new
            CUDA_ERROR(cudaStreamSynchronize(stream));
            delta_old = delta_new;

            // delta_new = <r,s>
            delta_new = reduce_handle.dot(R, S, INVALID32, stream);

            // exit if error is getting too low across three coordinates
            if (is_converged(m_start_residual, delta_new)) {
                m_final_residual = delta_new;
                return;
            }

            // beta = delta_new/delta_old
            beta = delta_new / delta_old;

            // p = beta*p + s
            axpy(P, S, T(1.), beta, stream);

            m_iter_taken++;
        }
        m_final_residual = delta_new;
    }

    virtual std::string name() override
    {
        return std::string("PCG Matrix Free Attr");
    }

    virtual ~PCGMatFreeAttrSolver()
    {
    }


    /**
     * @brief initialize the residual
     */
    void init_R(const AttributeT& B,
                const AttributeT& S,
                AttributeT&       R,
                cudaStream_t      stream = NULL)
    {
        int num_attr = R.get_num_attributes();

        m_rx->for_each<HandleT>(
            DEVICE,
            [B, S, R, num_attr] __device__(const HandleT vh) mutable {
                for (int i = 0; i < num_attr; ++i) {
                    R(vh, i) = B(vh, i) - S(vh, i);
                }
            },
            stream);
    };

   protected:
    PrecondMatVecT m_precond_mat_vec;
};

}  // namespace rxmesh