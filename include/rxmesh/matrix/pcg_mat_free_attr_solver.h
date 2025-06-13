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
        this->S.reset(0.0, rxmesh::DEVICE, stream);
        this->P.reset(0.0, rxmesh::DEVICE, stream);
        this->R.reset(0.0, rxmesh::DEVICE, stream);

        // init S
        // S = Ax
        this->m_mat_vec(X, this->S, stream);

        // init R
        // R = B - S
        init_R(B, this->S, this->R, stream);

        // init P
        // P = inv(M) * R
        m_precond_mat_vec(this->R, this->P, stream);


        this->delta_new = std::abs(
            this->reduce_handle.dot(this->R, this->P, INVALID32, stream));
    }

    virtual void solve(const AttributeT& B,
                       AttributeT&       X,
                       cudaStream_t      stream = NULL) override
    {
        this->m_start_residual = this->delta_new;

        this->m_iter_taken = 0;

        while (this->m_iter_taken < this->m_max_iter) {
            // s = Ap
            this->m_mat_vec(this->P, this->S, stream);

            // alpha = delta_new / <S,P>
            this->alpha =
                this->reduce_handle.dot(this->S, this->P, INVALID32, stream);
            this->alpha = this->delta_new / this->alpha;

            // X =  alpha*P + X
            this->axpy(X, this->P, this->alpha, T(1.), stream);

            // reset residual
            if (this->m_iter_taken > 0 &&
                this->m_iter_taken % this->m_reset_residual_freq == 0) {
                // s= Ax
                this->m_mat_vec(X, this->S, stream);
                // r = b-s
                this->subtract(this->R, B, this->S, stream);
            } else {
                // r = - alpha*s + r
                this->axpy(this->R, this->S, -this->alpha, T(1.), stream);
            }

            // S = inv(M) *R
            m_precond_mat_vec(this->R, this->S, stream);

            // delta_old = delta_new
            CUDA_ERROR(cudaStreamSynchronize(stream));
            this->delta_old = this->delta_new;

            // delta_new = <r,s>
            this->delta_new =
                this->reduce_handle.dot(this->R, this->S, INVALID32, stream);

            // exit if error is getting too low across three coordinates
            if (this->is_converged(this->m_start_residual, this->delta_new)) {
                this->m_final_residual = this->delta_new;
                return;
            }

            // beta = delta_new/delta_old
            this->beta = this->delta_new / this->delta_old;

            // p = beta*p + s
            this->axpy(this->P, this->S, T(1.), this->beta, stream);

            this->m_iter_taken++;
        }
        this->m_final_residual = this->delta_new;
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

        this->m_rx->template for_each<HandleT>(
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