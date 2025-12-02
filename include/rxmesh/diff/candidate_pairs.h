#pragma once

#include "rxmesh/matrix/dense_matrix.h"

namespace rxmesh {

/**
 * @brief Storing a pair of candidate contact pairs using their handles
 */
template <typename HandleT0, typename HandleT1, typename HessMatT>
struct CandidatePairs
{
    using IndexT = int;

    template <typename T, int V, typename O, bool W>
    friend struct DiffScalarProblem;

    using PairT = std::pair<HandleT0, HandleT1>;

    __device__                 CandidatePairs(const CandidatePairs&) = default;
    __device__                 CandidatePairs(CandidatePairs&&)      = default;
    __device__ CandidatePairs& operator=(const CandidatePairs&)      = default;
    __device__ CandidatePairs& operator=(CandidatePairs&&)           = default;
    __device__                 CandidatePairs()                      = default;

    virtual ~CandidatePairs()
    {
    }

    /**
     * @brief allocate memory with max_capacity number of pairs. Allocation
     * happens once and we don't support dynamic allocation, i.e., insertion
     * fail if the number of candidates exceeds max_capacity.
     * max_capactiy indicates the expected max number of new pairs. Each pair
     * will insert two (because of hessian symmetry) of blocks of size
     * variable_dim^2 into the hessian. Note that there are also other two
     * blocks of the same size along the diagonal that a contact pair will
     * contribute to -- but this is already allocated in the hessian matrix
     */
    __host__ CandidatePairs(int            max_capacity,
                            HessMatT&      hess,
                            const Context& ctx)
        : m_hess(hess),
          m_variable_dim(hess.K_),
          m_pairs_id(DenseMatrix<IndexT, Eigen::ColMajor>(
              max_capacity * m_variable_dim * m_variable_dim * 2,
              2)),
          m_pairs_handle(DenseMatrix<PairT, Eigen::ColMajor>(max_capacity, 1)),
          m_current_num_pairs(DenseMatrix<int>(1, 1)),
          m_current_num_index(DenseMatrix<int>(1, 1)),
          m_context(ctx)
    {
        reset();
    }


    /**
     * @brief Insert a candidate pair and return true if it succeeded. Otherwise
     * return false, i.e., in case of exceeding the max capacity.
     * Here, we insert two things 1) the handles in a buffer to iterate over
     * later 2) the corresponding new indices of the local Hessian block in the
     * sparse Hessian matrix that these two (possibly) contacting pair
     * generates.
     * We always insert new handles (so long as the buffer is not overflown).
     * For 2) we check if the block is already allocated. If not, we add the
     * indices to another buffer which is later used in the update_hessian in
     * the Problem
     */
    __device__ __host__ bool insert(const HandleT0& c0, const HandleT1& c1)
    {

        // first check if we have a space
        /// if we have a space to insert new pairs

        auto add_candidate_with_indices = [&](int id) {
            // add the handles
            m_pairs_handle(id).first  = c0;
            m_pairs_handle(id).second = c1;

            // add the indices
            id *= m_variable_dim * m_variable_dim * 2;

            for (int i = 0; i < m_variable_dim; ++i) {
                for (int j = 0; j < m_variable_dim; ++j) {

                    int r_id = m_context.linear_id(c0) * m_variable_dim + i;
                    int c_id = m_context.linear_id(c1) * m_variable_dim + j;

                    m_pairs_id(id, 0) = r_id;
                    m_pairs_id(id, 1) = c_id;
                    id++;

                    m_pairs_id(id, 0) = c_id;
                    m_pairs_id(id, 1) = r_id;
                    id++;
                }
            }
        };


        auto add_candidate = [&](int id) {
            // add the handles
            m_pairs_handle(id).first  = c0;
            m_pairs_handle(id).second = c1;
        };


#ifdef __CUDA_ARCH__
        int id = ::atomicAdd(m_current_num_pairs.data(DEVICE), 1);
        if (id < m_pairs_handle.rows()) {            
            if (m_hess.is_non_zero(c0, c1)) {
                add_candidate(id);
            } else {
                ::atomicAdd(m_current_num_index.data(DEVICE),
                            m_variable_dim * m_variable_dim * 2);
                add_candidate_with_indices(id);
            }
            return true;
        } else {
            ::atomicAdd(m_current_num_pairs.data(DEVICE), -1);
            return false;
        }
#else
        if (m_current_num_pairs(0) < m_pairs_handle.rows()) {
            int id = m_current_num_pairs(0);
            m_current_num_pairs(0)++;            
            if (m_hess.is_non_zero(c0, c1)) {
                add_candidate(id);
            } else {
                m_current_num_index(0) += m_variable_dim * m_variable_dim * 2;
                add_candidate_with_indices(id);
            }
            return true;
        } else {
            return false;
        }
#endif
    }

    /**
     * @brief return a candidate pair using their Id
     */
    __device__ __host__ const PairT& get_pair(int id) const
    {
        assert(id < m_pairs_handle.rows());
        assert(id < m_current_num_pairs(0));
        assert(m_current_num_pairs(0) > 0);

        return m_pairs_handle(id, 0);
    }

    /**
     * @brief return the current number of the candidates (i.e., number of
     * handles, not indices)
     */
    __device__ __host__ int num_pairs()
    {
#ifdef __CUDA_ARCH__
        return m_current_num_pairs(0);
#else
        m_current_num_pairs.move(DEVICE, HOST);
        return m_current_num_pairs(0);
#endif
    }


    /**
     * @brief return the current number of the new indices
     */
    __device__ __host__ int num_index()
    {
#ifdef __CUDA_ARCH__
        return m_current_num_index(0);
#else
        m_current_num_index.move(DEVICE, HOST);
        return m_current_num_index(0);
#endif
    }

    /**
     * @brief return the max number of candidate pairs that can be stored (i.e.,
     * number of handles, not indices)
     */
    __device__ __host__ int capacity() const
    {
        return m_pairs_handle.rows();
    }

    /**
     * @brief reset the number of candidate pairs, i.e., make the size equal 0
     */
    __device__ __host__ void reset()
    {
        m_current_num_pairs(0) = 0;
        m_current_num_index(0) = 0;

#ifndef __CUDA_ARCH__
        m_current_num_pairs.move(HOST, DEVICE);
        m_current_num_index.move(HOST, DEVICE);
#endif
    }

    /**
     * @brief release the memory in both host and device
     * @return
     */
    __host__ void release()
    {
        m_pairs_id.release();
        m_pairs_handle.release();
        m_current_num_pairs.release();
        m_current_num_index.release();
    }


   private:
    DenseMatrix<IndexT, Eigen::ColMajor> m_pairs_id;
    DenseMatrix<PairT, Eigen::ColMajor>  m_pairs_handle;

    // track the number of pairs (not the number of indices)
    DenseMatrix<int> m_current_num_pairs;
    DenseMatrix<int> m_current_num_index;

    // TODO remove this and use m_hess.k_
    int m_variable_dim;

    HessMatT m_hess;

    Context m_context;
};

template <typename HessMatT>
using CandidatePairsVV = CandidatePairs<VertexHandle, VertexHandle, HessMatT>;

}  // namespace rxmesh