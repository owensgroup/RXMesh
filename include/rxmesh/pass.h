#pragma once
#include <stdint.h>

#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/macros.h"

namespace rxmesh {

enum class Status : uint8_t
{
    UNSEEN = 0,  // means we have not tested/touched it before
    SKIP   = 1,  // means we have tested it and it is okay to skip
    UPDATE = 2,  // means we should update it i.e., we have tested it before
    ADDED  = 3,  // means it has been added to during the split/flip/collapse
};

/**
 * @brief record a pass over all mesh elements of certain type
 */
template <typename HandleT>
struct Pass
{
    using HandleType = HandleT;

    __host__ Pass(RXMeshDynamic& rx)
    {
        CUDA_ERROR(cudaMalloc((void**)&m_d_counter, sizeof(uint32_t)));
        m_status = *rx.add_attribute<Status, HandleT>("rx:status", 1);
        m_status.reset(Status::UNSEEN, DEVICE);
    };


    __device__ __host__       Pass()                  = default;
    __device__ __host__       Pass(const Pass& other) = default;
    __device__ __host__       Pass(Pass&&)            = default;
    __device__ __host__ Pass& operator=(const Pass&)  = default;
    __device__ __host__ Pass& operator=(Pass&&)       = default;
    __device__                __host__ ~Pass()        = default;

    __host__ uint32_t remaining_items(const RXMeshDynamic& rx)
    {
        using namespace rxmesh;

        // if there is at least one edge that is UNSEEN, then we are not done
        // yet
        CUDA_ERROR(cudaMemset(m_d_counter, 0, sizeof(uint32_t)));

        rx.for_each<HandleT>(
            DEVICE,
            [status      = *m_status,
             m_d_counter = d_counter] __device__(const HandleT eh) mutable {
                if (status(eh) == Status::UNSEEN ||
                    status(eh) == Status::UPDATE) {
                    ::atomicAdd(d_counter, 1);
                }
            });

        uint32_t h_counter;

        CUDA_ERROR(cudaMemcpy(
            &h_counter, m_d_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        return h_counter;
    }

    /**
     * @brief reset the status UNSEEN
     * @return
     */
    __host__ void reset()
    {
        m_status.reset(Status::UNSEEN, DEVICE);
    }

    /**
     * @brief set the status of a given mesh element
     * @return
     */
    __host__ __device__ void set(const HandleT& h, Status status)
    {
        m_status(h) = status;
    }

    /**
     * @brief get the status of a given mesh element
     */
    __host__ __device__ Status get(const HandleT& h)
    {
        return m_status(h);
    }

    /**
     * @brief return status attribute
     */
    __host__ __device__ Attribute<Status, HandleT>& get_status_attribute()
    {
        return m_status;
    }

    /**
     * @brief free internal memory
     */
    __host__ void release()
    {
        GPU_FREE(m_d_counter);
    }

    Attribute<Status, HandleT> m_status;
    uint32_t*                  m_d_counter;
};

}  // namespace rxmesh