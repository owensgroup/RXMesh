#pragma once
#include "rxmesh/rxmesh_static.h"

namespace rxmesh {

class RXMeshDynamic : public RXMeshStatic
{
   public:
    RXMeshDynamic(const RXMeshDynamic&) = delete;

    /**
     * @brief Main constructor used to initialize internal member variables
     * @param fv Face incident vertices as read from an obj file
     * @param quite run in quite mode
     */
    RXMeshDynamic(std::vector<std::vector<uint32_t>>& fv,
                  const bool                          quite = false)
        : RXMeshStatic(fv, quite)
    {
    }


    template <uint32_t blockThreads>
    void prepare_launch_box(const std::vector<Op>    op,
                            LaunchBox<blockThreads>& launch_box,
                            const void*              kernel,
                            const bool               oriented = false) const
    {
        // If there is no query operations, we at least need to load the patch
        // EV and FE. Otherwise, we need to make sure that we at least can load
        // EV and FE after performing the queries (TODO)

        launch_box.blocks         = this->m_num_patches;
        launch_box.smem_bytes_dyn = 0;

        if (op.empty()) {
            // load FE and EV
            launch_box.smem_bytes_dyn =
                3 * this->m_max_faces_per_patch * sizeof(uint16_t);
            launch_box.smem_bytes_dyn +=
                2 * this->m_max_edges_per_patch * sizeof(uint16_t);
        } else {
            RXMESH_ERROR(
                "RXMeshDynamic::prepare_launch_box() doing query with updates "
                "is not supported yet!");
        }

        if (!this->m_quite) {
            RXMESH_TRACE(
                "RXMeshDynamic::calc_shared_memory() launching {} blocks with "
                "{} threads on the device",
                launch_box.blocks,
                blockThreads);
        }

        check_shared_memory(launch_box.smem_bytes_dyn,
                            launch_box.smem_bytes_static,
                            launch_box.num_registers_per_thread,
                            kernel);
    }

    virtual ~RXMeshDynamic() = default;
};
}  // namespace rxmesh