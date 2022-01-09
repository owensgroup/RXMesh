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
                            const std::vector<DynOp> dyn_op,
                            LaunchBox<blockThreads>& launch_box,
                            const void*              kernel,
                            const bool               oriented = false) const
    {
        // If there is no query operations, we at least need to load the patch
        // EV and FE. Otherwise, we need to make sure that we at least can load
        // EV and FE after performing the queries (TODO)

        launch_box.blocks         = this->m_num_patches;
        launch_box.smem_bytes_dyn = 0;
        for (auto o : dyn_op) {
            launch_box.smem_bytes_dyn =
                std::max(launch_box.smem_bytes_dyn,
                         this->template calc_shared_memory<blockThreads>(o));
        }

        for (auto o : op) {
            launch_box.smem_bytes_dyn = std::max(
                launch_box.smem_bytes_dyn,
                this->template RXMeshStatic::calc_shared_memory<blockThreads>(
                    o, oriented));
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

   private:
    template <uint32_t blockThreads>
    size_t calc_shared_memory(const DynOp op) const
    {
        if (op == DynOp::EdgeFlip && !this->is_edge_manifold()) {
            RXMESH_ERROR(
                "RXMeshDynamic::calc_shared_memory() edge flips is only "
                "supported on manifold mesh.");
        }

        size_t dynamic_smem = 0;
        if (op == DynOp::EdgeFlip) {
            // load FE, than transpose it into EF, then update FE. Thus, we need
            // to have both in memory at one point. Then, load EV and update it
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t);
            dynamic_smem +=
                std::max(2 * this->m_max_edges_per_patch * sizeof(uint16_t),
                         3 * this->m_max_faces_per_patch * sizeof(uint16_t));
        }

        return dynamic_smem;
    }
};
}  // namespace rxmesh