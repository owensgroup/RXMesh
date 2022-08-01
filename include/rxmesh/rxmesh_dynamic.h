#pragma once
#include "rxmesh/rxmesh_static.h"

namespace rxmesh {

class RXMeshDynamic : public RXMeshStatic
{
   public:
    RXMeshDynamic(const RXMeshDynamic&) = delete;

    /**
     * @brief Constructor using path to obj file
     * @param file_path path to an obj file
     * @param quite run in quite mode
     */
    RXMeshDynamic(const std::string file_path, const bool quite = false)
        : RXMeshStatic(file_path, quite)
    {
    }

    /**
     * @brief Constructor using triangles and vertices
     * @param fv Face incident vertices as read from an obj file
     * @param quite run in quite mode
     */
    RXMeshDynamic(std::vector<std::vector<uint32_t>>& fv,
                  const bool                          quite = false)
        : RXMeshStatic(fv, quite)
    {
    }

    /**
     * @brief populate the launch_box with grid size and dynamic shared memory
     * needed for a kernel that may use dynamic and query operations
     * @param op List of query operations done inside the kernel
     * @param launch_box input launch box to be populated
     * @param kernel The kernel to be launched
     * @param oriented if the query is oriented. Valid only for Op::VV queries
     */
    template <uint32_t blockThreads>
    void prepare_launch_box(const std::vector<Op>    op,
                            LaunchBox<blockThreads>& launch_box,
                            const void*              kernel,
                            const bool               oriented = false) const
    {

        launch_box.blocks         = this->m_num_patches;
        launch_box.smem_bytes_dyn = 0;


        for (auto o : op) {
            launch_box.smem_bytes_dyn =
                std::max(launch_box.smem_bytes_dyn,
                         this->calc_shared_memory<blockThreads>(o, oriented));
        }

        // For dynamic changes we load EV and FE
        launch_box.smem_bytes_dyn =
            std::max(launch_box.smem_bytes_dyn,
                     3 * this->m_max_faces_per_patch * sizeof(uint16_t) +
                         2 * this->m_max_edges_per_patch * sizeof(uint16_t) +
                         2 * ShmemAllocator::default_alignment);

        // cavity ID of fake deleted elements
        launch_box.smem_bytes_dyn +=
            this->m_max_vertices_per_patch * sizeof(uint16_t) +
            this->m_max_edges_per_patch * sizeof(uint16_t) +
            this->m_max_faces_per_patch * sizeof(uint16_t) +
            3 * ShmemAllocator::default_alignment;

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
                            blockThreads,
                            kernel);
    }

    virtual ~RXMeshDynamic() = default;

    /**
     * @brief Validate the topology information stored in RXMesh. All checks are
     * done on the information stored on the GPU memory and thus all checks are
     * done on the GPU
     * @return true in case all information stored are valid
     */
    bool validate();

    /**
     * @brief update the host side. Use this function to update the host side
     * after performing (dynamic) updates on the GPU. This function may
     * re-allocates the host side memory buffers in case it is not enough (e.g.,
     * after performing mesh refinement on the GPU)
     */
    void update_host();
};
}  // namespace rxmesh