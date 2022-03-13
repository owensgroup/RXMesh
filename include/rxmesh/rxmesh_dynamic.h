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
     * @param dyn_op List of dynamic update done inside the kernel
     * @param launch_box input launch box to be populated
     * @param kernel The kernel to be launched
     * @param oriented if the query is oriented. Valid only for Op::VV queries
     */
    template <uint32_t blockThreads>
    void prepare_launch_box(const std::vector<Op>    op,
                            const std::vector<DynOp> dyn_op,
                            LaunchBox<blockThreads>& launch_box,
                            const void*              kernel,
                            const bool               oriented = false) const
    {
        static_assert(
            blockThreads && ((blockThreads & (blockThreads - 1)) == 0),
            " RXMeshDynamic::prepare_launch_box() CUDA block size should be of "
            "power 2");


        launch_box.blocks         = this->m_num_patches;
        launch_box.smem_bytes_dyn = 0;
        for (auto o : dyn_op) {
            launch_box.smem_bytes_dyn =
                std::max(launch_box.smem_bytes_dyn,
                         this->template calc_shared_memory<blockThreads>(o));
        }

        for (auto o : op) {
            launch_box.smem_bytes_dyn =
                std::max(launch_box.smem_bytes_dyn,
                         this->RXMeshStatic::calc_shared_memory<blockThreads>(
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

    /**
     * @brief Validate the topology information stored in RXMesh. All checks are
     * done on the information stored on the GPU memory and thus all checks are
     * done on the GPU
     * @return true in case all information stored are valid
     */
    bool validate();

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
            // load FE, then transpose it into EF, then update FE. Thus, we need
            // to have both in memory at one point. Then, load EV and update it
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t);
            dynamic_smem += 2 * this->m_max_edges_per_patch * sizeof(uint16_t);
            dynamic_smem += DIVIDE_UP(this->m_max_faces_per_patch -
                                          this->m_max_not_owned_faces,
                                      2) *
                            2 * sizeof(uint16_t);
        }

        if (op == DynOp::DeleteFace) {
            // load FE only
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t);
        }

        if (op == DynOp::DeleteEdge) {
            dynamic_smem = std::max(3 * this->m_max_faces_per_patch,
                                    2 * this->m_max_edges_per_patch) *
                           sizeof(uint16_t);
            dynamic_smem +=
                DIVIDE_UP(this->m_max_edges_per_patch, 32) * sizeof(uint32_t);
        }

        if (op == DynOp::DeleteVertex) {
            dynamic_smem = std::max(3 * this->m_max_faces_per_patch,
                                    2 * this->m_max_edges_per_patch) *
                           sizeof(uint16_t);
            dynamic_smem +=
                std::max(DIVIDE_UP(this->m_max_vertices_per_patch, 32),
                         DIVIDE_UP(this->m_max_edges_per_patch, 32)) *
                sizeof(uint32_t);
        }

        if (op == DynOp::EdgeCollapse) {
            dynamic_smem = std::max(3 * this->m_max_faces_per_patch,
                                    2 * this->m_max_edges_per_patch) *
                           sizeof(uint16_t);
        }
        return dynamic_smem;
    }
};
}  // namespace rxmesh