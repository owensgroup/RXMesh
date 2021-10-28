#pragma once
#include <assert.h>
#include <cuda_profiler_api.h>
#include <memory>
#include "rxmesh/kernels/prototype.cuh"
#include "rxmesh/launch_box.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/timer.h"

namespace rxmesh {

/**
 * @brief This class is responsible for query operations of static meshes. It
 * extends RXMesh with methods needed to launch kernel and do computation on the
 * mesh as well as managing mesh attributes
 */
class RXMeshStatic : public RXMesh
{    
   public:
    RXMeshStatic(const RXMeshStatic&) = delete;

    RXMeshStatic(std::vector<std::vector<uint32_t>>& fv,
                 const bool                          quite = true)
        : RXMesh(fv, quite){};

    virtual ~RXMeshStatic()
    {
    }

    /**
     * @brief populate the launch_box with grid size and dynamic shared memory
     * needed for kernel launch
     * TODO provide variadic version of this function that can accept multiple
     * ops
     * @param op Query operation done inside this the kernel
     * @param launch_box input launch box to be populated
     * @param is_higher_query if the query done will be a higher ordered e.g.,
     * k-ring
     * @param oriented if the query is oriented. Valid only for Op::VV queries
     */
    template <uint32_t blockThreads>
    void prepare_launch_box(const Op                 op,
                            LaunchBox<blockThreads>& launch_box,
                            const bool               is_higher_query = false,
                            const bool               oriented = false) const
    {
        static_assert(
            blockThreads && ((blockThreads & (blockThreads - 1)) == 0),
            " RXMeshStatic::prepare_launch_box() CUDA block size "
            "should be of power "
            "2. ");

        launch_box.blocks = this->m_num_patches;

        const uint32_t output_fixed_offset =
            (op == Op::EV) ? 2 : ((op == Op::FV || op == Op::FE) ? 3 : 0);

        this->template calc_shared_memory<blockThreads>(
            op, launch_box, is_higher_query, oriented);
    }

    /**
     * @brief Adding a new face attribute
     * @tparam T type of the attribute
     * @param name name of the attribute
     * @param t initial value
     * @return shared pointer to the created attribute
     */
    template <class T>
    std::shared_ptr<RXMeshFaceAttribute<T>> add_face_attribute(
        const std::string& name,
        const T            t = T())
    {
        return std::dynamic_pointer_cast<RXMeshFaceAttribute<T>>(
            m_attr_container.template add<T>(name.c_str()));
    }

    /**
     * @brief Adding a new edge attribute
     * @tparam T type of the attribute
     * @param name name of the attribute
     * @param t initial value
     * @return shared pointer to the created attribute
     */
    template <class T>
    std::shared_ptr<RXMeshEdgeAttribute<T>> add_edge_attribute(
        const std::string& name,
        const T            t = T())
    {
        return std::dynamic_pointer_cast<RXMeshEdgeAttribute<T>>(
            m_attr_container.template add<T>(name.c_str()));
    }

    /**
     * @brief Adding a new vertex attribute
     * @tparam T type of the attribute
     * @param name name of the attribute
     * @param t initial value
     * @return shared pointer to the created attribute
     */
    template <class T>
    std::shared_ptr<RXMeshVertexAttribute<T>> add_vertex_attribute(
        const std::string& name,
        const T            t = T())
    {
        return std::dynamic_pointer_cast<add_vertex_attribute<T>>(
            m_attr_container.template add<T>(name.c_str()));
    }

   protected:
    template <uint32_t blockThreads>
    void calc_shared_memory(const Op                 op,
                            LaunchBox<blockThreads>& launch_box,
                            const bool               is_higher_query,
                            const bool               oriented = false) const
    {
        // Operations that uses matrix transpose needs a template parameter
        // that is by default TRANSPOSE_ITEM_PER_THREAD. Here we check if
        // this default parameter is valid otherwise, it needs to be increased.
        if (op == Op::VV || op == Op::VE) {
            if (2 * this->m_max_edges_per_patch >
                blockThreads * TRANSPOSE_ITEM_PER_THREAD) {
                RXMESH_ERROR(
                    "RXMeshStatic::calc_shared_memory() "
                    "TRANSPOSE_ITEM_PER_THREAD = {} needs "
                    "to be increased for op = {}",
                    TRANSPOSE_ITEM_PER_THREAD,
                    op_to_string(op));
            }
        } else if (op == Op::VE || op == Op::EF || op == Op::FF) {
            if (3 * this->m_max_faces_per_patch >
                blockThreads * TRANSPOSE_ITEM_PER_THREAD) {
                RXMESH_ERROR(
                    "RXMeshStatic::calc_shared_memory() "
                    "TRANSPOSE_ITEM_PER_THREAD = {} needs "
                    "to be increased for op = {}",
                    TRANSPOSE_ITEM_PER_THREAD,
                    op_to_string(op));
            }
        }


        if (oriented && op != Op::VV) {
            RXMESH_ERROR(
                "RXMeshStatic::calc_shared_memory() Oriented is only "
                "allowed on VV. The input op is {}",
                op_to_string(op));
        }

        if (oriented && op == Op::VV && !this->m_is_input_closed) {
            RXMESH_ERROR(
                "RXMeshStatic::calc_shared_memory() Can't generate oriented "
                "output (VV) for input with boundaries");
        }

        launch_box.smem_bytes_dyn = 0;

        if (op == Op::FE) {
            // only faces will be loaded and no extra shared memory is needed
            launch_box.smem_bytes_dyn =
                3 * this->m_max_faces_per_patch * sizeof(uint16_t);
        } else if (op == Op::EV) {
            // only edges will be loaded and no extra shared memory is needed
            launch_box.smem_bytes_dyn =
                2 * this->m_max_edges_per_patch * sizeof(uint16_t);
        } else if (op == Op::FV) {
            // We load both faces and edges. We don't change edges.
            // faces are updated to contain FV instead of FE by reading from
            // edges
            launch_box.smem_bytes_dyn =
                3 * this->m_max_faces_per_patch * sizeof(uint16_t) +
                2 * this->m_max_edges_per_patch * sizeof(uint16_t);
        } else if (op == Op::VE) {
            // load edges and then transpose it in place
            // The transpose needs two buffer; one for prefix sum and another
            // for the actual output
            // The prefix sum will be stored in place (where edges are loaded)
            // The output will be stored in another buffer with size equal to
            // the edges since this output buffer will stored the nnz and the
            // nnz of a matrix the same before/after transpose
            launch_box.smem_bytes_dyn =
                (2 * 2 * this->m_max_edges_per_patch) * sizeof(uint16_t);
        } else if (op == Op::EF || op == Op::VF) {
            // same as above but with faces
            launch_box.smem_bytes_dyn =
                (2 * 3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
                sizeof(uint16_t);
        } else if (op == Op::VV) {
            // similar to VE but we also need to store the edges (EV) even after
            // we do the transpose.
            launch_box.smem_bytes_dyn =
                (3 * 2 * this->m_max_edges_per_patch) * sizeof(uint16_t);
        } else if (op == Op::FF) {
            // FF needs to store FE and EF along side with the output itself
            // FE needs 3*max_num_faces
            // EF is FE transpose
            // FF is max_num_faces + (on average) 3*max_num_faces
            // Since we have so many boundary faces (due to ribbons), they will
            // make up this averaging

            launch_box.smem_bytes_dyn =
                (3 * this->m_max_faces_per_patch +        // FE
                 2 * (3 * this->m_max_faces_per_patch) +  // EF
                 4 * this->m_max_faces_per_patch          // FF
                 ) *
                sizeof(uint16_t);
        }

        if (op == Op::VV && oriented) {
            // For oriented VV, we additionally need to store FE and EF along
            // with the (transposed) VE
            // FE needs 3*max_num_faces
            // Since oriented is only done on manifold, EF needs only
            // 2*max_num_edges since every edge is neighbor to maximum of two
            // faces (which we write on the same place as the extra EV)
            launch_box.smem_bytes_dyn += (/*2 * this->m_max_edges_per_patch +*/
                                          3 * this->m_max_faces_per_patch) *
                                         sizeof(uint16_t);
        }

        // to store output ltog map without the need to overlap it with
        // where we store mesh edges/faces
        // The +1 is for padding
        if (op == Op::EV || op == Op::FV /*|| op == Op::VV*/) {
            // For VV, we overwrite the extra storage we used above
            // to store the mapping which is more than enough to store the
            // vertices ltog
            launch_box.smem_bytes_dyn +=
                (this->m_max_vertices_per_patch + 1) * sizeof(uint32_t);

        } else if (op == Op::FE || op == Op::VE || op == Op::EE) {
            launch_box.smem_bytes_dyn +=
                (this->m_max_edges_per_patch + 1) * sizeof(uint32_t);
        } else if (op == Op::VF || op == Op::EF /*|| op == Op::FF*/) {
            launch_box.smem_bytes_dyn +=
                (this->m_max_faces_per_patch + 1) * sizeof(uint32_t);
        }


        launch_box.smem_bytes_static = check_shared_memory<blockThreads>(
            op, launch_box.smem_bytes_dyn, is_higher_query);


        if (!this->m_quite) {
            RXMESH_TRACE(
                "rxmesh::calc_shared_memory() launching {} blocks with {} "
                "threads on the device",
                launch_box.blocks,
                blockThreads);
        }
    }

    template <uint32_t threads>
    uint32_t check_shared_memory(const Op       op,
                                 const uint32_t smem_bytes_dyn,
                                 bool           is_higher_query) const
    {
        // check if total shared memory (static + dynamic) consumed by
        // k_base_query are less than the max shared per block
        cudaFuncAttributes func_attr;
        switch (op) {
            case Op::VV: {
                if (is_higher_query) {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr,
                        detail::higher_query_prototype<Op::VV, threads>));
                } else {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr, detail::query_prototype<Op::VV, threads>));
                }

                break;
            }
            case Op::VE: {
                if (is_higher_query) {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr,
                        detail::higher_query_prototype<Op::VE, threads>));
                } else {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr, detail::query_prototype<Op::VE, threads>));
                }
                break;
            }
            case Op::VF: {
                if (is_higher_query) {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr,
                        detail::higher_query_prototype<Op::VF, threads>));
                } else {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr, detail::query_prototype<Op::VF, threads>));
                }
                break;
            }
            case Op::EV: {
                if (is_higher_query) {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr,
                        detail::higher_query_prototype<Op::EV, threads>));
                } else {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr, detail::query_prototype<Op::EV, threads>));
                }
                break;
            }
            case Op::EE: {
                break;
            }
            case Op::EF: {
                if (is_higher_query) {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr,
                        detail::higher_query_prototype<Op::EF, threads>));
                } else {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr, detail::query_prototype<Op::EF, threads>));
                }
                break;
            }
            case Op::FV: {
                if (is_higher_query) {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr,
                        detail::higher_query_prototype<Op::FV, threads>));
                } else {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr, detail::query_prototype<Op::FV, threads>));
                }
                break;
            }
            case Op::FE: {
                if (is_higher_query) {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr,
                        detail::higher_query_prototype<Op::FE, threads>));
                } else {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr, detail::query_prototype<Op::FE, threads>));
                }
                break;
            }
            case Op::FF: {
                if (is_higher_query) {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr,
                        detail::higher_query_prototype<Op::FF, threads>));
                } else {
                    CUDA_ERROR(cudaFuncGetAttributes(
                        &func_attr, detail::query_prototype<Op::FF, threads>));
                }
                break;
            }
        }

        uint32_t smem_bytes_static = func_attr.sharedSizeBytes;
        uint32_t num_regs          = func_attr.numRegs;
        int      device_id;
        CUDA_ERROR(cudaGetDevice(&device_id));
        cudaDeviceProp devProp;
        CUDA_ERROR(cudaGetDeviceProperties(&devProp, device_id));

        if (!this->m_quite) {
            RXMESH_TRACE(
                "RXMeshStatic::check_shared_memory() query_prototype with "
                "{} "
                "required shared memory = {} (dynamic) +  {} (static) = {} "
                "(bytes) and {} registers",
                op_to_string(op),
                smem_bytes_dyn,
                smem_bytes_static,
                smem_bytes_dyn + smem_bytes_static,
                num_regs);

            RXMESH_TRACE(
                "RXMeshStatic::check_shared_memory() available total shared "
                "memory per block = {} (bytes) = {} (Kb)",
                devProp.sharedMemPerBlock,
                float(devProp.sharedMemPerBlock) / 1024.0f);
        }

        if (smem_bytes_static + smem_bytes_dyn > devProp.sharedMemPerBlock) {
            RXMESH_ERROR(
                " RXMeshStatic::check_shared_memory() shared memory needed for"
                " query_prototype ({} bytes) exceeds the max shared memory "
                "per block on the current device ({} bytes)",
                smem_bytes_static + smem_bytes_dyn,
                devProp.sharedMemPerBlock);
            exit(EXIT_FAILURE);
        }
        return static_cast<uint32_t>(smem_bytes_static);
    }


    RXMeshAttributeContainer m_attr_container;
};
}  // namespace rxmesh