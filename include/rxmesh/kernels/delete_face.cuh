namespace rxmesh {

namespace detail {
/**
 * @brief delete edge
 */
template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void delete_face(PatchInfo&       patch_info,
                                       const predicateT predicate)
{
    // Extract the argument in the predicate lambda function
    using PredicateTTraits = detail::FunctionTraits<predicateT>;
    using HandleT          = typename PredicateTTraits::template arg<0>::type;
    static_assert(
        std::is_same_v<HandleT, FaceHandle>,
        "First argument in predicate lambda function should be FaceHandle");

    // patch basic info
    const uint16_t num_owned_faces = patch_info.num_owned_faces;

    // shared memory used to store FE
    extern __shared__ uint16_t shrd_mem[];
    uint16_t*                  s_fe = shrd_mem;

    // load FE and make sure to sync before
    load_mesh_async<Op::FE>(patch_info, s_fe, s_fe, false);
    __syncthreads();

    // load over all edges---one thread per edge
    uint16_t local_id = threadIdx.x;

    // TODO we need to make sure that the whole warp go into the loop
    while (local_id < num_owned_faces) {

        bool to_delete = predicate({patch_info.patch_id, local_id});
        if (to_delete) {
            s_fe[3 * local_id + 0] = INVALID16;
            s_fe[3 * local_id + 1] = INVALID16;
            s_fe[3 * local_id + 2] = INVALID16;
        }

        // if thread in the N-th lane has deleted its edge, then warp_maks N-th
        // bit will be 1
        uint32_t warp_mask = __ballot_sync(__activemask(), to_delete);

        // let the thread in first laen writes the new bit mask
        uint32_t lane_id = threadIdx.x % 32;  // 32 = warp size
        if (lane_id == 0) {
            // here we first need to bitwise invert the warp_mask and then AND
            // it with the bitmask stored in global memory
            // Example for 4 threads/faces where the first faces
            // has been deleted before and the second face is deleted now
            // Bitmask stored in global memory is: 0 1 1 1
            // Bitmask in warp_mask:               0 1 0 0
            // The result we want to store:        0 0 1 1
            uint32_t mask_id           = local_id / 32;
            uint32_t old_mask          = patch_info.mask_f[mask_id];
            patch_info.mask_f[mask_id] = ((~warp_mask) & old_mask);
        }


        local_id += blockThreads;
    }
    __syncthreads();
}
}  // namespace detail
}  // namespace rxmesh