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
    //__shared__ uint32_t        s_num_deleted_faces;
    extern __shared__ uint16_t shrd_mem[];
    uint16_t*                  s_fe = shrd_mem;

    // if (threadIdx.x == 0) {
    //    s_num_deleted_faces = 0;
    //}
    // load FE and make sure to sync before
    load_mesh_async<Op::FE>(patch_info, s_fe, s_fe, false);
    __syncthreads();

    // load over all edges---one thread per edge
    uint16_t local_id = threadIdx.x;

    // we need to make sure that the whole warp go into the loop
    // Round to next multiple of 32
    // https://codegolf.stackexchange.com/a/17852
    uint16_t len = num_owned_faces;
    if (len % 32 != 0) {
        len = (num_owned_faces | 31) + 1;
    }
    while (local_id < len) {

        bool to_delete = false;
        if (local_id < num_owned_faces) {
            to_delete = predicate({patch_info.patch_id, local_id});
        }

        if (to_delete) {
            s_fe[3 * local_id + 0] = INVALID16;
            s_fe[3 * local_id + 1] = INVALID16;
            s_fe[3 * local_id + 2] = INVALID16;
        }

        // if thread in the N-th lane has deleted its edge, then warp_maks N-th
        // bit will be 1
        __syncwarp();
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

            // count how many faces this warp have deleted
            // uint32_t warp_num_deleted_face = __popc(warp_mask);
            //::atomicAdd(&s_num_deleted_faces, warp_num_deleted_face);
        }
        local_id += blockThreads;
    }

    // Since we assign one block per patch, we know that there should not be
    // other threads outside this block that access this patch. Thus, no need
    // for atomics
    // if (threadIdx.x == 0) {
    //    assert(patch_info.num_owned_faces >= s_num_deleted_faces);
    //    patch_info.num_owned_faces -= s_num_deleted_faces;
    //
    //    assert(patch_info.num_faces >= s_num_deleted_faces);
    //    patch_info.num_faces -= s_num_deleted_faces;
    //
    //    atomicSub(total_num_faces, s_num_deleted_faces);
    //}
    __syncthreads();
    detail::load_uint16<blockThreads>(
        s_fe,
        patch_info.num_faces * 3,
        reinterpret_cast<uint16_t*>(patch_info.fe));
}
}  // namespace detail
}  // namespace rxmesh