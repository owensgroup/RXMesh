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



}
}  // namespace detail
}  // namespace rxmesh