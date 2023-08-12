#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <numeric>

#include "rxmesh/util/bitmask_util.h"

namespace rxmesh {
/**
 * @brief This struct store the index of the not-owned mesh elements as a 32-bit
 * unsigned int where the high 16 bits store the local index within the patch
 * and the low 16 bits stores the local index in owner patch along with the
 * owner patch id. These lower 16 bits are actually divided into high 4 bits
 * that store an index into PatchStash and low 12 bits that stores the index
 * into this owner patch
 * So, the 32-bit stored in this struct would look like
 *
 * ----------------|----|------------
 *          A      | B  |      C
 * A is the local ID
 * B is the index within PatchStash
 * C is the local index within the owner patch
 */
struct LPPair
{
    using KeyT   = uint16_t;
    using ValueT = uint16_t;

    // Local index (high) number of bits within the patch
    constexpr static uint32_t LIDNumBits = 16;

    // Local index (low) number of bits bit within the owner patch
    constexpr static uint32_t LIDOwnerNumBits = 12;

    // Number of bits reserved for the owner patch ID in the PatchStash
    constexpr static uint32_t PatchStashNumBits = 4;

    /**
     * @brief Constructor using the local ID (key),
     * local ID with the owner patch, patch id within the PatchStash (value)
     * @param local_id the local ID in the (not the owner) patch
     * @param local_id_in_owner_patch the local index within the owner patch
     * @param owner_patch the owner patch id within the PatchStash
     */
    __host__ __device__ explicit LPPair(uint16_t local_id,
                                        uint16_t local_id_in_owner_patch,
                                        uint8_t  owner_patch)
    {
        static_assert(
            LIDNumBits + LIDOwnerNumBits + PatchStashNumBits == 32,
            "The sum of LIDNumBits, LIDOwnerNumBits, and PatchStashNumBits "
            "should be 32");

        assert(local_id_in_owner_patch == INVALID16 ||
               local_id_in_owner_patch <= (1 << LIDOwnerNumBits));
        assert(owner_patch == INVALID8 ||
               owner_patch <= (1 << PatchStashNumBits));


        ValueT pv = make_value(local_id_in_owner_patch, owner_patch);
        m_pair    = local_id;
        m_pair    = m_pair << LIDNumBits;
        m_pair |= pv;
    }

    __host__ __device__ LPPair(KeyT key, ValueT value)
    {
        m_pair = key;
        m_pair = m_pair << 16;
        m_pair |= value;
    }

    __host__ __device__ LPPair() : m_pair(INVALID32){};
    __host__ __device__ LPPair(uint32_t p) : m_pair(p){};
    LPPair(const LPPair& other) = default;
    LPPair(LPPair&&)            = default;
    LPPair& operator=(const LPPair&) = default;
    LPPair& operator=(LPPair&&) = default;
    ~LPPair()                   = default;

    /**
     * @brief concatenate the lid and stash_id into a single uint16_t where the
     * lid take the lower 12 bits and stash_id takes the high 4 bits
     */
    __host__ __device__ static ValueT make_value(const uint16_t lid,
                                                 const uint8_t  stash_id)
    {
        // assert that the high 4 bits are zeros
        assert(lid == INVALID16 ||
               detail::extract_high_bits<PatchStashNumBits>(lid) == 0);

        uint16_t p16 = static_cast<uint16_t>(stash_id);
        p16          = p16 << LIDOwnerNumBits;
        return (lid | p16);
    }

    /**
     * @brief Construct and return a tombstone pair
     */
    __host__ __device__ static LPPair sentinel_pair()
    {
        return LPPair(INVALID16, INVALID16, INVALID8);
    }

    /**
     * @brief Check if a pair is a tombstone
     */
    __device__ __host__ __inline__ bool is_sentinel() const
    {
        return m_pair == INVALID32;
    }

    /**
     * @brief returns the local index in the patch (possibly not the owner
     * patch)
     */
    __device__ __host__ __inline__ uint16_t local_id() const
    {
        // get the high 16 bits by shift right
        return m_pair >> (LIDOwnerNumBits + PatchStashNumBits);
    }

    /**
     * @brief the key used for hashing
     */
    __device__ __host__ __inline__ KeyT key() const
    {
        // get the high 16 bits by shift right
        return local_id();
    }


    /**
     * @brief the value associated with the key
     */
    __device__ __host__ __inline__ ValueT value() const
    {
        // get the low 16 bits
        return m_pair & INVALID16;
    }

    /**
     * @brief returns the local index in the owner patch
     */
    __device__ __host__ __inline__ uint16_t local_id_in_owner_patch() const
    {
        // get the low 12 bits by clearing the high bits
        return detail::extract_low_bits<LIDOwnerNumBits>(m_pair);
    }

    /**
     * @brief returns the id within the patch stash where the owner patch ID is
     * stored
     */
    __device__ __host__ __inline__ uint8_t patch_stash_id() const
    {
        const uint16_t temp = m_pair >> (LIDOwnerNumBits);
        return static_cast<uint8_t>(
            detail::extract_low_bits<PatchStashNumBits>(temp));
    }

    uint32_t m_pair;
};
}  // namespace rxmesh