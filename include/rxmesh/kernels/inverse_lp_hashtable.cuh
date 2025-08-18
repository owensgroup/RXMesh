#pragma once
#include <cooperative_groups.h>

#include "rxmesh/lp_hashtable.cuh"


namespace rxmesh {

struct InverseLPHashTable
{
    __device__ InverseLPHashTable()
        : m_capacity(0), m_s_table(nullptr), m_s_stash(nullptr)
    {
    }
    InverseLPHashTable(const InverseLPHashTable& other)      = default;
    InverseLPHashTable(InverseLPHashTable&&)                 = default;
    InverseLPHashTable& operator=(const InverseLPHashTable&) = default;
    InverseLPHashTable& operator=(InverseLPHashTable&&)      = default;
    ~InverseLPHashTable()                                    = default;

    explicit __device__ InverseLPHashTable(const LPHashTable& lp_table,
                                           LPPair*            s_ptr,
                                           LPPair*            s_stash)
    {
        m_capacity = lp_table.m_capacity;
        m_s_table  = s_ptr;

        m_s_stash = s_stash;

        assert(m_s_table);
        assert(m_s_stash);
    } 

    /**
     * @brief Reset the hash table to sentinel key-value (tombstone). This API
     * is for the device only and must be called by a single block
     */
    template <uint32_t blockThreads>
    __device__ __inline__ void clear()
    {
        assert(m_capacity > 0);
        assert(m_s_table);
        assert(m_s_stash);

        for (uint32_t i = threadIdx.x; i < m_capacity; i += blockThreads) {
            m_s_table[i].m_pair = INVALID32;
        }
        for (uint32_t i = threadIdx.x; i < LPHashTable::stash_size;
             i += blockThreads) {
            m_s_stash[i].m_pair = INVALID32;
        }
    }

    /**
     * @brief apply a function for each non-sentinel entry in the hash table
     */
    template <uint32_t blockThreads, typename FuncT>
    __device__ __inline__ void for_each(FuncT func) const
    {
        assert(m_capacity > 0);
        assert(m_s_table);
        assert(m_s_stash);

        for (uint32_t i = threadIdx.x; i < m_capacity; i += blockThreads) {
            const LPPair lp = m_s_table[i];
            if (lp.m_pair != INVALID32) {

                func(lp.local_id(),
                     lp.local_id_in_owner_patch(),
                     lp.patch_stash_id());
            }
        }
        for (uint32_t i = threadIdx.x; i < LPHashTable::stash_size;
             i += blockThreads) {
            const LPPair lp = m_s_stash[i];
            if (lp.m_pair != INVALID32) {
                func(lp.local_id(),
                     lp.local_id_in_owner_patch(),
                     lp.patch_stash_id());
            }
        }
    }

    template <uint32_t blockThreads, typename FuncT>
    __device__ __inline__ void for_each_lp(FuncT func) const
    {
        assert(m_capacity > 0);
        assert(m_s_table);
        assert(m_s_stash);

        for (uint32_t i = threadIdx.x; i < m_capacity; i += blockThreads) {
            const LPPair lp = m_s_table[i];
            if (lp.m_pair != INVALID32) {

                func(lp);
            }
        }
        for (uint32_t i = threadIdx.x; i < LPHashTable::stash_size;
             i += blockThreads) {
            const LPPair lp = m_s_stash[i];
            if (lp.m_pair != INVALID32) {
                func(lp);
            }
        }
    }

    /**
     * @brief Return the key of an LPPair which can be used for comparisons in
     * the inverse table. The key of LPHashTable is the local id. Thus, the key
     * in the InverseLPHashTable is the <patch stash id,local id in owner patch>
     * i.e., the low LIDOwnerNumBits + PatchStashNumBits bits in pair.
     * Since we use 13 bits for LIDOwnerNumBits and 6 bits PatchStashNumBits, we
     * return 32-bit index
     */
    __device__ __inline__ uint32_t get_key(const LPPair& pair) const
    {

        static_assert(LPPair::LIDOwnerNumBits + LPPair::PatchStashNumBits <= 32,
                      "We assume the local id + patch stash is less 32 bits");

        assert(m_capacity > 0);
        assert(m_s_table);
        assert(m_s_stash);

        uint32_t ret = detail::extract_low_bits<LPPair::LIDOwnerNumBits +
                                                    LPPair::PatchStashNumBits,
                                                uint32_t>(pair.m_pair);

        return ret;
    }

    /**
     * @brief Return the value of an LPPair. The value of LPPair in LPHashTable
     * is the local index in owner patch+ patch stash id. Thus, the value in the
     * InverseLPHashTable is the local id, i.e., the high LIDNumBits bits in the
     * pair. Thus, we return uint16_t
     * @param pair
     * @return
     */
    __device__ __inline__ uint16_t get_value(const LPPair& pair) const
    {
        static_assert(LPPair::LIDNumBits <= 16,
                      "We assume the local id is less 16 bits");

        assert(m_capacity > 0);
        assert(m_s_table);
        assert(m_s_stash);

        uint16_t ret = static_cast<uint16_t>(
            detail::extract_high_bits<LPPair::LIDNumBits, uint32_t>(
                pair.m_pair));

        return ret;
    }

    /**
     * @brief Combine the local id in the owner patch and patch stash to make a
     * key that can be used in the inverse LPPHashTable
     */
    __device__ __inline__ uint32_t get_key(
        const uint16_t local_id_in_owner_patch,
        const uint8_t  owner_patch) const
    {
        assert(m_capacity > 0);
        assert(m_s_table);
        assert(m_s_stash);

        uint32_t key = static_cast<uint32_t>(owner_patch);
        key          = key << LPPair::LIDOwnerNumBits;
        key |= local_id_in_owner_patch;
        return key;
    }

    /**
     * @brief insert a new LPPair in the table. The key of this pair is
     * the value used when this pair is inserted in LPHashTable, i.e., the low
     * LIDOwnerNumBits+PatchStashNumBits bits
     */
    __device__ __inline__ bool insert(const LPHashTable& lp_table, LPPair pair)
    {
        assert(m_capacity > 0);
        assert(m_s_table);
        assert(m_s_stash);

        auto bucket_id = lp_table.m_hasher0(get_key(pair)) % m_capacity;

        uint16_t cuckoo_counter = 0;

        do {
            const uint32_t input_key = get_key(pair);

            pair.m_pair =
                ::atomicExch((uint32_t*)m_s_table + bucket_id, pair.m_pair);


            // compare against initial key to avoid duplicated
            // i.e., if we are inserting a pair such that its key already
            // exists, this comparison would allow updating the pair
            const uint32_t k = get_key(pair);
            if (pair.is_sentinel() || k == input_key) {
                return true;
            } else {
                auto bucket0 = lp_table.m_hasher0(k) % m_capacity;
                auto bucket1 = lp_table.m_hasher1(k) % m_capacity;
                auto bucket2 = lp_table.m_hasher2(k) % m_capacity;
                auto bucket3 = lp_table.m_hasher3(k) % m_capacity;


                auto new_bucket_id = bucket0;
                if (bucket_id == bucket2) {
                    new_bucket_id = bucket3;
                } else if (bucket_id == bucket1) {
                    new_bucket_id = bucket2;
                } else if (bucket_id == bucket0) {
                    new_bucket_id = bucket1;
                }

                bucket_id = new_bucket_id;
            }
            cuckoo_counter++;

            //if (cuckoo_counter > 1000) {
            //    printf("\n cuckoo_counter= %u", cuckoo_counter);
            //}

        } while (cuckoo_counter < lp_table.m_max_cuckoo_chains);

        const uint32_t input_key = get_key(pair);

        for (uint8_t i = 0; i < lp_table.stash_size; ++i) {
            LPPair prv;

            prv.m_pair = ::atomicCAS(reinterpret_cast<uint32_t*>(m_s_stash + i),
                                     INVALID32,
                                     pair.m_pair);

            if (prv.is_sentinel() || get_key(prv) == input_key) {
                return true;
            }
        }

        return false;
    }


    /**
     * @brief
     */
    __device__ __inline__ LPPair find(const LPHashTable& lp_table,
                                      const uint16_t local_id_in_owner_patch,
                                      const uint8_t  owner_patch) const
    {
        return find(lp_table, get_key(local_id_in_owner_patch, owner_patch));
    }

    /**
     * @brief
     */
    __device__ __inline__ LPPair find(const LPHashTable& lp_table,
                                      const uint32_t     key) const
    {

        assert(m_capacity == lp_table.m_capacity);

        assert(m_capacity > 0);
        assert(m_s_table);
        assert(m_s_stash);

        constexpr int num_hfs = 4;

        uint32_t bucket_id = lp_table.m_hasher0(key) % m_capacity;

        for (int hf = 0; hf < num_hfs; ++hf) {

            uint32_t* ptr = reinterpret_cast<uint32_t*>(m_s_table + bucket_id);

            LPPair found = LPPair(atomic_read(ptr));


            if (get_key(found) == key) {
                return found;
                // since we only look for pairs that we know that they exist in
                // the table, we skip this since the pair could be in the stash
                //  } else if (found.m_pair == INVALID32) {
                //    return found;
            } else {
                if (hf == 0) {
                    bucket_id = lp_table.m_hasher1(key) % m_capacity;
                } else if (hf == 1) {
                    bucket_id = lp_table.m_hasher2(key) % m_capacity;
                } else if (hf == 2) {
                    bucket_id = lp_table.m_hasher3(key) % m_capacity;
                }
            }
        }

        for (bucket_id = 0; bucket_id < lp_table.stash_size; ++bucket_id) {

            LPPair st_pair = m_s_stash[bucket_id];

            if (get_key(st_pair) == key) {
                return st_pair;
            }
        }
        return LPPair::sentinel_pair();
    }

    uint16_t m_capacity;
    LPPair*  m_s_table;
    LPPair*  m_s_stash;
};
}  // namespace rxmesh
