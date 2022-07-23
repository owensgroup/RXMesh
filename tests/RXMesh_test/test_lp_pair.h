#include "gtest/gtest.h"

#include "rxmesh/lp_pair.cuh"

TEST(RXMesh, LPPair)
{
    using namespace rxmesh;

    uint16_t local_id             = 5;
    uint16_t local_id_owner_patch = 10;
    uint8_t  patch                = 7;
    LPPair   pair(local_id, local_id_owner_patch, patch);
    EXPECT_EQ(pair.local_id(), local_id);
    EXPECT_EQ(pair.local_id_in_owner_patch(), local_id_owner_patch);
    EXPECT_EQ(pair.patch_stash_id(), patch);

    LPPair tombstone = LPPair::sentinel_pair();
    EXPECT_TRUE(tombstone.is_sentinel());
}


TEST(RXMesh, LPHashTable)
{
    using namespace rxmesh;

    uint32_t size = 256;

    auto random_with_bounds = [&](auto& vec, uint32_t high_val, uint32_t size) {
        for (uint32_t i = 0; i < size; ++i) {
            vec[i] = i % high_val;
        }
        random_shuffle(vec.data(), size);
    };

    std::vector<uint16_t> local_id(size);
    fill_with_random_numbers(local_id.data(), local_id.size());

    std::vector<uint16_t> owner_local_id(size);
    random_with_bounds(
        owner_local_id, 1 << LPPair::LIDOwnerNumBits, owner_local_id.size());

    std::vector<uint8_t> patch_id(size);
    random_with_bounds(
        patch_id, 1 << LPPair::PatchStashNumBits, patch_id.size());

    std::vector<LPPair> pairs(size);
    for (uint32_t i = 0; i < size; ++i) {
        pairs[i] = LPPair(local_id[i], owner_local_id[i], patch_id[i]);
    }

    float       load_factor = 0.7;
    LPHashTable table(
        static_cast<uint16_t>(static_cast<float>(size) / load_factor), false);

    for (auto& p : pairs) {
        EXPECT_TRUE(table.insert(p));
    }

    for (uint32_t i = 0; i < size; ++i) {
        auto p = table.find(local_id[i]);
        EXPECT_NE(p.m_pair, LPPair::sentinel_pair().m_pair);
        EXPECT_EQ(p.local_id_in_owner_patch(), owner_local_id[i]);
        EXPECT_EQ(p.patch_stash_id(), patch_id[i]);
    }

    table.free();
}

TEST(RXMesh, DISABLED_BenchmarkLPHashTable)
{
    using namespace rxmesh;

    std::cout << "size, cap, num_failed\n";
    const float    load_factor = 0.9;
    const uint32_t low_size    = 128;
    const uint32_t high_size   = 2048;
    const int      num_run     = 1E6;

    auto random_with_bounds = [&](auto& vec, uint32_t high_val, uint32_t size) {
        for (uint32_t i = 0; i < size; ++i) {
            vec[i] = i % high_val;
        }
        random_shuffle(vec.data(), size);
    };


    for (uint32_t size = low_size; size <= high_size; size++) {

        int      num_failed = 0;
        uint32_t cap        = 0;

        for (int t = 0; t < num_run; ++t) {
            std::vector<uint16_t> local_id(size);
            fill_with_random_numbers(local_id.data(), local_id.size());

            std::vector<uint16_t> owner_local_id(size);
            random_with_bounds(owner_local_id,
                               1 << LPPair::LIDOwnerNumBits,
                               owner_local_id.size());

            std::vector<uint8_t> patch_id(size);
            random_with_bounds(
                patch_id, 1 << LPPair::PatchStashNumBits, patch_id.size());

            std::vector<LPPair> pairs(size);
            for (uint32_t i = 0; i < size; ++i) {
                pairs[i] = LPPair(local_id[i], owner_local_id[i], patch_id[i]);
            }

            LPHashTable table(
                static_cast<uint16_t>(static_cast<float>(size) / load_factor),
                false);

            bool fail = false;
            for (auto& p : pairs) {
                if (!table.insert(p, table.get_table())) {
                    fail = true;
                    break;
                }
            }
            if (fail) {
                num_failed++;
            }

            cap = table.get_capacity();

            if (!fail) {
                for (uint32_t i = 0; i < size; ++i) {
                    auto p = table.find(local_id[i]);
                    EXPECT_NE(p.m_pair, LPPair::sentinel_pair().m_pair);
                    EXPECT_EQ(p.local_id_in_owner_patch(), owner_local_id[i]);
                    EXPECT_EQ(p.patch_stash_id(), patch_id[i]);
                }
            }
            table.free();
        }
        std::cout << size << ", " << cap << ", " << num_failed << "\n";
    }
}