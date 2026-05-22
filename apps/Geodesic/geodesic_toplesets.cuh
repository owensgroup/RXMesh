#pragma once

// Device-side topleset (BFS-level) computation for the Geodesic / DGPC
// PTP-style schedulers. Replaces the OpenMesh-based CPU implementation in
// geodesic_ptp_openmesh.h with a sequence of Op::EV sweeps that need only
// the seed mask and the RXMesh static structure -- no OpenMesh required.
//
// Scheme: edge-relax pull-style BFS. For each edge (v0, v1), if exactly one
// endpoint is at the current BFS level `lvl` and the other is unset, the
// unset endpoint is promoted to `lvl + 1`. Multiple edges incident on the
// same vertex may race to write the same value (lvl + 1) -- benign because
// all racing writers write the same value. The only true atomic is the
// single counter that records whether any progress was made.
//
// Performance notes (vs. an Op::VV pull):
//   - Op::EV needs no oriented one-ring, no per-vertex variable iterator,
//     and no shared-memory ring build. Each thread does two reads + at most
//     one write, which is much cheaper per launched thread.
//   - The launch box is built once outside the loop. The previous version
//     called rx.for_each<Op::VV, 256> per level which re-did
//     prepare_launch_box every iteration -- a major source of overhead on
//     small/medium meshes.
//   - The current BFS level is held in a 1x1 device buffer so the lambda's
//     captures don't change between launches; no re-launch-prep required.
//   - The "did any vertex change?" counter is copied D2H asynchronously and
//     polled every K levels to hide the round-trip latency.
//
// Companion: build_limits_from_toplesets builds the CSR-like band-offset
// array required by geodesic_ptp_rxmesh.h on the host.

#include <cuda_runtime.h>
#include <vector>

#include "rxmesh/attribute.h"
#include "rxmesh/kernels/query_kernel.cuh"
#include "rxmesh/launch_box.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/timer.h"

template <typename TopT = int>
inline float compute_toplesets_device(
    rxmesh::RXMeshStatic&                   rx,
    const rxmesh::VertexAttribute<uint8_t>& seed_mask,
    rxmesh::VertexAttribute<TopT>&          topleset,
    uint32_t&                               num_levels)
{
    using namespace rxmesh;

    constexpr TopT kInvalid = static_cast<TopT>(INVALID32);
    constexpr int  kPollK   = 4;  // sync the D2H counter every K levels

    GPUTimer timer;
    timer.start();

    topleset.reset(kInvalid, DEVICE);

    rx.for_each_vertex(
        DEVICE,
        [topleset, seed_mask] __device__(const VertexHandle vh) mutable {
            if (seed_mask(vh, 0) != uint8_t(0)) {
                topleset(vh, 0) = TopT(0);
            }
        });

    // Device-side BFS state. d_lvl holds the current level so the lambda
    // capture set is invariant across launches; d_added counts edge writes
    // performed in this sweep (used only as a "did any progress happen?"
    // predicate, hence overcounting from multiple edges hitting the same
    // vertex is fine).
    DenseMatrix<int> d_lvl(rx, 1, 1, LOCATION_ALL);
    DenseMatrix<int> d_added(rx, 1, 1, LOCATION_ALL);
    int* const       lvl_ptr   = d_lvl.data(DEVICE);
    int* const       added_ptr = d_added.data(DEVICE);

    // Pinned host buffer for the asynchronous d_added readback.
    int* h_added = nullptr;
    cudaMallocHost(&h_added, sizeof(int));
    *h_added = 0;

    const auto level_lambda = [topleset, kInvalid, lvl_ptr, added_ptr]
        __device__(const EdgeHandle& /*eh*/, const VertexIterator& v) mutable {
            const TopT lvl  = static_cast<TopT>(*lvl_ptr);
            const TopT t0   = topleset(v[0], 0);
            const TopT t1   = topleset(v[1], 0);
            const bool z_at = (t0 == lvl);
            const bool o_at = (t1 == lvl);
            const bool z_un = (t0 == kInvalid);
            const bool o_un = (t1 == kInvalid);

            if (z_at && o_un) {
                topleset(v[1], 0) = lvl + TopT(1);
                ::atomicAdd(added_ptr, 1);
            }
            if (o_at && z_un) {
                topleset(v[0], 0) = lvl + TopT(1);
                ::atomicAdd(added_ptr, 1);
            }
        };

    constexpr uint32_t blockThreads = 256;
    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box(
        {Op::EV},
        lb,
        (void*)detail::query_kernel<blockThreads, Op::EV, decltype(level_lambda)>,
        /*oriented=*/false);

    int level                  = 0;
    int levels_since_last_sync = 0;
    while (true) {
        d_lvl(0, 0) = level;
        d_lvl.move(HOST, DEVICE);
        d_added.reset(0, DEVICE);

        rx.for_each<Op::EV, blockThreads>(lb, level_lambda, /*oriented=*/false);

        // Stride the D2H sync to hide round-trip latency. Issue an async
        // copy of the per-sweep counter after every kernel; only sync
        // (and possibly terminate) every kPollK levels.
        cudaMemcpyAsync(
            h_added, added_ptr, sizeof(int), cudaMemcpyDeviceToHost);
        ++levels_since_last_sync;
        ++level;

        if (levels_since_last_sync == kPollK) {
            cudaStreamSynchronize(0);
            levels_since_last_sync = 0;
            if (*h_added == 0) {
                // The most recent sweep made no progress. Up to kPollK - 1
                // earlier sweeps in this window may also have been no-ops
                // (we only see the last counter), so num_levels can
                // overshoot by < kPollK. Empty trailing bands in `limits`
                // are harmless to the downstream PTP driver: its band
                // convergence test (n_cond == h_error) trivially passes
                // for n_cond == 0 and the (i, j) window slides through.
                break;
            }
        }
    }

    // Drain any pending async copy before tearing down h_added.
    cudaStreamSynchronize(0);

    cudaFreeHost(h_added);

    num_levels = static_cast<uint32_t>(level);

    timer.stop();
    return timer.elapsed_millis();
}

// Build the band-offset array `limits` from a host-side topleset attribute.
// On entry `limits` must have at least `num_levels + 2` rows allocated on
// HOST. On exit:
//   limits(0, 0)               = 0
//   limits(k, 0)                = #vertices with level < k (k=1..num_levels)
//   limits(num_levels + 1, 0)   = num_vertices
//   limits_size                 = num_levels + 2
//
// Vertices with topleset == INVALID32 (unreachable) are NOT counted in any
// band but are added to the trailing entry so that the final entry equals
// num_vertices, matching the contract of geodesic_ptp_openmesh's output.
template <typename TopT = int>
inline void build_limits_from_toplesets(
    rxmesh::RXMeshStatic&                rx,
    const rxmesh::VertexAttribute<TopT>& topleset_host,
    int                                  num_levels,
    rxmesh::DenseMatrix<int>&            limits,
    int&                                 limits_size)
{
    using namespace rxmesh;

    const uint32_t num_vertices = rx.get_num_vertices();

    std::vector<int> count(num_levels, 0);
    rx.for_each_vertex(
        HOST,
        [&](const VertexHandle vh) {
            const TopT l = topleset_host(vh, 0);
            if (l != static_cast<TopT>(INVALID32) &&
                static_cast<uint32_t>(l) < num_levels) {
                ++count[static_cast<uint32_t>(l)];
            }
        },
        NULL,
        /*with_omp=*/false);

    limits(0, 0) = 0;
    int prefix   = 0;
    for (int l = 0; l < num_levels; ++l) {
        prefix += count[l];
        limits(l + 1, 0) = prefix;
    }
    limits(num_levels + 1, 0) = static_cast<int>(num_vertices);
    limits_size               = static_cast<int>(num_levels + 2);
}
