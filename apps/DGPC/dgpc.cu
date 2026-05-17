// Implementation of:
// Eivind Lyche Melv�r and Martin Reimers
// "Geodesic polar coordinates on polygonal meshes."
//
// Reference
// https://github.com/adobe/lagrange/blob/main/modules/geodesic/src/GeodesicEngineDGPC.cpp
//
// The Lagrange CPU implementation uses a serial Dijkstra with a priority
// queue. Here we replace the queue with iterative parallel vertex relaxation
// until no vertex's distance changes. The triangle update rule
// (Heron-area based unfolding) is identical to Lagrange's try_compute_dgpc,
// ported to use rx_coord_t.

#include <CLI/CLI.hpp>

#include "rxmesh/query.h"
#include "rxmesh/rxmesh_static.h"


using namespace rxmesh;

namespace {

constexpr rx_coord_t DGPC_INVALID = rx_coord_t(-1);
constexpr rx_coord_t DGPC_EPS     = rx_coord_t(1e-12);
constexpr rx_coord_t DGPC_PI      = rx_coord_t(3.14159265358979323846);

__device__ __host__ __forceinline__ rx_coord_t heron_area_stable(rx_coord_t e0,
                                                                 rx_coord_t e1,
                                                                 rx_coord_t e2)
{
    rx_coord_t a, b, c;
    if (e0 > e2) {
        a = e0;
        b = e1;
    } else {
        a = e1;
        b = e0;
    }
    c = e2;
    rx_coord_t r =
        (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c));
    if (r < rx_coord_t(0)) {
        return rx_coord_t(0);
    }
    return sqrt(r);
}

// Try to relax dist(vi)/theta(vi) using the unfolded triangle (vi, vj, vk).
// Returns true if a candidate (out_dist, out_theta) is produced. The caller
// is responsible for comparing against the current dist(vi) and committing.
__device__ __forceinline__ bool try_compute_dgpc(
    const VertexAttribute<rx_coord_t>& coords,
    const VertexAttribute<rx_coord_t>& dist,
    const VertexAttribute<rx_coord_t>& theta,
    const VertexHandle&                vi,
    const VertexHandle&                vj,
    const VertexHandle&                vk,
    rx_coord_t                         radius,
    rx_coord_t&                        out_dist,
    rx_coord_t&                        out_theta)
{
    const rx_coord_t Uj = dist(vj, 0);
    const rx_coord_t Uk = dist(vk, 0);
    if (Uj == DGPC_INVALID || Uk == DGPC_INVALID) {
        return false;
    }

    const vec3<rx_coord_t> pi = coords.to_glm<3>(vi);
    const vec3<rx_coord_t> pj = coords.to_glm<3>(vj);
    const vec3<rx_coord_t> pk = coords.to_glm<3>(vk);

    const vec3<rx_coord_t> ekj = pk - pj;
    const rx_coord_t       lkj = glm::length(ekj);
    if (lkj <= rx_coord_t(0)) {
        return false;
    }
    const rx_coord_t H = heron_area_stable(Uj, Uk, lkj);

    const vec3<rx_coord_t> ej = pj - pi;
    const vec3<rx_coord_t> ek = pk - pi;
    const rx_coord_t       A  = glm::length(glm::cross(ej, ek));
    if (A <= rx_coord_t(0)) {
        return false;
    }

    const rx_coord_t lkj2 = lkj * lkj;
    const rx_coord_t xj =
        A * (lkj2 + Uk * Uk - Uj * Uj) + glm::dot(ek, ekj) * H;
    const rx_coord_t xk =
        A * (lkj2 + Uj * Uj - Uk * Uk) - glm::dot(ej, ekj) * H;

    const rx_coord_t lj             = glm::length(ej);
    const rx_coord_t lk             = glm::length(ek);
    const rx_coord_t dijk_through_j = Uj + lj;
    const rx_coord_t dijk_through_k = Uk + lk;

    rx_coord_t alpha = rx_coord_t(-1);
    rx_coord_t Uijk  = rx_coord_t(0);

    if (xj > rx_coord_t(0) && xk > rx_coord_t(0)) {
        Uijk = glm::length(xj * ej + xk * ek) / (rx_coord_t(2) * A * lkj2);
    } else {
        if (dijk_through_j < dijk_through_k) {
            alpha = rx_coord_t(0);
            Uijk  = dijk_through_j;
        } else {
            alpha = rx_coord_t(1);
            Uijk  = dijk_through_k;
        }
    }

    const rx_coord_t height = H * rx_coord_t(0.25) / lkj;
    if (Uijk > radius && (height > radius || alpha >= rx_coord_t(0))) {
        return false;
    }

    const rx_coord_t curr = dist(vi, 0);
    if (!(curr == DGPC_INVALID ||
          (curr / Uijk > rx_coord_t(1) + DGPC_EPS && curr > Uijk + DGPC_EPS))) {
        return false;
    }

    if (alpha == rx_coord_t(-1)) {
        const rx_coord_t lj2 = lj * lj;
        const rx_coord_t lk2 = lk * lk;
        rx_coord_t       phi_ij_cos =
            (Uj * Uj + Uijk * Uijk - lj2) / (rx_coord_t(2) * Uj * Uijk);
        if (phi_ij_cos < rx_coord_t(-1))
            phi_ij_cos = rx_coord_t(-1);
        else if (phi_ij_cos > rx_coord_t(1))
            phi_ij_cos = rx_coord_t(1);
        const rx_coord_t phi_ij = acos(phi_ij_cos);

        rx_coord_t phi_ki_cos =
            (Uk * Uk + Uijk * Uijk - lk2) / (rx_coord_t(2) * Uk * Uijk);
        if (phi_ki_cos < rx_coord_t(-1))
            phi_ki_cos = rx_coord_t(-1);
        else if (phi_ki_cos > rx_coord_t(1))
            phi_ki_cos = rx_coord_t(1);
        const rx_coord_t phi_ki = acos(phi_ki_cos);

        if (phi_ij < DGPC_EPS && phi_ki < DGPC_EPS) {
            alpha = rx_coord_t(0.5);
        } else {
            alpha = phi_ij / (phi_ij + phi_ki);
        }
    }

    const rx_coord_t tj = theta(vj, 0);
    const rx_coord_t tk = theta(vk, 0);
    rx_coord_t       new_theta;
    if (fabs(tk - tj) > DGPC_PI) {
        new_theta = (rx_coord_t(1) - alpha) * tj + alpha * tk +
                    ((tj < tk) ? (rx_coord_t(1) - alpha) : alpha) *
                        rx_coord_t(2) * DGPC_PI;
        if (new_theta > DGPC_PI) {
            new_theta -= rx_coord_t(2) * DGPC_PI;
        }
    } else {
        new_theta = (rx_coord_t(1) - alpha) * tj + alpha * tk;
    }

    out_dist  = Uijk;
    out_theta = new_theta;
    return true;
}

}  // namespace

// Frontier-based parallel relaxation kernel. Uses Query::dispatch<Op::VV>
// with an active_set predicate so only vertices currently in the frontier
// run the per-triangle DGPC update. On a successful update we mark the
// vertex AND each of its one-ring neighbors in next_mask so they are
// revisited in the next sweep. The fixed point is identical to
// the previous full-sweep loop (and to Lagrange's Dijkstra), since the
// per-triangle update is unchanged and monotone in dist(vi).
template <uint32_t blockThreads>
__global__ static void relax_dgpc_rxmesh(
    const __grid_constant__ rxmesh::Context   context,
    const rxmesh::VertexAttribute<rx_coord_t> coords,
    rxmesh::VertexAttribute<rx_coord_t>       dist,
    rxmesh::VertexAttribute<rx_coord_t>       theta,
    const rxmesh::VertexAttribute<uint8_t>    active_mask,
    rxmesh::VertexAttribute<uint8_t>          next_mask,
    int*                                      d_changed,
    const rx_coord_t                          radius)
{
    using namespace rxmesh;

    auto in_active_set = [&](VertexHandle p_id) {
        return active_mask(p_id, 0) != uint8_t(0);
    };

    auto relax = [&](VertexHandle& vi, const VertexIterator& nbrs) {
        const uint32_t n = nbrs.size();
        if (n < 2) {
            return;
        }

        rx_coord_t best_dist  = dist(vi, 0);
        rx_coord_t best_theta = theta(vi, 0);
        bool       improved   = false;

        for (uint32_t v = 0; v < n; ++v) {
            const VertexHandle vj = nbrs[v];
            const VertexHandle vk = nbrs[(v + 1) % n];

            rx_coord_t cand_dist  = rx_coord_t(0);
            rx_coord_t cand_theta = rx_coord_t(0);
            if (try_compute_dgpc(coords,
                                 dist,
                                 theta,
                                 vi,
                                 vj,
                                 vk,
                                 radius,
                                 cand_dist,
                                 cand_theta)) {
                if (best_dist == DGPC_INVALID || cand_dist < best_dist) {
                    best_dist  = cand_dist;
                    best_theta = cand_theta;
                    improved   = true;
                }
            }
        }

        if (improved) {
            dist(vi, 0)  = best_dist;
            theta(vi, 0) = best_theta;

            // Mark vi and its one-ring as live for the next sweep.
            // Concurrent writers all write the same value (1) so the race
            // is benign and no atomic is required.
            next_mask(vi, 0) = uint8_t(1);
            for (uint32_t v = 0; v < n; ++v) {
                next_mask(nbrs[v], 0) = uint8_t(1);
            }

            ::atomicAdd(d_changed, 1);
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(
        block, shrd_alloc, relax, in_active_set, /*oriented=*/true);
}

void dgpc(RXMeshStatic& rx)
{
    // ------------------------------------------------------------------
    // Seed parameters. TODO(user): wire to CLI flags.
    // ------------------------------------------------------------------
    uint32_t   seed_face_global = 0;  // global face id of the source facet
    rx_coord_t bc0              = rx_coord_t(1.0 / 3.0);
    rx_coord_t bc1              = rx_coord_t(1.0 / 3.0);
    rx_coord_t bc2              = rx_coord_t(1.0 / 3.0);
    vec3<rx_coord_t> ref_dir(rx_coord_t(1), rx_coord_t(0), rx_coord_t(0));
    vec3<rx_coord_t> second_ref_dir(
        rx_coord_t(0), rx_coord_t(1), rx_coord_t(0));
    rx_coord_t radius_in = rx_coord_t(-1);  // <=0 means "infinite"

    const rx_coord_t radius = (radius_in <= rx_coord_t(0)) ?
                                  std::numeric_limits<rx_coord_t>::max() :
                                  radius_in;

    // Snap bc into the interior if it sits on/outside an edge
    // (matches Lagrange lines 354-359).
    {
        rx_coord_t bcs[3] = {bc0, bc1, bc2};
        rx_coord_t mn     = bcs[0];
        int        mn_i   = 0;
        for (int i = 1; i < 3; ++i) {
            if (bcs[i] < mn) {
                mn   = bcs[i];
                mn_i = i;
            }
        }
        rx_coord_t mx   = bcs[0];
        int        mx_i = 0;
        for (int i = 1; i < 3; ++i) {
            if (bcs[i] > mx) {
                mx   = bcs[i];
                mx_i = i;
            }
        }
        // Vertex-seed branch is detected by max(bc) > 1 - eps; in that case
        // we leave bc unchanged (initialize_from_vertex uses only the seed
        // vertex's one-ring). Otherwise, if the point sits on/outside an
        // edge, nudge it inward.
        if (!(mx > rx_coord_t(1) - DGPC_EPS) && mn < DGPC_EPS) {
            bcs[mn_i] = DGPC_EPS;
            bcs[(mn_i + 1) % 3] =
                rx_coord_t(1) - bcs[(mn_i + 2) % 3] - bcs[mn_i];
            bc0 = bcs[0];
            bc1 = bcs[1];
            bc2 = bcs[2];
        }
        (void)mx_i;
    }

    // Decide seed mode on host.
    rx_coord_t bcs[3]         = {bc0, bc1, bc2};
    int        seed_corner    = 0;  // index into face vertices when vertex-seed
    bool       seed_is_vertex = false;
    {
        rx_coord_t mx = bcs[0];
        for (int i = 1; i < 3; ++i) {
            if (bcs[i] > mx) {
                mx          = bcs[i];
                seed_corner = i;
            }
        }
        if (mx > rx_coord_t(1) - DGPC_EPS) {
            seed_is_vertex = true;
        }
    }

    // ------------------------------------------------------------------
    // Attributes / matrices.
    // ------------------------------------------------------------------
    auto coords  = *rx.get_input_vertex_coordinates();
    auto fnormal = *rx.add_face_attribute<rx_coord_t>("dgpc_fNormal", 3);
    auto dist    = *rx.add_vertex_attribute<rx_coord_t>("dgpc_dist", 1);
    auto theta   = *rx.add_vertex_attribute<rx_coord_t>("dgpc_theta", 1);

    // Active-set masks for frontier-driven relaxation. active_mask drives
    // Query::dispatch's active_set predicate; next_mask collects vertices
    // that need to be revisited in the next sweep.
    auto active_mask = *rx.add_vertex_attribute<uint8_t>("dgpc_active_mask", 1);
    auto next_mask   = *rx.add_vertex_attribute<uint8_t>("dgpc_next_mask", 1);

    dist.reset(DGPC_INVALID, DEVICE);
    theta.reset(rx_coord_t(0), DEVICE);
    active_mask.reset(uint8_t(0), DEVICE);
    next_mask.reset(uint8_t(0), DEVICE);

    DenseMatrix<int> d_changed(rx, 1, 1, LOCATION_ALL);

    // Seed face vertex unique_ids and the seed face normal. We resolve them
    // via a single Op::FV pass and store into 1x1 / 3x1 DenseMatrix buffers
    // so the subsequent Op::VV pass can capture them by value.
    DenseMatrix<uint64_t>   seed_v_buf(rx, 3, 1, LOCATION_ALL);
    DenseMatrix<rx_coord_t> seed_normal_buf(rx, 3, 1, LOCATION_ALL);
    seed_v_buf.reset(uint64_t(0), DEVICE);
    seed_normal_buf.reset(rx_coord_t(0), DEVICE);

    // ------------------------------------------------------------------
    // 1. Per-face normals.
    // ------------------------------------------------------------------
    rx.for_each<Op::FV, 256>(
        [=] __device__(const FaceHandle& fh, const VertexIterator& fv) mutable {
            const vec3<rx_coord_t> v0 = coords.to_glm<3>(fv[0]);
            const vec3<rx_coord_t> v1 = coords.to_glm<3>(fv[1]);
            const vec3<rx_coord_t> v2 = coords.to_glm<3>(fv[2]);
            fnormal.from_glm(fh, glm::normalize(glm::cross(v1 - v0, v2 - v0)));
        });

    // ------------------------------------------------------------------
    // 2. Resolve seed face vertices: write the three VertexHandle unique_ids
    //    of the seed face into seed_v_buf, and copy its normal into
    //    seed_normal_buf. Two steps:
    //      (a) Resolve the seed FaceHandle on the host by global id
    //          (map_to_global is host-only).
    //      (b) Run a device Op::FV pass that, for the matching face,
    //          writes its three vertex uids + normal into the buffers.
    // ------------------------------------------------------------------
    {
        FaceHandle seed_fh;
        rx.for_each_face(
            HOST,
            [&](const FaceHandle fh) {
                if (rx.map_to_global(fh) == seed_face_global) {
                    seed_fh = fh;
                }
            },
            NULL,
            /*with_omp=*/false);

        if (!seed_fh.is_valid()) {
            RXMESH_ERROR(
                "DGPC: source facet global id {} not found in the mesh.",
                seed_face_global);
            return;
        }
        const uint64_t seed_face_uid = seed_fh.unique_id();

        // Step B: device pass to write vertex uids + normal of that face
        // into the dense buffers.
        rx.for_each<Op::FV, 256>(
            [=] __device__(const FaceHandle&     fh,
                           const VertexIterator& fv) mutable {
                if (fh.unique_id() != seed_face_uid) {
                    return;
                }
                seed_v_buf(0, 0)         = fv[0].unique_id();
                seed_v_buf(1, 0)         = fv[1].unique_id();
                seed_v_buf(2, 0)         = fv[2].unique_id();
                const vec3<rx_coord_t> n = fnormal.to_glm<3>(fh);
                seed_normal_buf(0, 0)    = n.x;
                seed_normal_buf(1, 0)    = n.y;
                seed_normal_buf(2, 0)    = n.z;
            });
    }

    // ------------------------------------------------------------------
    // 3. Seed initialization via a single Op::VV oriented pass.
    //    Each thread checks whether its VertexHandle is a seed vertex; if
    //    so it writes dist/theta. Non-seed threads do nothing.
    // ------------------------------------------------------------------
    {
        const rx_coord_t       bc0_c            = bc0;
        const rx_coord_t       bc1_c            = bc1;
        const rx_coord_t       bc2_c            = bc2;
        const vec3<rx_coord_t> ref_dir_c        = ref_dir;
        const vec3<rx_coord_t> second_ref_dir_c = second_ref_dir;
        const bool             seed_is_vertex_c = seed_is_vertex;
        const int              seed_corner_c    = seed_corner;

        rx.for_each<Op::VV, 256>(
            [=] __device__(const VertexHandle&   vh,
                           const VertexIterator& nbrs) mutable {
                const uint64_t uid  = vh.unique_id();
                const uint64_t uid0 = seed_v_buf(0, 0);
                const uint64_t uid1 = seed_v_buf(1, 0);
                const uint64_t uid2 = seed_v_buf(2, 0);

                const vec3<rx_coord_t> seed_normal(seed_normal_buf(0, 0),
                                                   seed_normal_buf(1, 0),
                                                   seed_normal_buf(2, 0));

                // ----------------- Vertex-seed branch -----------------
                if (seed_is_vertex_c) {
                    uint64_t seed_uid = uid0;
                    if (seed_corner_c == 1) {
                        seed_uid = uid1;
                    } else if (seed_corner_c == 2) {
                        seed_uid = uid2;
                    }
                    if (uid != seed_uid) {
                        return;
                    }

                    const uint32_t n = nbrs.size();
                    if (n < 2) {
                        dist(vh, 0)  = rx_coord_t(0);
                        theta(vh, 0) = rx_coord_t(0);
                        return;
                    }

                    // Detect closed vs open ring: closed iff valence ==
                    // (count of consecutive pairs that share a face). Cheap
                    // heuristic: the oriented Op::VV returns the closed
                    // chain on interior vertices and the open chain on
                    // boundary vertices. We treat the chain as closed in
                    // angle accumulation and renormalize total_angle to
                    // 2*pi -- this matches Lagrange's `!on_boundary` path.
                    const bool on_boundary = false;

                    // Project ref direction into tangent plane.
                    vec3<rx_coord_t> ref =
                        ref_dir_c -
                        glm::dot(ref_dir_c, seed_normal) * seed_normal;
                    rx_coord_t rn = glm::length(ref);
                    if (rn < DGPC_EPS) {
                        ref = second_ref_dir_c -
                              glm::dot(second_ref_dir_c, seed_normal) *
                                  seed_normal;
                        rn = glm::length(ref);
                        if (rn < DGPC_EPS) {
                            // Both ref_dir and second_ref_dir parallel to
                            // the seed normal. Pick a tangent.
                            if (fabs(seed_normal.x) < fabs(seed_normal.y)) {
                                ref =
                                    glm::cross(seed_normal,
                                               vec3<rx_coord_t>(rx_coord_t(1),
                                                                rx_coord_t(0),
                                                                rx_coord_t(0)));
                            } else {
                                ref =
                                    glm::cross(seed_normal,
                                               vec3<rx_coord_t>(rx_coord_t(0),
                                                                rx_coord_t(1),
                                                                rx_coord_t(0)));
                            }
                            rn = glm::length(ref);
                        }
                    }
                    ref = ref / rn;

                    const vec3<rx_coord_t> p = coords.to_glm<3>(vh);

                    // Pick start vertex: a face vertex of the seed face that
                    // is NOT the seed itself (matches Lagrange's
                    // start_vertex_id).
                    uint64_t start_uid = uid0;
                    if (start_uid == seed_uid) {
                        start_uid = uid1;
                    }

                    // Find start_vertex_local_id in nbrs.
                    uint32_t start_local = 0;
                    bool     start_found = false;
                    for (uint32_t i = 0; i < n; ++i) {
                        if (nbrs[i].unique_id() == start_uid) {
                            start_local = i;
                            start_found = true;
                            break;
                        }
                    }
                    if (!start_found) {
                        // Fallback: use index 0 as start.
                        start_local = 0;
                    }

                    // Accumulate per-segment angles (closed ring case: use
                    // all n segments with wrap).
                    rx_coord_t total_angle = rx_coord_t(0);
                    // We can't store an array of size n in registers
                    // easily; instead, walk twice: once to compute
                    // total_angle, once to assign cumulative angles.
                    for (uint32_t i = 0; i < n; ++i) {
                        const vec3<rx_coord_t> ec =
                            coords.to_glm<3>(nbrs[i]) - p;
                        const vec3<rx_coord_t> en =
                            coords.to_glm<3>(nbrs[(i + 1) % n]) - p;
                        if (i != n - 1 || !on_boundary) {
                            const rx_coord_t a =
                                atan2(glm::length(glm::cross(ec, en)),
                                      glm::dot(ec, en));
                            total_angle += a;
                        }
                    }
                    if (total_angle <= rx_coord_t(0)) {
                        dist(vh, 0)  = rx_coord_t(0);
                        theta(vh, 0) = rx_coord_t(0);
                        return;
                    }
                    const rx_coord_t scale =
                        on_boundary ? rx_coord_t(1) :
                                      (rx_coord_t(2) * DGPC_PI / total_angle);

                    const vec3<rx_coord_t> start_v =
                        coords.to_glm<3>(nbrs[start_local]);
                    const rx_coord_t start_theta = atan2(
                        glm::dot(glm::cross(ref, start_v - p), seed_normal),
                        glm::dot(ref, start_v - p));

                    // Sum of scaled angles before start_local.
                    rx_coord_t pre_sum = rx_coord_t(0);
                    for (uint32_t i = 0; i < start_local; ++i) {
                        const vec3<rx_coord_t> ec =
                            coords.to_glm<3>(nbrs[i]) - p;
                        const vec3<rx_coord_t> en =
                            coords.to_glm<3>(nbrs[(i + 1) % n]) - p;
                        const rx_coord_t a =
                            atan2(glm::length(glm::cross(ec, en)),
                                  glm::dot(ec, en)) *
                            scale;
                        pre_sum += a;
                    }

                    rx_coord_t angle_cumu = start_theta - pre_sum;

                    for (uint32_t i = 0; i < n; ++i) {
                        if (angle_cumu > DGPC_PI) {
                            angle_cumu -= rx_coord_t(2) * DGPC_PI;
                        } else if (angle_cumu < -DGPC_PI) {
                            angle_cumu += rx_coord_t(2) * DGPC_PI;
                        }

                        const vec3<rx_coord_t> rv = coords.to_glm<3>(nbrs[i]);
                        dist(nbrs[i], 0)          = glm::length(rv - p);
                        theta(nbrs[i], 0)         = angle_cumu;

                        const vec3<rx_coord_t> ec = rv - p;
                        const vec3<rx_coord_t> en =
                            coords.to_glm<3>(nbrs[(i + 1) % n]) - p;
                        const rx_coord_t seg =
                            atan2(glm::length(glm::cross(ec, en)),
                                  glm::dot(ec, en)) *
                            scale;
                        angle_cumu += seg;
                    }

                    dist(vh, 0)  = rx_coord_t(0);
                    theta(vh, 0) = rx_coord_t(0);
                    return;
                }

                // ----------------- Facet-seed branch -----------------
                int corner = -1;
                if (uid == uid0) {
                    corner = 0;
                } else if (uid == uid1) {
                    corner = 1;
                } else if (uid == uid2) {
                    corner = 2;
                }
                if (corner < 0) {
                    return;
                }

                // Project ref direction into the seed face tangent plane.
                vec3<rx_coord_t> ref =
                    ref_dir_c - glm::dot(ref_dir_c, seed_normal) * seed_normal;
                rx_coord_t rn = glm::length(ref);
                if (rn < DGPC_EPS) {
                    ref = second_ref_dir_c -
                          glm::dot(second_ref_dir_c, seed_normal) * seed_normal;
                    rn = glm::length(ref);
                }
                if (rn < DGPC_EPS) {
                    return;
                }
                ref = ref / rn;

                // We need positions of the three seed-face vertices. Each
                // thread for the seed face's vertex sees only its own
                // VertexHandle's neighbors -- the other two seed-face
                // vertices appear in `nbrs` (consecutive in the oriented
                // ring because they share a face with vh).
                vec3<rx_coord_t> v[3];
                v[corner]    = coords.to_glm<3>(vh);
                bool found_a = false, found_b = false;
                for (uint32_t i = 0; i < nbrs.size(); ++i) {
                    const uint64_t nuid = nbrs[i].unique_id();
                    if (corner != 0 && nuid == uid0) {
                        v[0]    = coords.to_glm<3>(nbrs[i]);
                        found_a = (corner == 1) ? true : found_a;
                        found_b = (corner == 2) ? true : found_b;
                    } else if (corner != 1 && nuid == uid1) {
                        v[1] = coords.to_glm<3>(nbrs[i]);
                        if (corner == 0)
                            found_a = true;
                        if (corner == 2)
                            found_b = true;
                    } else if (corner != 2 && nuid == uid2) {
                        v[2] = coords.to_glm<3>(nbrs[i]);
                        if (corner == 0)
                            found_b = true;
                        if (corner == 1)
                            found_b = true;
                    }
                }
                if (!found_a || !found_b) {
                    // Should not happen on a valid manifold mesh, but be
                    // defensive: bail out -- the relaxation step will pick
                    // up from the other two seeded corners.
                }

                const vec3<rx_coord_t> p =
                    bc0_c * v[0] + bc1_c * v[1] + bc2_c * v[2];
                const vec3<rx_coord_t> d = v[corner] - p;
                dist(vh, 0)              = glm::length(d);
                theta(vh, 0) = atan2(glm::dot(glm::cross(ref, d), seed_normal),
                                     glm::dot(ref, d));
            },
            /*oriented=*/true);
    }

    // ------------------------------------------------------------------
    // 3b. Initialize the active set. The relaxation kernel is "for each
    //     vi, try to lower dist(vi) using neighbor (vj, vk) pairs." So
    //     the initial frontier must be the *neighbors* of the seeded
    //     vertices (the seeded ones themselves are already at their
    //     correct value and have nothing to relax against). We compute
    //     this with a single Op::VV pass: a vertex is active iff at
    //     least one of its one-ring neighbors has a valid dist.
    // ------------------------------------------------------------------
    rx.for_each<Op::VV, 256>(
        [=] __device__(const VertexHandle&   vh,
                       const VertexIterator& nbrs) mutable {
            for (uint32_t i = 0; i < nbrs.size(); ++i) {
                if (dist(nbrs[i], 0) != DGPC_INVALID) {
                    active_mask(vh, 0) = uint8_t(1);
                    return;
                }
            }
        });

    // ------------------------------------------------------------------
    // 4. Iterative parallel relaxation (frontier-driven).
    //    Each sweep runs Query::dispatch<Op::VV> with an active_set
    //    predicate so only vertices currently in active_mask execute the
    //    per-triangle DGPC update. On a successful update we mark vi and
    //    its one-ring in next_mask -- those are the candidates for the
    //    next sweep. The fixed point and final values are identical to
    //    the previous full-sweep loop.
    // ------------------------------------------------------------------
    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box({Op::VV},
                          lb,
                          (void*)relax_dgpc_rxmesh<blockThreads>,
                          /*oriented=*/true);

    // Double-buffered active set: ping-pongs between the two attributes
    // each sweep. Uses raw pointers (mirroring apps/Geodesic/geodesic_ptp_
    // rxmesh.h's double_buffer pattern) so we don't copy/swap attribute
    // bookkeeping.
    VertexAttribute<uint8_t>* mask_buf[2] = {&active_mask, &next_mask};
    int                       cur         = 0;

    const int max_outer = 1024;
    for (int it = 0; it < max_outer; ++it) {
        d_changed.reset(0, DEVICE);
        mask_buf[1 - cur]->reset(uint8_t(0), DEVICE);

        relax_dgpc_rxmesh<blockThreads>
            <<<lb.blocks, blockThreads, lb.smem_bytes_dyn>>>(
                rx.get_context(),
                coords,
                dist,
                theta,
                *mask_buf[cur],
                *mask_buf[1 - cur],
                d_changed.data(DEVICE),
                radius);

        d_changed.move(DEVICE, HOST);
        if (d_changed(0, 0) == 0) {
            RXMESH_INFO("DGPC: converged after {} relaxation sweeps.", it + 1);
            break;
        }

        cur = 1 - cur;
    }

    dist.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("Geo", dist);
}

int main(int argc, char** argv)
{
    CLI::App app{"DGPC: Discrete Geodesic Polar Coordinates"};

    std::string mesh_path = STRINGIFY(INPUT_DIR) "sphere3.obj";
    uint32_t    device_id = 0;

    app.add_option("-i,--input", mesh_path, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "sphere3.obj"));

    app.add_option("-d,--device_id", device_id, "GPU device ID")
        ->default_val(0u);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(device_id, spdlog::level::trace);

    RXMeshStatic rx(mesh_path);

    GPUTimer timer;
    timer.start();
    dgpc(rx);
    timer.stop();

    RXMESH_INFO("DGPC took {} (ms) ", timer.elapsed_millis());

#if USE_POLYSCOPE
    polyscope::show();
#endif
}
