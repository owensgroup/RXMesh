#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/context.h"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/dynamic_util.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/types.h"

namespace rxmesh {
namespace detail {

template <uint32_t rowOffset,
          uint32_t blockThreads,
          int      itemPerThread = TRANSPOSE_ITEM_PER_THREAD>
__device__ __forceinline__ void block_mat_transpose(
    const uint32_t  num_rows,
    const uint32_t  num_cols,
    uint16_t*       mat,
    uint16_t*       output,
    const uint32_t* row_active_mask,
    int             shift)
{
    assert(num_rows * rowOffset <= itemPerThread * blockThreads);

    // 1) Load mat into registers and zero out mat
    uint16_t thread_data[itemPerThread];
    uint16_t local_offset[itemPerThread];
    uint32_t nnz = num_rows * rowOffset;

    auto index = [&](uint16_t i) {
        // return itemPerThread * threadIdx.x + i;
        return threadIdx.x + blockThreads * i;
    };

    for (int i = 0; i < itemPerThread; ++i) {
        uint16_t id = index(i);
        // avoid reading out-of-bound from mat
        if (id < nnz) {
            // skip tombstones in mat
            const uint16_t row     = id / rowOffset;
            const bool     deleted = is_deleted(row, row_active_mask);
            const uint16_t val     = mat[id];
            int            pred    = int(val != INVALID16 && !deleted);
            thread_data[i] = pred * (val >> shift) + (1 - pred) * INVALID16;
            mat[id]        = 0;
        } else {
            thread_data[i] = INVALID16;
        }
    }

    if (num_cols > nnz) {
        // zero-ing the rest of mat
        for (uint32_t i = threadIdx.x + nnz; i < num_cols; i += blockThreads) {
            mat[i] = 0;
        }
    }
    /*uint32_t m = max(nnz, num_cols);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < m; i += blockThreads) {
        mat[i] = 0;
    }*/
    __syncthreads();

#if __CUDA_ARCH__ >= 700
    // 2) compute the number of items in each bucket/col
    __half* mat_half = (__half*)(mat);
    for (uint32_t i = 0; i < itemPerThread; ++i) {
        if (thread_data[i] != INVALID16) {
            local_offset[i] = ::atomicAdd(&mat_half[thread_data[i]], 1);
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < num_cols; i += blockThreads) {
        uint16_t val = uint16_t(mat_half[i]);
        mat[i]       = val;
    }
#else
    for (uint32_t i = 0; i < itemPerThread; ++i) {
        if (thread_data[i] != INVALID16) {
            local_offset[i] = atomicAdd(&mat[thread_data[i]], 1u);
        }
    }
    __syncthreads();
#endif

    // 3) exclusive scan on mat to compute the offset
    cub_block_exclusive_sum<uint16_t, blockThreads>(mat, num_cols);

    // 4) actually write the values
    for (uint32_t i = 0; i < itemPerThread; ++i) {
        uint16_t item = thread_data[i];
        if (item != INVALID16) {
            uint16_t offset = mat[item] + local_offset[i];
            uint16_t row    = index(i) / rowOffset;
            output[offset]  = row;
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void e_v_diamond(
    cooperative_groups::thread_block& block,
    const PatchInfo&                  patch_info,
    ShmemAllocator&                   shrd_alloc,
    uint16_t*&                        s_output_value)
{
    const uint16_t num_edges(patch_info.num_edges[0]),
        num_faces(patch_info.num_faces[0]);

    s_output_value = shrd_alloc.alloc<uint16_t>(4 * num_edges);

    uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(3 * num_faces);

    int c = 4 * int(num_edges);
    for (int e = threadIdx.x; e < c; e += blockThreads) {
        s_output_value[e] = INVALID16;
    }
    block.sync();

    load_async(block,
               reinterpret_cast<uint16_t*>(patch_info.fe),
               3 * num_faces,
               s_fe,
               true);

    c = 2 * int(num_edges);

    for (int e = threadIdx.x; e < int(num_edges); e += blockThreads) {
        auto [v0, v1] = patch_info.get_edge_vertices(e);
        s_output_value[4 * e + 0] = v0;
        s_output_value[4 * e + 2] = v1;
    }    
    block.sync();

    c = 3 * int(num_faces);
    for (int f = threadIdx.x; f < c; f += blockThreads) {
        uint16_t face = f / 3;
        if (!is_deleted(face, patch_info.active_mask_f)) {
            uint16_t local_e = f % 3;
            uint16_t edge_i  = INVALID16;
            flag_t   dir_i   = 0;
            Context::unpack_edge_dir(s_fe[f], edge_i, dir_i);

            // the vertex at the end of this edge
            assert(edge_i < num_edges);
            uint16_t id = (4 * edge_i) + (2 * dir_i);
            assert(id < 4 * num_edges);
            uint16_t vertex_i = s_output_value[id];
            assert(vertex_i < patch_info.num_vertices[0]);

            // the edge where vertex_i is oppsoite to it
            uint16_t local_e1 = (local_e + 1) % 3;
            uint16_t edge_i1  = INVALID16;
            flag_t   dir_i1   = 0;
            Context::unpack_edge_dir(
                s_fe[3 * face + local_e1], edge_i1, dir_i1);
            // if dir_i1==0 --> 4 * edge_i1 + 1
            // if dir_i1==1 --> 4 * edge_i1 + 3
            id = (4 * edge_i1 + 1) + (2 * dir_i1);
            assert(id < 4 * num_edges);
            s_output_value[id] = vertex_i;
        }
    }

    shrd_alloc.dealloc<uint16_t>(3 * num_faces);
}


template <uint32_t blockThreads>
__device__ __forceinline__ void e_e_manifold(
    cooperative_groups::thread_block& block,
    const PatchInfo&                  patch_info,
    ShmemAllocator&                   shrd_alloc,
    uint16_t*&                        s_output_value)
{

    // works for edge-manifold as we assume that each edge is incident to
    // only 4 other edges (or 2 in case of boundary edges).
    const uint16_t num_edges(patch_info.num_edges[0]),
        num_faces(patch_info.num_faces[0]);

    int c = 4 * int(num_edges);

    s_output_value = shrd_alloc.alloc<uint16_t>(c);


    for (int e = threadIdx.x; e < c; e += blockThreads) {
        s_output_value[e] = INVALID16;
    }
    block.sync();

    for (int f = threadIdx.x; f < int(num_faces); f += blockThreads) {

        if (!is_deleted(f, patch_info.active_mask_f)) {

            uint16_t f_e[3];
            flag_t   f_dir[3];
            for (int i = 0; i < 3; ++i) {
                f_e[i] = patch_info.fe[3 * f + i].id;
                Context::unpack_edge_dir(f_e[i], f_e[i], f_dir[i]);
                assert(f_e[i] < num_edges);
                assert(!is_deleted(f_e[i], patch_info.active_mask_e));
            }


            for (int cur = 0; cur < 3; ++cur) {
                const int nxt = (cur + 1) % 3;
                const int prv = (cur - 1) % 3;

                const uint16_t cur_e = f_e[cur];
                const uint16_t nxt_e = f_e[nxt];
                const uint16_t prv_e = f_e[prv];

                // in case of oriented faces, we use the edge direction to guide
                // where we should write the edges
                int nxt_i = 4 * cur_e + 2 * f_dir[cur] + 0;
                int prv_i = 4 * cur_e + 2 * f_dir[cur] + 1;

                uint16_t ret_n =
                    ::atomicCAS(s_output_value + nxt_i, INVALID16, nxt_e);

                uint16_t ret_p =
                    ::atomicCAS(s_output_value + prv_i, INVALID16, prv_e);

                if (ret_n != INVALID16) {
                    assert(ret_p == INVALID16);

                    int nxt_i = 4 * cur_e + 2 * (f_dir[cur] ^ 1) + 0;
                    int prv_i = 4 * cur_e + 2 * (f_dir[cur] ^ 1) + 1;

                    ret_n =
                        ::atomicCAS(s_output_value + nxt_i, INVALID16, nxt_e);

                    ret_p =
                        ::atomicCAS(s_output_value + prv_i, INVALID16, prv_e);

                    assert(ret_p == INVALID16);
                    assert(ret_n == INVALID16);
                }
            }
        }
    }
}


template <uint32_t blockThreads>
__device__ __forceinline__ void e_f_manifold(const uint16_t  num_edges,
                                             const uint16_t  num_faces,
                                             const uint16_t* s_fe,
                                             uint16_t*       s_ef,
                                             const uint32_t* active_mask_f)
{
    // s_ef should be filled with INVALID16 before calling this function
    int c = 3 * int(num_faces);
    for (int e = threadIdx.x; e < c; e += blockThreads) {
        uint16_t face_id = e / 3;

        assert(face_id < num_faces);

        if (!is_deleted(face_id, active_mask_f)) {
            uint16_t edge = s_fe[e] >> 1;

            assert(edge < num_edges);

            auto ret = atomicCAS(s_ef + 2 * edge, INVALID16, face_id);
            if (ret != INVALID16) {
                ret = atomicCAS(s_ef + 2 * edge + 1, INVALID16, face_id);
                // if (ret != INVALID16) {
                //     printf("\n B= %u", blockIdx.x);
                // }
                assert(ret == INVALID16);
            }
        }
    }
}
template <uint32_t blockThreads>
__device__ __forceinline__ void orient_edges_around_vertices(
    const PatchInfo& patch_info,
    ShmemAllocator&  shrd_alloc,
    uint16_t*&       s_output_offset,
    uint16_t*&       s_output_value)
{
    const uint16_t num_edges    = patch_info.num_edges[0];
    const uint16_t num_faces    = patch_info.num_faces[0];
    const uint16_t num_vertices = patch_info.num_vertices[0];


    // start by loading the faces while also doing transposing EV
    uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(3 * num_faces);
    uint16_t* s_ef = shrd_alloc.alloc<uint16_t>(2 * num_edges);

    for (uint32_t i = threadIdx.x; i < 2 * num_edges; i += blockThreads) {
        s_ef[i] = INVALID16;
    }

    load_async(reinterpret_cast<const uint16_t*>(patch_info.fe),
               3 * num_faces,
               reinterpret_cast<uint16_t*>(s_fe),
               true);

    // We could have used block_mat_transpose to transpose FE so we can look
    // up the "two" faces sharing an edge. But we can do better because we know
    // that we are working on manifold so it is only two edges per face. We
    // also wanna keep FE for quick look up on a face's three edges.

    __syncthreads();

    e_f_manifold<blockThreads>(
        num_edges, num_faces, s_fe, s_ef, patch_info.active_mask_f);

    // To orient, we pin the first edge and check all the subsequent edges
    // For each edge, we search for the two faces containing it (should be
    // only two faces since this is a manifold mesh).
    __syncthreads();

    // TODO check active_mask_v
    for (uint32_t v = threadIdx.x; v < num_vertices; v += blockDim.x) {

        // TODO if the vertex is not owned by this patch, then there is no
        // reason to orient its edges because no serious computation is done on
        // it

        int start = int(s_output_offset[v]);
        int end   = int(s_output_offset[v + 1]);


        assert(end >= start);
        uint16_t start_id = start;

        // if the mesh is not closed, pick a boundary edge as starting point
        // TODO we may eliminate this in case of closed mesh
        for (int e_id = start; e_id < end; ++e_id) {
            uint16_t e_0 = s_output_value[e_id];
            uint16_t f0(s_ef[2 * e_0]), f1(s_ef[2 * e_0 + 1]);
            if (f0 == INVALID16 || f1 == INVALID16) {
                start_id = e_id;
                break;
            }
        }

        uint16_t e_id = start_id;

        uint16_t edges_count = 0;
        while (true) {

            uint16_t e_0 = s_output_value[e_id];
            uint16_t f0(s_ef[2 * e_0]), f1(s_ef[2 * e_0 + 1]);

            // candidate next edge (only one of them will win)
            uint16_t e_candid_0(INVALID16), e_candid_1(INVALID16);

            if (f0 != INVALID16) {
                if ((s_fe[3 * f0 + 0] >> 1) == e_0) {
                    e_candid_0 = s_fe[3 * f0 + 2] >> 1;
                }
                if ((s_fe[3 * f0 + 1] >> 1) == e_0) {
                    e_candid_0 = s_fe[3 * f0 + 0] >> 1;
                }
                if ((s_fe[3 * f0 + 2] >> 1) == e_0) {
                    e_candid_0 = s_fe[3 * f0 + 1] >> 1;
                }
            }

            if (f1 != INVALID16) {
                if ((s_fe[3 * f1 + 0] >> 1) == e_0) {
                    e_candid_1 = s_fe[3 * f1 + 2] >> 1;
                }
                if ((s_fe[3 * f1 + 1] >> 1) == e_0) {
                    e_candid_1 = s_fe[3 * f1 + 0] >> 1;
                }
                if ((s_fe[3 * f1 + 2] >> 1) == e_0) {
                    e_candid_1 = s_fe[3 * f1 + 1] >> 1;
                }
            }

            for (int vn = e_id + 1; vn < end; ++vn) {
                uint16_t e_winning_candid = s_output_value[vn];
                if (e_candid_0 == e_winning_candid ||
                    e_candid_1 == e_winning_candid) {
                    uint16_t temp            = s_output_value[e_id + 1];
                    s_output_value[e_id + 1] = e_winning_candid;
                    s_output_value[vn]       = temp;
                    break;
                }
            }

            edges_count++;
            if (edges_count > end - start - 1) {
                break;
            }
            e_id = ((e_id - start + 1) % (end - start)) + start;
        }
    }

    shrd_alloc.dealloc<uint16_t>(2 * num_edges);
    shrd_alloc.dealloc<uint16_t>(3 * num_faces);
}

template <uint32_t blockThreads,
          uint32_t itemPerThread = TRANSPOSE_ITEM_PER_THREAD>
__device__ __forceinline__ void v_e(const uint16_t  num_vertices,
                                    const uint16_t  num_edges,
                                    uint16_t*       d_edges,
                                    uint16_t*       d_output,
                                    const uint32_t* active_mask_e)
{
    // M_ve = M_ev^{T}. M_ev is already encoded and we need to just transpose
    // it
    // Here we do the transpose in place and the result is that d_output
    // contains the row id of the transpose matrix (i.e. the edges id) while
    // d_edges will contain the offset that starts with zero and end with
    // num_edges*2 (zero is stored and the end can be inferred). Thus,
    // d_output should be allocated to size = num_edges*2

    block_mat_transpose<2u, blockThreads, itemPerThread>(
        num_edges, num_vertices, d_edges, d_output, active_mask_e, 0);
}

template <uint32_t blockThreads>
__device__ __forceinline__ void v_v(cooperative_groups::thread_block& block,
                                    const PatchInfo& patch_info,
                                    ShmemAllocator&  shrd_alloc,
                                    uint16_t*        s_output_offset,
                                    uint16_t*        s_output_value,
                                    bool             oriented,
                                    bool             smem_dup)
{
    // smem_dup indicate if we should store the duplicated EV in shared memory
    // if oriented is false, we have the option to either store the duplicated
    // EV in shared memory (which is allocated/de-allocated inside this
    // function), or just read it from global memory. If oriented is true, then
    // we don't have an option since oriented requires a lot of shared memory
    // that is okay to reuse to store the duplicates
    //
    //  M_vv = M_EV^{T} \dot M_EV
    //  This requires computing M_EV^{T} which we compute in shared memory
    //  similar to v_e. Doing that, we have store in s_output_value the edges
    //  incident to each vertex. After that we need to replace each edge with
    //  the other end vertex which is duplicated by writing it to
    //  s_edges_duplicate
    const uint16_t  num_vertices   = patch_info.num_vertices[0];
    const uint16_t  num_edges      = patch_info.num_edges[0];
    const uint32_t* active_mask_e  = patch_info.active_mask_e;
    const uint32_t* active_mask_v  = patch_info.active_mask_v;
    uint16_t*       s_ev_duplicate = nullptr;

    // assert(2 * 2 * num_edges >= num_vertices + 1 + 2 * num_edges);

    if (!oriented && smem_dup) {
        s_ev_duplicate =
            shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges[0]);
        for (int i = threadIdx.x; i < 2 * num_edges; i += blockThreads) {
            s_ev_duplicate[i] = s_output_offset[i];
        }
        // we should sync here to avoid writing to s_ev before reading it into
        // s_ev_duplicate but we rely on the sync in block_mat_transpose
    } else if (!smem_dup) {
        s_ev_duplicate = reinterpret_cast<uint16_t*>(patch_info.ev);
    }

    v_e<blockThreads>(num_vertices,
                      num_edges,
                      s_output_offset,
                      s_output_value,
                      active_mask_e);

    if (oriented) {
        block.sync();

        orient_edges_around_vertices<blockThreads>(
            patch_info, shrd_alloc, s_output_offset, s_output_value);

        block.sync();

        s_ev_duplicate =
            shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges[0]);

        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * num_edges,
                   s_ev_duplicate,
                   true);
    }

    block.sync();

    // TODO we can load-balance this better than this
    for (uint32_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        uint32_t start = s_output_offset[v];
        uint32_t end   = s_output_offset[v + 1];

        for (uint32_t e = start; e < end; ++e) {
            uint16_t edge = s_output_value[e];
            uint16_t v0   = s_ev_duplicate[2 * edge];
            uint16_t v1   = s_ev_duplicate[2 * edge + 1];

            assert(v0 != INVALID16 && v1 != INVALID16);
            assert(v0 == v || v1 == v);
            // s_output_value[e] = (v0 == v) ? v1 : v0;
            s_output_value[e] = (v0 == v) * v1 + (v1 == v) * v0;
        }
    }

    if ((!oriented && smem_dup) || oriented) {
        shrd_alloc.dealloc<uint16_t>(2 * patch_info.num_edges[0]);
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void f_v(const uint16_t  num_edges,
                                    const uint16_t* d_edges,
                                    const uint16_t  num_faces,
                                    uint16_t*       d_faces,
                                    const uint32_t* active_mask_f)
{
    // M_FV = M_FE \dot M_EV

    // Hint: Since a single thread is responsible of reading one
    // face in d_faces (i.e., three items), then this thread
    // can safely over-write what is in d_faces.

    for (uint32_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        uint16_t f_v[3];
        uint32_t f_id = 3 * f;
        if (!is_deleted(f, active_mask_f)) {
            for (uint32_t i = 0; i < 3; i++) {
                uint16_t e = d_faces[f_id + i];
                if (e == INVALID16) {
                    f_v[i] = INVALID16;
                    continue;
                }
                flag_t e_dir(0);
                Context::unpack_edge_dir(e, e, e_dir);
                // if the direction is flipped, we take the second vertex
                uint16_t e_id = (2 * e) + (1 * e_dir);
                assert(e_id < 2 * num_edges);
                f_v[i] = d_edges[e_id];
            }
            for (uint32_t i = 0; i < 3; i++) {
                d_faces[f_id + i] = f_v[i];
            }
        } else {
            for (uint32_t i = 0; i < 3; i++) {
                d_faces[f_id + i] = INVALID16;
            }
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void v_f(const uint16_t  num_faces,
                                    const uint16_t  num_edges,
                                    const uint16_t  num_vertices,
                                    uint16_t*       d_edges,
                                    uint16_t*       d_faces,
                                    const uint32_t* active_mask_f)
{
    // M_vf = M_ev^{T} \dot M_fe^{T} = (M_ev \dot M_fe)^{T} = M_fv^{T}

    // We follow the math here by computing M_fv and then transpose it
    // In doing so we reuse all the shared memory used to store d_edges
    // and d_faces
    // First M_fv is computing in place i.e., d_face will contain the
    // face vertices of each face (instead of edges)
    // Second, the transpose happens in place i.e., d_faces will hold the
    // offset and d_edges will hold the value (row id)

    f_v<blockThreads>(num_edges, d_edges, num_faces, d_faces, active_mask_f);
    __syncthreads();

    block_mat_transpose<3u, blockThreads>(
        num_faces, num_vertices, d_faces, d_edges, active_mask_f, 0);
}

template <uint32_t blockThreads>
__device__ __forceinline__ void e_f(const uint16_t  num_edges,
                                    const uint16_t  num_faces,
                                    uint16_t*       d_faces,
                                    uint16_t*       d_output,
                                    const uint32_t* active_mask_f,
                                    int             shift = 1)
{
    // M_ef = M_fe^{T}. M_fe is already encoded and we need to just transpose
    // it

    // Here we do the transpose in place and the result is that d_output
    // contains the row id of the transpose matrix (i.e. the faces id) while
    // d_faces will contain the offset that starts with zero and end with
    // num_faces*3 (zero is stored and the end can be inferred). Thus,
    // d_output should be allocated to size = num_faces*3

    block_mat_transpose<3u, blockThreads>(
        num_faces, num_edges, d_faces, d_output, active_mask_f, shift);
}

template <uint32_t blockThreads>
__device__ __forceinline__ void f_f(const uint16_t  num_edges,
                                    const uint16_t  num_faces,
                                    uint16_t*       s_FE,
                                    uint16_t*       s_EF_offset,
                                    uint16_t*       s_FF_offset,
                                    uint16_t*       s_FF_output,
                                    const uint32_t* active_mask_f)
{
    // First construct M_EF in shared memory
    uint16_t* s_EF_output = &s_EF_offset[num_edges + 1];

    // copy FE in to EF_offset so we can do the transpose in place without
    // losing FE
    for (int i = threadIdx.x; i < num_faces * 3; i += blockThreads) {
        flag_t   dir(0);
        uint16_t e     = s_FE[i] >> 1;
        s_EF_offset[i] = e;
        s_FE[i]        = e;
    }
    __syncthreads();

    e_f<blockThreads>(
        num_edges, num_faces, s_EF_offset, s_EF_output, active_mask_f, 0);
    __syncthreads();

    // Every thread (T) is responsible for a face (F)
    // Each thread reads the edges (E) incident to its face (F). For each edge
    // (E), we read the "number" of incident faces (FF) to this edge (num_EF).
    // The number neighbor edges to the face F due to edge E is num_EF -1

    // TODO we can store this sum of neighbor faces in registers and then do
    // the exclusive sum on it and finally store it in shared memory
    for (int f = threadIdx.x; f < int(num_faces); f += blockThreads) {
        uint16_t num_neighbour_faces = 0;
        for (int e = 0; e < 3; ++e) {
            uint16_t edge = s_FE[3 * f + e];

            assert(s_EF_offset[edge + 1] >= s_EF_offset[edge]);

            num_neighbour_faces +=
                s_EF_offset[edge + 1] - s_EF_offset[edge] - 1;
        }
        s_FF_offset[f] = num_neighbour_faces;
    }
    __syncthreads();

    cub_block_exclusive_sum<uint16_t, blockThreads>(s_FF_offset, num_faces);

    for (int f = threadIdx.x; f < int(num_faces); f += blockThreads) {
        uint16_t offset = s_FF_offset[f];
        for (int e = 0; e < 3; ++e) {
            uint16_t edge = s_FE[3 * f + e];
            for (int ef = s_EF_offset[edge]; ef < int(s_EF_offset[edge + 1]);
                 ++ef) {
                uint16_t n_face = s_EF_output[ef];
                if (n_face != f) {
                    s_FF_output[offset] = n_face;
                    ++offset;
                }
            }
        }
        assert(offset == s_FF_offset[f + 1]);
    }
}

template <uint32_t blockThreads, Op op>
__device__ __forceinline__ void query(cooperative_groups::thread_block& block,
                                      const PatchInfo& patch_info,
                                      ShmemAllocator&  shrd_alloc,
                                      uint16_t*&       s_output_offset,
                                      uint16_t*&       s_output_value,
                                      bool             oriented)
{

    if constexpr (op == Op::VV) {
        // assert(patch_info.num_vertices[0] <= 2 * patch_info.num_edges[0]);
        uint16_t* s_ev =
            shrd_alloc.alloc<uint16_t>(std::max(patch_info.num_vertices[0] + 1,
                                                2 * patch_info.num_edges[0]) +
                                       2 * patch_info.num_edges[0]);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * patch_info.num_edges[0],
                   s_ev,
                   true);
        s_output_offset = &s_ev[0];
        s_output_value  = &s_ev[patch_info.num_vertices[0] + 1];
        v_v<blockThreads>(block,
                          patch_info,
                          shrd_alloc,
                          s_output_offset,
                          s_output_value,
                          oriented,
                          true);
    }

    if constexpr (op == Op::VE) {
        // assert(patch_info.num_vertices[0] <= 2 * patch_info.num_edges[0]);
        uint16_t* s_ev =
            shrd_alloc.alloc<uint16_t>(std::max(patch_info.num_vertices[0] + 1,
                                                2 * patch_info.num_edges[0]) +
                                       2 * patch_info.num_edges[0]);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * patch_info.num_edges[0],
                   s_ev,
                   true);
        s_output_offset = s_ev;
        s_output_value  = &s_ev[patch_info.num_vertices[0] + 1];
        v_e<blockThreads>(patch_info.num_vertices[0],
                          patch_info.num_edges[0],
                          s_ev,
                          s_output_value,
                          patch_info.active_mask_e);
        if (oriented) {
            orient_edges_around_vertices<blockThreads>(
                patch_info, shrd_alloc, s_output_offset, s_output_value);
        }
    }

    if constexpr (op == Op::VF) {
        // assert(patch_info.num_vertices[0] <= 2 * patch_info.num_edges[0]);
        uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(std::max(
            3 * patch_info.num_faces[0], 1 + patch_info.num_vertices[0]));
        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(
            std::max(2 * patch_info.num_edges[0], 3 * patch_info.num_faces[0]));
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces[0],
                   s_fe,
                   false);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * patch_info.num_edges[0],
                   s_ev,
                   true);
        s_output_offset = &s_fe[0];
        s_output_value  = &s_ev[0];
        v_f<blockThreads>(patch_info.num_faces[0],
                          patch_info.num_edges[0],
                          patch_info.num_vertices[0],
                          s_ev,
                          s_fe,
                          patch_info.active_mask_f);
    }

    if constexpr (op == Op::EV) {
        s_output_value =
            shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges[0]);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * patch_info.num_edges[0],
                   s_output_value,
                   true);
    }

    if constexpr (op == Op::EF) {
        assert(patch_info.num_edges[0] <= 3 * patch_info.num_faces[0]);
        uint16_t* s_fe =
            shrd_alloc.alloc<uint16_t>(2 * 3 * patch_info.num_faces[0]);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces[0],
                   s_fe,
                   true);
        s_output_offset = &s_fe[0];
        s_output_value  = &s_fe[patch_info.num_edges[0] + 1];
        e_f<blockThreads>(patch_info.num_edges[0],
                          patch_info.num_faces[0],
                          s_fe,
                          s_output_value,
                          patch_info.active_mask_f,
                          1);
    }

    if constexpr (op == Op::FV) {
        // alloc and load FE and EV, operate on FE to convert it to FV
        // then dealloc EV
        uint16_t* s_fe =
            shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces[0]);
        uint16_t* s_ev =
            shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges[0]);
        s_output_value = s_fe;
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * patch_info.num_edges[0],
                   s_ev,
                   false);

        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces[0],
                   s_fe,
                   true);

        f_v<blockThreads>(patch_info.num_edges[0],
                          s_ev,
                          patch_info.num_faces[0],
                          s_fe,
                          patch_info.active_mask_f);

        shrd_alloc.dealloc<uint16_t>(2 * patch_info.num_edges[0]);
    }

    if constexpr (op == Op::FE) {
        s_output_value =
            shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces[0]);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces[0],
                   s_output_value,
                   true);
    }

    if constexpr (op == Op::FF) {
        assert(patch_info.num_edges[0] <= 3 * patch_info.num_faces[0]);
        s_output_offset =
            shrd_alloc.alloc<uint16_t>(4 * patch_info.num_faces[0]);
        s_output_value = &s_output_offset[patch_info.num_faces[0] + 1];

        uint16_t* s_fe =
            shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces[0]);
        uint16_t* s_ef =
            shrd_alloc.alloc<uint16_t>(2 * 3 * patch_info.num_faces[0]);

        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces[0],
                   s_fe,
                   true);

        f_f<blockThreads>(patch_info.num_edges[0],
                          patch_info.num_faces[0],
                          s_fe,
                          s_ef,
                          s_output_offset,
                          s_output_value,
                          patch_info.active_mask_f);

        shrd_alloc.dealloc<uint16_t>(2 * 3 * patch_info.num_faces[0]);
        shrd_alloc.dealloc<uint16_t>(3 * patch_info.num_faces[0]);
    }

    if constexpr (op == Op::EVDiamond) {
        e_v_diamond<blockThreads>(
            block, patch_info, shrd_alloc, s_output_value);
    }

    if constexpr (op == Op::EE) {
        e_e_manifold<blockThreads>(
            block, patch_info, shrd_alloc, s_output_value);
    }
}

}  // namespace detail
}  // namespace rxmesh
