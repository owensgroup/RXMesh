#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/rxmesh_loader.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/rxmesh.h"
#include "rxmesh/rxmesh_context.h"

namespace rxmesh {
template <uint32_t rowOffset,
          uint32_t blockThreads,
          uint32_t itemPerThread = TRANSPOSE_ITEM_PER_THREAD>
__device__ __forceinline__ void block_mat_transpose(const uint32_t num_rows,
                                                    const uint32_t num_cols,
                                                    uint16_t*      mat,
                                                    uint16_t*      output,
                                                    int            shift = 0)
{
    // 1) Load mat into registers and zero out mat
    uint16_t thread_data[itemPerThread];
    uint16_t local_offset[itemPerThread];
    uint32_t nnz = num_rows * rowOffset;

    for (uint32_t i = 0; i < itemPerThread; ++i) {
        uint32_t index = itemPerThread * threadIdx.x + i;
        // TODO
        // int      pred = int(index < nnz);
        // thread_data[i] = pred * (mat[index] >> shift) + (1 - pred) *
        // INVALID16;
        if (index < nnz) {
            thread_data[i] = mat[index] >> shift;
            mat[index]     = 0;
        } else {
            thread_data[i] = INVALID16;
        }
    }

    /*if (num_cols > nnz) {
        // zero-ing the rest of mat
        for (uint32_t i = threadIdx.x + nnz; i < num_cols; i += blockThreads) {
            mat[i] = 0;
        }
    }*/
    uint32_t m = max(nnz, num_cols);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < m; i += blockThreads) {
        mat[i] = 0;
    }
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
        } else {
            break;
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
            uint16_t row    = (itemPerThread * threadIdx.x + i) / rowOffset;
            output[offset]  = row;
        } else {
            break;
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void v_v_oreinted(uint16_t*& s_offset_all_patches,
                                             uint16_t*& s_output_all_patches,
                                             uint16_t*  s_patch_edges,
                                             const RXMeshContext& context,
                                             const uint4&         ad_size,
                                             const uint16_t       num_vertices,
                                             const uint16_t num_owned_vertices)
{
    const uint32_t num_faces = ad_size.w / 3;
    const uint32_t num_edges = ad_size.y / 2;

    s_offset_all_patches = &s_patch_edges[0];
    s_output_all_patches =
        &s_patch_edges[num_vertices + 1 + (num_vertices + 1) % 2];

    // start by loading the faces while also doing transposing EV (might
    // increase ILP)
    uint16_t* s_patch_FE = &s_output_all_patches[2 * num_edges];
    uint16_t* s_patch_EF = &s_patch_FE[3 * num_faces + (3 * num_faces) % 2];
    load_patch_faces(context, s_patch_FE, ad_size);

    for (uint32_t i = threadIdx.x; i < num_edges * 2; i += blockThreads) {
        s_patch_EF[i] = INVALID16;
    }

    block_mat_transpose<2u, blockThreads>(
        num_edges, num_vertices, s_offset_all_patches, s_output_all_patches);

    // block_mat_transpose<2u, blockThreads>(
    //    num_faces, num_edges, s_patch_EF_offset, s_patch_EF_output);

    // We could have used block_mat_transpose to transpose FE so we can look
    // up the "two" faces sharing an edge. But we can do better because we know
    // that we are working on manifold so it is only two edges per face. We
    // also wanna keep FE for quick look up on a face's three edges.

    // We need to sync here to make sure that s_patch_FE is loaded but there is
    // a sync in block_mat_transpose that takes care of this


    for (uint16_t e = threadIdx.x; e < 3 * num_faces; e += blockThreads) {
        uint16_t edge    = s_patch_FE[e] >> 1;
        uint16_t face_id = e / 3;

        auto ret = atomicCAS(s_patch_EF + 2 * edge, INVALID16, face_id);
        if (ret != INVALID16) {
            ret = atomicCAS(s_patch_EF + 2 * edge + 1, INVALID16, face_id);
            assert(ret == INVALID16);
        }
    }

    // To orient, we pin the first edge and check all the subsequent edges
    // For each edge, we search for the two faces containing it (should be
    // only two faces since this is a manifold mesh).
    __syncthreads();

    for (uint32_t v = threadIdx.x; v < num_owned_vertices; v += blockDim.x) {

        // if the vertex is not owned by this patch, then there is no reason
        // to orient its edges because no serious computation is done on it

        uint16_t start = s_offset_all_patches[v];
        uint16_t end   = s_offset_all_patches[v + 1];


        for (uint16_t e_id = start; e_id < end - 1; ++e_id) {
            uint16_t e_0 = s_output_all_patches[e_id];
            uint16_t f0(s_patch_EF[2 * e_0]), f1(s_patch_EF[2 * e_0 + 1]);

            // we don't do it for boundary faces
            assert(f0 != INVALID16 && f1 != INVALID16 && f0 < num_faces &&
                   f1 < num_faces);


            // candidate next edge (only one of them will win)
            uint16_t e_candid_0, e_candid_1;

            if ((s_patch_FE[3 * f0 + 0] >> 1) == e_0) {
                e_candid_0 = s_patch_FE[3 * f0 + 2] >> 1;
            }
            if ((s_patch_FE[3 * f0 + 1] >> 1) == e_0) {
                e_candid_0 = s_patch_FE[3 * f0 + 0] >> 1;
            }
            if ((s_patch_FE[3 * f0 + 2] >> 1) == e_0) {
                e_candid_0 = s_patch_FE[3 * f0 + 1] >> 1;
            }

            if ((s_patch_FE[3 * f1 + 0] >> 1) == e_0) {
                e_candid_1 = s_patch_FE[3 * f1 + 2] >> 1;
            }
            if ((s_patch_FE[3 * f1 + 1] >> 1) == e_0) {
                e_candid_1 = s_patch_FE[3 * f1 + 0] >> 1;
            }
            if ((s_patch_FE[3 * f1 + 2] >> 1) == e_0) {
                e_candid_1 = s_patch_FE[3 * f1 + 1] >> 1;
            }

            for (uint16_t vn = e_id + 1; vn < end; ++vn) {
                uint16_t e_winning_candid = s_output_all_patches[vn];
                if (e_candid_0 == e_winning_candid ||
                    e_candid_1 == e_winning_candid) {
                    uint16_t temp = s_output_all_patches[e_id + 1];
                    s_output_all_patches[e_id + 1] = e_winning_candid;
                    s_output_all_patches[vn]       = temp;
                    break;
                }
            }
        }
    }

    __syncthreads();

    // Load EV into s_patch_EF since both has the same size (2*#E)
    s_patch_edges = &s_patch_EF[0];
    load_patch_edges(context, s_patch_edges, ad_size);
    __syncthreads();

    for (uint32_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        uint32_t start = s_offset_all_patches[v];
        uint32_t end   = s_offset_all_patches[v + 1];


        for (uint32_t e = start; e < end; ++e) {
            uint16_t edge = s_output_all_patches[e];
            uint16_t v0   = s_patch_edges[2 * edge];
            uint16_t v1   = s_patch_edges[2 * edge + 1];

            assert(v0 == v || v1 == v);
            // d_output[e] = (v0 == v) ? v1 : v0;
            s_output_all_patches[e] = (v0 == v) * v1 + (v1 == v) * v0;
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void v_e(const uint32_t num_vertices,
                                    const uint32_t num_edges,
                                    uint16_t*      d_edges,
                                    uint16_t*      d_output)
{
    // M_ve = M_ev^{T}. M_ev is already encoded and we need to just transpose
    // it
    // Here we do the transpose in place and the result is that d_output
    // contains the row id of the transpose matrix (i.e. the edges id) while
    // d_edges will contain the offset that starts with zero and end with
    // num_edges*2 (zero is stored and the end can be inferred). Thus,
    // d_output should be allocated to size = num_edges*2

    block_mat_transpose<2u, blockThreads>(
        num_edges, num_vertices, d_edges, d_output);
}

template <uint32_t blockThreads>
__device__ __forceinline__ void v_v(const uint32_t num_vertices,
                                    const uint32_t num_edges,
                                    uint16_t*      d_edges,
                                    uint16_t*      d_output)
{
    // M_vv = M_EV^{T} \dot M_EV
    // This requires computing M_EV^{T} which we compute in shared memory
    // similar to v_e. Doing that, we have store in d_output the edges
    // incident to each vertex. After that we need to replace each edge with
    // the other end vertex which is duplicated by writing it to
    // s_edges_duplicate

    uint16_t* s_edges_duplicate = &d_edges[2 * 2 * num_edges];

    assert(2 * 2 * num_edges >= num_vertices + 1 + 2 * num_edges);

    for (uint16_t i = threadIdx.x; i < 2 * num_edges; i += blockThreads) {
        s_edges_duplicate[i] = d_edges[i];
    }

    // TODO we might be able to remove this sync if transpose has a sync
    // that is done before writing to mat
    __syncthreads();

    v_e<blockThreads>(num_vertices, num_edges, d_edges, d_output);

    __syncthreads();

    for (uint32_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        uint32_t start = d_edges[v];
        uint32_t end   = d_edges[v + 1];

        for (uint32_t e = start; e < end; ++e) {
            uint16_t edge = d_output[e];
            uint16_t v0   = s_edges_duplicate[2 * edge];
            uint16_t v1   = s_edges_duplicate[2 * edge + 1];

            assert(v0 == v || v1 == v);
            // d_output[e] = (v0 == v) ? v1 : v0;
            d_output[e] = (v0 == v) * v1 + (v1 == v) * v0;
        }
    }
}

__device__ __forceinline__ void f_v(const uint32_t  num_edges,
                                    const uint16_t* d_edges,
                                    const uint32_t  num_faces,
                                    uint16_t*       d_faces)
{
    // M_FV = M_FE \dot M_EV

    // Hint: Since a single thread is responsible of reading one
    // face in d_faces (i.e., three items), then this thread
    // can safely over-write what is in d_faces.

    for (uint32_t f = threadIdx.x; f < num_faces; f += blockDim.x) {
        uint16_t f_v[3];
        uint32_t f_id = 3 * f;
        // TODO use vector load and store instead of looping
        for (uint32_t i = 0; i < 3; i++) {
            uint16_t e = d_faces[f_id + i];
            flag_t   e_dir(0);
            RXMeshContext::unpack_edge_dir(e, e, e_dir);
            // if the direction is flipped, we take the second vertex
            uint16_t e_id = (2 * e) + (1 * e_dir);
            assert(e_id < 2 * num_edges);
            f_v[i] = d_edges[e_id];
        }
        for (uint32_t i = 0; i < 3; i++) {
            d_faces[f * 3 + i] = f_v[i];
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void v_f(const uint32_t num_faces,
                                    const uint32_t num_edges,
                                    const uint32_t num_vertices,
                                    uint16_t*      d_edges,
                                    uint16_t*      d_faces)
{
    // M_vf = M_ev^{T} \dot M_fe^{T} = (M_ev \dot M_fe)^{T} = M_fv^{T}

    // We follow the math here by computing M_fv and then transpose it
    // In doing so we reuse all the shared memory used to store d_edges
    // and d_faces
    // First M_fv is computing in place i.e., d_face will contain the
    // face vertices of each face (instead of edges)
    // Second, the transpose happens in place i.e., d_faces will hold the
    // offset and d_edges will hold the value (row id)

    f_v(num_edges, d_edges, num_faces, d_faces);
    __syncthreads();

    block_mat_transpose<3u, blockThreads>(
        num_faces, num_vertices, d_faces, d_edges);
}

template <uint32_t blockThreads>
__device__ __forceinline__ void e_f(const uint32_t num_edges,
                                    const uint32_t num_faces,
                                    uint16_t*      d_faces,
                                    uint16_t*      d_output,
                                    int            shift = 1)
{
    // M_ef = M_fe^{T}. M_fe is already encoded and we need to just transpose
    // it

    // Here we do the transpose in place and the result is that d_output
    // contains the row id of the transpose matrix (i.e. the faces id) while
    // d_faces will contain the offset that starts with zero and end with
    // num_faces*3 (zero is stored and the end can be inferred). Thus,
    // d_output should be allocated to size = num_faces*3

    block_mat_transpose<3u, blockThreads>(
        num_faces, num_edges, d_faces, d_output, shift);
}

template <uint32_t blockThreads>
__device__ __forceinline__ void f_f(const uint32_t num_edges,
                                    const uint32_t num_faces,
                                    uint16_t*      s_FE,
                                    uint16_t*      s_FF_offset,
                                    uint16_t*      s_FF_output)
{
    // First construct M_EF in shared memory

    uint16_t* s_EF_offset = &s_FE[num_faces * 3];
    uint16_t* s_EF_output = &s_EF_offset[num_edges + 1];

    // copy FE in to EF_offset so we can do the transpose in place without
    // losing FE
    for (uint16_t i = threadIdx.x; i < num_faces * 3; i += blockThreads) {
        flag_t   dir(0);
        uint16_t e     = s_FE[i] >> 1;
        s_EF_offset[i] = e;
        s_FE[i]        = e;
    }
    __syncthreads();

    e_f<blockThreads>(num_edges, num_faces, s_EF_offset, s_EF_output, 0);
    __syncthreads();

    // Every thread (T) is responsible for a face (F)
    // Each thread reads the edges (E) incident to its face (F). For each edge
    // (E), we read the "number" of incident faces (FF) to this edge (num_EF).
    // The number neighbor edges to the face F due to edge E is num_EF -1

    // TODO we can store this sum of neighbor faces in registers and then do
    // the exclusive sum on it and finally store it in shared memory
    for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        uint16_t num_neighbour_faces = 0;
        for (uint16_t e = 0; e < 3; ++e) {
            uint16_t edge = s_FE[3 * f + e];
            // printf("\n t= %u f= %u, e= %u, b0= %u, b1= %u ", threadIdx.x, f,
            //       edge, s_EF_offset[edge], s_EF_offset[edge + 1]);

            assert(s_EF_offset[edge + 1] >= s_EF_offset[edge]);

            num_neighbour_faces +=
                s_EF_offset[edge + 1] - s_EF_offset[edge] - 1;
        }
        s_FF_offset[f] = num_neighbour_faces;
    }
    __syncthreads();

    cub_block_exclusive_sum<uint16_t, blockThreads>(s_FF_offset, num_faces);

    for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        uint16_t offset = s_FF_offset[f];
        for (uint16_t e = 0; e < 3; ++e) {
            uint16_t edge = s_FE[3 * f + e];
            for (uint16_t ef = s_EF_offset[edge]; ef < s_EF_offset[edge + 1];
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

    /*{
        if (threadIdx.x == 0) {
            printf("\n s_EF_output");
            for (uint16_t f = 0; f < num_faces; ++f) {
                printf("\n face = %u>>", f);
                for (uint16_t ff = s_FF_offset[f]; ff < s_FF_offset[f + 1];
                     ++ff) {
                    printf(" %u ", s_FF_output[ff]);
                }
            }
        }
    }*/
}

template <uint32_t blockThreads, Op op>
__device__ __forceinline__ void query(uint16_t*&     s_offset_all_patches,
                                      uint16_t*&     s_output_all_patches,
                                      uint16_t*      s_patch_edges,
                                      uint16_t*      s_patch_faces,
                                      const uint32_t num_vertices,
                                      const uint32_t num_edges,
                                      const uint32_t num_faces)
{


    switch (op) {
        case Op::VV: {
            assert(num_vertices <= 2 * num_edges);
            s_offset_all_patches = &s_patch_edges[0];
            s_output_all_patches = &s_patch_edges[num_vertices + 1];
            v_v<blockThreads>(
                num_vertices, num_edges, s_patch_edges, s_output_all_patches);
            break;
        }
        case Op::VE: {
            assert(num_vertices <= 2 * num_edges);
            s_offset_all_patches = &s_patch_edges[0];
            s_output_all_patches = &s_patch_edges[num_vertices + 1];
            v_e<blockThreads>(
                num_vertices, num_edges, s_patch_edges, s_output_all_patches);
            break;
        }
        case Op::VF: {
            assert(num_vertices <= 2 * num_edges);
            s_output_all_patches = &s_patch_edges[0];
            s_offset_all_patches = &s_patch_faces[0];
            v_f<blockThreads>(num_faces,
                              num_edges,
                              num_vertices,
                              s_patch_edges,
                              s_patch_faces);
            break;
        }
        case Op::EV: {
            s_output_all_patches = s_patch_edges;
            break;
        }
        case Op::EF: {
            assert(num_edges <= 3 * num_faces);
            s_offset_all_patches = &s_patch_faces[0];
            s_output_all_patches = &s_patch_faces[num_edges + 1];
            e_f<blockThreads>(
                num_edges, num_faces, s_patch_faces, s_output_all_patches);
            break;
        }
        case Op::FV: {
            s_output_all_patches = s_patch_faces;
            f_v(num_edges, s_patch_edges, num_faces, s_patch_faces);
            break;
        }
        case Op::FE: {
            s_output_all_patches = s_patch_faces;
            break;
        }
        case Op::FF: {
            assert(num_edges <= 3 * num_faces);
            s_offset_all_patches =
                &s_patch_faces[3 * num_faces + 2 * 3 * num_faces];
            //                    ^^^^FE             ^^^^^EF
            s_output_all_patches = &s_offset_all_patches[num_faces + 1];
            f_f<blockThreads>(num_edges,
                              num_faces,
                              s_patch_faces,
                              s_offset_all_patches,
                              s_output_all_patches);

            break;
        }
        default:
            assert(1 != 1);
            break;
    }  // namespace RXMESH
}

}  // namespace rxmesh
