#pragma once

#include <thrust/unique.h>
#include <cub/block/block_radix_sort.cuh>

#include "filtering_util.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/kernels/query_dispatcher.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

constexpr float EPS = 10e-6;


/**
 * compute_vertex_normal()
 */
template <typename T, uint32_t blockThreads>
__global__ static void compute_vertex_normal(const rxmesh::Context      context,
                                             rxmesh::VertexAttribute<T> coords,
                                             rxmesh::VertexAttribute<T> normals)
{
    using namespace rxmesh;
    auto vn_lambda = [&](FaceHandle face_id, VertexIterator& fv) {
        // this face's three vertices
        VertexHandle v0(fv[0]), v1(fv[1]), v2(fv[2]);

        // get the face's three vertices coordinates
        Vector<3, T> c0(coords(fv[0], 0), coords(fv[0], 1), coords(fv[0], 2));
        Vector<3, T> c1(coords(fv[1], 0), coords(fv[1], 1), coords(fv[1], 2));
        Vector<3, T> c2(coords(fv[2], 0), coords(fv[2], 1), coords(fv[2], 2));

        // compute the face normal
        Vector<3, T> n = cross(c1 - c0, c2 - c0);
        n.normalize();

        // add the face's normal to its vertices
        for (uint32_t v = 0; v < 3; ++v) {      // for every vertex in this face
            for (uint32_t i = 0; i < 3; ++i) {  // for the vertex 3 coordinates
                atomicAdd(&normals(fv[v], i), n[i]);
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    query.dispatch<Op::FV>(
        block, vn_lambda, [](FaceHandle) { return true; }, false);
}


/**
 * compute_new_coordinates()
 */
template <typename T>
__device__ __inline__ void compute_new_coordinates(
    const rxmesh::VertexHandle&       v_id,
    const rxmesh::VertexHandle        vv[],
    const uint8_t                     num_vv,
    rxmesh::Vector<3, T>&             v,
    const rxmesh::Vector<3, T>&       n,
    const T                           sigma_c_sq,
    const rxmesh::VertexAttribute<T>& input_coords,
    rxmesh::VertexAttribute<T>&       filtered_coords)
{
    T sigma_s_sq = compute_sigma_s_sq(v_id, vv, num_vv, v, n, input_coords);

    T sum        = 0;
    T normalizer = 0;
    for (uint8_t i = 0; i < num_vv; ++i) {
        rxmesh::Vector<3, T> q(input_coords(vv[i], 0),
                               input_coords(vv[i], 1),
                               input_coords(vv[i], 2));
        q -= v;
        T t  = q.norm();
        T h  = dot(q, n);
        T wc = exp(-0.5 * t * t / sigma_c_sq);
        T ws = exp(-0.5 * h * h / sigma_s_sq);

        sum += wc * ws * h;
        normalizer += wc * ws;
    }
    v += (n * (sum / normalizer));

    filtered_coords(v_id, 0) = v[0];
    filtered_coords(v_id, 1) = v[1];
    filtered_coords(v_id, 2) = v[2];
}

/**
 * bilateral_filtering_low_level_API()
 * TODO refactor this to use handles
 */
/*template <typename T, uint32_t blockThreads, uint32_t maxVVSize>
__launch_bounds__(blockThreads) __global__
    static void bilateral_filtering_low_level_API(
        const rxmesh::Context context,
        rxmesh::Attribute<T>  input_coords,
        rxmesh::Attribute<T>  filtered_coords,
        rxmesh::Attribute<T>  vertex_normals)
{
    constexpr uint32_t special = 0xFFFFFFFE;

    using namespace rxmesh;
    uint32_t vv[maxVVSize];
    uint32_t vv_patch[maxVVSize];
    uint16_t vv_local[maxVVSize];

    uint8_t      num_vv     = 0;
    T            sigma_c_sq = 0;
    T            radius     = 0;
    Vector<3, T> vertex, normal;
    uint32_t     v_id = INVALID32;

    __shared__ uint32_t s_num_patches;
    __shared__ uint32_t s_block_patches[blockThreads];
    s_block_patches[threadIdx.x] = INVALID32;
    __shared__ uint32_t s_current_num_patches;

    if (threadIdx.x == 0) {
        s_current_num_patches = 0;
        s_num_patches         = 0;
    }

    uint32_t patch_id = blockIdx.x;
    // This lambda function gets the 1-ring, compute the search radius, and then
    // keeps processing the 1-ring of 1-ring as long as the vertices being
    // processed are within the same patch (patch_id). If a vertex within the
    // k-ring is not in the patch, it will be added to s_block_patches so the
    // whole block would process this patch later.
    auto compute_vv_1st_level = [&](uint32_t p_id, Iterator& iter) {
        v_id      = p_id;
        vertex[0] = input_coords(v_id, 0);
        vertex[1] = input_coords(v_id, 1);
        vertex[2] = input_coords(v_id, 2);

        normal[0] = vertex_normals(v_id, 0);
        normal[1] = vertex_normals(v_id, 1);
        normal[2] = vertex_normals(v_id, 2);

        normal.normalize();

        vv[0]       = v_id;
        vv_patch[0] = INVALID32;
        ++num_vv;

        sigma_c_sq = 1e10;

        for (uint32_t v = 0; v < iter.size(); ++v) {
            const uint32_t     vv_id = iter[v];
            const Vector<3, T> q(input_coords(vv_id, 0),
                                 input_coords(vv_id, 1),
                                 input_coords(vv_id, 2));

            T len = dist2(vertex, q);
            if (len < sigma_c_sq) {
                sigma_c_sq = len;
            }
        }

        radius = 4.0 * sigma_c_sq;


        // add 1-ring if it is within the radius
        for (uint32_t v = 0; v < iter.size(); ++v) {
            uint32_t vv_id = iter[v];

            const Vector<3, T> vvc(input_coords(vv_id, 0),
                                   input_coords(vv_id, 1),
                                   input_coords(vv_id, 2));

            T dist = dist2(vertex, vvc);

            if (dist <= radius) {
                uint8_t id = num_vv++;
                assert(id < maxVVSize);
                vv[id]       = vv_id;
                vv_local[id] = iter.neighbour_local_id(v);
                vv_patch[id] = special;
            }
        }


        // process the 1-ring vertices that this in this patch and within
        // the radius
        uint8_t num_vv_start = 1;
        uint8_t num_vv_end   = num_vv;

        while (true) {

            for (uint16_t v = num_vv_start; v < num_vv_end; ++v) {

                // This condition means that this vertex is owned by this
                // patch, and thus we can process it now since we have its
                // results
                if (vv_local[v] < iter.m_num_src_in_patch) {

                    assert(vv_patch[v] == special);
                    assert(context.get_vertex_patch()[vv[v]] == patch_id);

                    // to indicate that it's processed
                    vv_patch[v] = INVALID32;

                    Iterator vv_iter(iter);
                    vv_iter.set(vv_local[v], 0);

                    for (uint32_t i = 0; i < vv_iter.size(); ++i) {
                        uint32_t vvv_id       = vv_iter[i];
                        uint16_t vvv_local_id = vv_iter.neighbour_local_id(i);

                        // make sure that it is not a duplicate
                        if (!linear_search(vv, vvv_id, num_vv)) {
                            const Vector<3, T> vvv(input_coords(vvv_id, 0),
                                                   input_coords(vvv_id, 1),
                                                   input_coords(vvv_id, 2));


                            T dist = dist2(vvv, vertex);
                            if (dist <= radius) {
                                uint8_t id = num_vv++;

                                assert(id < maxVVSize);
                                vv[id]       = vvv_id;
                                vv_local[id] = vvv_local_id;
                                vv_patch[id] = special;
                            }
                        }
                    }
                } else {
                    // if the vertex is not owned by this patch, we add its
                    // patch so we can process it later.

                    uint32_t pp = context.get_vertex_patch()[vv[v]];

                    // but we first check if this thread has added this
                    // patch before (this will reduce the duplicates
                    // significantly)

                    if (!linear_search(vv_patch, pp, num_vv)) {
                        uint32_t id = atomicAdd(&s_num_patches, 1u);
                        assert(id < blockThreads);
                        s_block_patches[id] = pp;
                    }
                    vv_patch[v] = pp;
                    vv_local[v] = INVALID16;
                }
            }

            // means we have not added anything new
            if (num_vv_end == num_vv) {
                break;
            }

            // otherwise, it means we have added new vertices that might
            // fall in this patch, so we better process them now.
            num_vv_start = num_vv_end;
            num_vv_end   = num_vv;
        }
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, compute_vv_1st_level);
    __syncthreads();


    while (s_num_patches > 0) {
        __syncthreads();

        // Filter out duplicate patches

        // sort
        typedef cub::BlockRadixSort<uint32_t, blockThreads, 1> BlockRadixSort;
        __shared__ typename BlockRadixSort::TempStorage        temp_storage;
        uint32_t                                               thread_key[1];
        thread_key[0] = s_block_patches[threadIdx.x];
        if (threadIdx.x < s_current_num_patches ||
            threadIdx.x >= s_num_patches) {
            thread_key[0] = INVALID32;
        }
        BlockRadixSort(temp_storage).Sort(thread_key);
        s_block_patches[threadIdx.x] = thread_key[0];
        __syncthreads();

        // uniquify
        uint32_t  num_current_patches = s_num_patches - s_current_num_patches;
        uint32_t* new_end =
            thrust::unique(thrust::device,
                           s_block_patches,
                           s_block_patches + num_current_patches);
        __syncthreads();

        if (threadIdx.x == 0) {
            s_current_num_patches = new_end - s_block_patches;
            s_num_patches         = s_current_num_patches;
        }
        __syncthreads();


        for (uint32_t p = 0; p < s_current_num_patches; ++p) {

            patch_id = s_block_patches[p];

            uint32_t  num_src_in_patch, *input_mapping, *output_mapping;
            uint16_t *offset_all_patches, *output_all_patches;

            detail::template query_block_dispatcher<Op::VV, blockThreads>(
                context,
                patch_id,
                [](uint32_t) { return true; },
                false,
                true,
                num_src_in_patch,
                input_mapping,
                output_mapping,
                offset_all_patches,
                output_all_patches);


            // mean that this thread has be assigned a vertex in
            // compute_vv_1st_level
            if (v_id != INVALID32) {

                // search within this thread list (vv) to see if any
                // unprocessed vertex falls in this patch
                for (uint16_t v = 1; v < num_vv; ++v) {
                    if (vv_patch[v] == patch_id) {

                        // the global index of this vertex
                        uint32_t vv_id = vv[v];

                        // search for its local index
                        uint16_t vv_local_id = vv_local[v];

                        if (vv_local_id == INVALID16) {
                            for (uint16_t j = 0; j < num_src_in_patch; ++j) {
                                if (vv_id == output_mapping[j]) {
                                    vv_local_id = j;
                                    break;
                                }
                            }
                        }
                        assert(vv_local_id != INVALID16);

                        // so that we don't process it again
                        vv_patch[v] = INVALID32;

                        Iterator vv_iter(vv_local_id,
                                         output_all_patches,
                                         offset_all_patches,
                                         output_mapping,
                                         0,
                                         num_src_in_patch);

                        for (uint32_t i = 0; i < vv_iter.size(); ++i) {
                            uint32_t vvv_id = vv_iter[i];
                            uint32_t vvv_local_id =
                                vv_iter.neighbour_local_id(i);


                            // make sure that it is not a duplicate
                            if (!linear_search(vv, vvv_id, num_vv)) {

                                const Vector<3, T> vvv(input_coords(vvv_id, 0),
                                                       input_coords(vvv_id, 1),
                                                       input_coords(vvv_id, 2));

                                T dist = dist2(vvv, vertex);

                                if (dist <= radius) {

                                    uint8_t id = num_vv++;
                                    assert(id < maxVVSize);
                                    vv[id] = vvv_id;

                                    uint32_t pp;
                                    if (vvv_local_id < num_src_in_patch) {
                                        pp = patch_id;
                                    } else {
                                        pp = context.get_vertex_patch()[vvv_id];
                                    }

                                    // search if this thread has added this
                                    // patch before so we reduce the
                                    // duplicates
                                    if (pp != patch_id) {
                                        if (!linear_search(
                                                vv_patch, pp, num_vv)) {
                                            uint32_t d =
                                                atomicAdd(&s_num_patches, 1u);
                                            assert(d < blockThreads);
                                            s_block_patches[d] = pp;
                                        }
                                        vv_local[id] = INVALID16;
                                    } else {
                                        vv_local[id] = vvv_local_id;
                                    }

                                    vv_patch[id] = pp;
                                }
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }


        __syncthreads();
        if (s_current_num_patches == s_num_patches) {
            break;
        }
    }


    if (v_id != INVALID32) {

        compute_new_coordinates(v_id,
                                vv,
                                num_vv,
                                vertex,
                                normal,
                                sigma_c_sq,
                                input_coords,
                                filtered_coords);
    }
}*/

template <typename T, uint32_t blockThreads, uint32_t maxVVSize>
__global__ static void bilateral_filtering(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> input_coords,
    rxmesh::VertexAttribute<T> filtered_coords,
    rxmesh::VertexAttribute<T> vertex_normals)
{
    using namespace rxmesh;
    VertexHandle vv[maxVVSize];

    uint32_t     num_vv     = 0;
    T            sigma_c_sq = 0;
    T            radius     = 0;
    Vector<3, T> vertex, normal;
    VertexHandle v_id;

    auto first_ring = [&](VertexHandle& p_id, VertexIterator& iter) {
        v_id      = p_id;
        vertex[0] = input_coords(v_id, 0);
        vertex[1] = input_coords(v_id, 1);
        vertex[2] = input_coords(v_id, 2);

        normal[0] = vertex_normals(v_id, 0);
        normal[1] = vertex_normals(v_id, 1);
        normal[2] = vertex_normals(v_id, 2);

        normal.normalize();

        vv[0] = v_id;
        ++num_vv;

        sigma_c_sq = 1e10;

        for (uint32_t v = 0; v < iter.size(); ++v) {
            const VertexHandle vv_id = iter[v];
            const Vector<3, T> q(input_coords(vv_id, 0),
                                 input_coords(vv_id, 1),
                                 input_coords(vv_id, 2));

            T len = dist2(vertex, q);
            if (len < sigma_c_sq) {
                sigma_c_sq = len;
            }
        }

        radius = 4.0 * sigma_c_sq;

        // add 1-ring if it is within the radius
        for (uint32_t v = 0; v < iter.size(); ++v) {
            const VertexHandle vv_id = iter[v];

            const Vector<3, T> vvc(input_coords(vv_id, 0),
                                   input_coords(vv_id, 1),
                                   input_coords(vv_id, 2));

            T dist = dist2(vertex, vvc);

            if (dist <= radius) {
                uint8_t id = num_vv++;
                assert(id < maxVVSize);
                vv[id] = vv_id;
            }
        }
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, first_ring);
    __syncthreads();


    uint32_t next_id = 1;
    while (true) {

        VertexHandle next_vertex;
        if (v_id.is_valid() && next_id < num_vv) {
            next_vertex = vv[next_id];
        }
        auto n_rings = [&](const VertexHandle& id, const VertexIterator& iter) {
            assert(id == next_vertex);

            for (uint32_t i = 0; i < iter.size(); ++i) {
                VertexHandle vvv_id = iter[i];

                if (vvv_id != v_id) {
                    // make sure that we don't store duplicate outputs
                    if (!linear_search(vv, vvv_id, num_vv)) {
                        const Vector<3, T> vvv(input_coords(vvv_id, 0),
                                               input_coords(vvv_id, 1),
                                               input_coords(vvv_id, 2));


                        T dist = dist2(vvv, vertex);
                        if (dist <= radius) {
                            uint32_t id = num_vv++;
                            assert(id < maxVVSize);
                            vv[id] = vvv_id;
                        }
                    }
                }
            }
        };


        higher_query_block_dispatcher<Op::VV, blockThreads>(
            context, next_vertex, n_rings);

        bool is_done = (next_id >= num_vv) || !v_id.is_valid();
        if (__syncthreads_and(is_done)) {
            break;
        }
        next_id++;
    }

    if (v_id.is_valid()) {
        compute_new_coordinates(v_id,
                                vv,
                                num_vv,
                                vertex,
                                normal,
                                sigma_c_sq,
                                input_coords,
                                filtered_coords);
    }
}