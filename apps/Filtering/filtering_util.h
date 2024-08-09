#include "rxmesh/attribute.h"

/**
 * compute_sigma_c()
 */
template <typename T>
__device__ __inline__ T compute_sigma_c_sq(
    const rxmesh::VertexHandle        vv[],
    const uint8_t                     num_vv,
    const rxmesh::vec3<T>&            v,
    const rxmesh::VertexAttribute<T>& input_coords)
{

    T sigma_c = 1e10;
    for (uint8_t i = 1; i < num_vv; ++i) {
        const rxmesh::vec3<T> q(input_coords(vv[i], 0),
                                input_coords(vv[i], 1),
                                input_coords(vv[i], 2));

        T len = dist2(v, q);
        if (len < sigma_c) {
            sigma_c = len;
        }
    }
    return sigma_c;
}

/**
 * compute_sigma_s()
 */
template <typename T>
__device__ __inline__ T compute_sigma_s_sq(
    const rxmesh::VertexHandle&       v_id,
    const rxmesh::VertexHandle        vv[],
    const uint8_t                     num_vv,
    const rxmesh::vec3<T>&            v,
    const rxmesh::vec3<T>&            n,
    const rxmesh::VertexAttribute<T>& input_coords)
{

    T sum     = 0;
    T sum_sqs = 0;

    for (uint32_t i = 0; i < num_vv; ++i) {
        rxmesh::vec3<T> q(input_coords(vv[i], 0),
                          input_coords(vv[i], 1),
                          input_coords(vv[i], 2));

        q -= v;
        T t = dot(q, n);
        t   = sqrt(t * t);
        sum += t;
        sum_sqs += t * t;
    }
    T c       = static_cast<T>(num_vv);
    T sigma_s = (sum_sqs / c) - ((sum * sum) / (c * c));
    sigma_s   = (sigma_s < 1.0e-20) ? (sigma_s + 1.0e-20) : sigma_s;
    return sigma_s;
}

/**
 * linear_search() returns true if item exists in list
 */
template <typename T, typename S>
__device__ __forceinline__ bool linear_search(const T list[],
                                              const T item,
                                              const S end,
                                              const S start = 0)
{
    for (S i = start; i < end; ++i) {
        if (list[i] == item) {
            return true;
        }
    }
    return false;
}
