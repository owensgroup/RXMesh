#include "bending_energy.h"
#include "rxmesh/diff/diff_iterator.h"
#include "rxmesh/diff/diff_scalar_problem.h"

using namespace rxmesh;

template <typename T>
__device__ __host__ T compute_dihedral_angle(const Eigen::Vector3<T>& p,
                                              const Eigen::Vector3<T>& q,
                                              const Eigen::Vector3<T>& r,
                                              const Eigen::Vector3<T>& s,
                                              T& n0_norm,
                                              T& n1_norm)
{
    // Compute edge vector
    Eigen::Vector3<T> e = r - p;
    T e_norm = e.norm();

    if (e_norm < T(1e-7)) {
        n0_norm = T(0);
        n1_norm = T(0);
        return T(0);  // degenerate edge
    }

    // Compute normals of the two triangles
    Eigen::Vector3<T> n0 = (r - p).cross(q - p);
    n0_norm = n0.norm();

    Eigen::Vector3<T> n1 = (s - p).cross(r - p);
    n1_norm = n1.norm();

    if (n0_norm < T(1e-7) || n1_norm < T(1e-7)) {
        return T(0);  // degenerate triangle
    }

    // Normalize the normals
    Eigen::Vector3<T> n0_normalized = n0 / n0_norm;
    Eigen::Vector3<T> n1_normalized = n1 / n1_norm;

    // Compute dihedral angle using acos (simpler, unsigned angle)
    T cos_theta = n0_normalized.dot(n1_normalized);

    // Clamp to avoid numerical issues with acos
    cos_theta = (cos_theta <= T(-1.0)) ? T(-0.99999f) : cos_theta;
    cos_theta = (cos_theta >= T(1.0)) ? T(0.99999f) : cos_theta;

    // Use cross product to get the sign
    Eigen::Vector3<T> n_cross = n0_normalized.cross(n1_normalized);
    T sign = (n_cross.dot(e) >= T(0)) ? T(1.0) : T(-1.0);

    T sin_theta_sq = T(1.0) - cos_theta * cos_theta;
    // Clamp to avoid sqrt of negative due to numerical errors
    sin_theta_sq = (sin_theta_sq < T(0)) ? T(0) : sin_theta_sq;
    T sin_theta = sign * sqrt(sin_theta_sq);
    T theta = atan2(sin_theta, cos_theta);


    return theta;
}

// Device function for bending energy computation kernel
template <typename ActiveT, typename T, typename EAttrF>
__device__ ActiveT compute_bending_energy_kernel(const EdgeHandle&     eh,
                                                  const VertexIterator& iter,
                                                  auto&                 obj,
                                                  const EAttrF&         rest_angle,
                                                  const EAttrF&         edge_area,
                                                  const T               bending_stiffness,
                                                  const T               h_sq)
{
    // Check if all vertices are valid (not boundary edge)
    if (!iter[0].is_valid() || !iter[1].is_valid() ||
        !iter[2].is_valid() || !iter[3].is_valid()) {
        // boundary edge, no bending energy
        return ActiveT(0.0f);
    }

    // Get vertex positions
    // Edge goes from p to r. q and s are opposite to the edge
    Eigen::Vector3<ActiveT> p = iter_val<ActiveT, 3>(eh, iter, obj, 0);
    Eigen::Vector3<ActiveT> q = iter_val<ActiveT, 3>(eh, iter, obj, 1);
    Eigen::Vector3<ActiveT> r = iter_val<ActiveT, 3>(eh, iter, obj, 2);
    Eigen::Vector3<ActiveT> s = iter_val<ActiveT, 3>(eh, iter, obj, 3);

    // Compute dihedral angle using helper function
    ActiveT n0_norm, n1_norm;
    ActiveT theta = compute_dihedral_angle(p, q, r, s, n0_norm, n1_norm);

    // Check for degenerate configuration
    if (n0_norm < ActiveT(1e-7) || n1_norm < ActiveT(1e-7)) {
        return ActiveT(0.0f);
    }

    // Get rest angle
    T theta_rest = rest_angle(eh);

    // Compute angle difference
    ActiveT d_theta = theta - theta_rest;

    // Bending energy: k_b * area * (theta - theta_rest)^2 * h^2
    // where area is the average area of the two triangles divided by 3
    // (distributed equally among the 3 edges)
    T area = edge_area(eh);

    // Skip if area is too small (degenerate configuration)
    if (area < T(1e-12)) {
        return ActiveT(0.0f);
    }

    ActiveT E = T(0.5) * bending_stiffness * area * d_theta * d_theta * h_sq;

    // Debug: check for misbehaving energy values
    if constexpr (std::is_same_v<ActiveT, T>) {
        // ActiveT is just T (float), no AD
        if (isnan(E) || isinf(E) || E < T(0)) {
            printf("Bending energy misbehaves: E=%f, area=%f, d_theta=%f, theta=%f, theta_rest=%f\n",
                   E, area, d_theta, theta, theta_rest);
        }
    } else {
        // ActiveT is Scalar type with AD
        if (isnan(E.val()) || isinf(E.val()) || E.val() < T(0)) {
            printf("Bending energy misbehaves: E=%f, area=%f, d_theta=%f, theta=%f, theta_rest=%f, n0_norm=%f, n1_norm=%f\n",
                   E.val(), area, d_theta.val(), theta.val(), theta_rest,
                   n0_norm.val(), n1_norm.val());
        }
    }

    return E;
}

// Device function for bending initialization kernel
template <typename T, typename VAttrT, typename EAttrF>
__device__ void init_bending_kernel(const EdgeHandle&     eh,
                                     const VertexIterator& iter,
                                     const VAttrT&         x,
                                     EAttrF&               rest_angle,
                                     EAttrF&               edge_area)
{
    // Check if all vertices are valid (not boundary edge)
    if (!iter[0].is_valid() || !iter[1].is_valid() ||
        !iter[2].is_valid() || !iter[3].is_valid()) {
        // boundary edge
        rest_angle(eh) = T(0);
        edge_area(eh)  = T(0);
        return;
    }

    // Get vertex positions
    // Edge goes from p to r. q and s are opposite to the edge
    Eigen::Vector3<T> p = x.template to_eigen<3>(iter[0]);
    Eigen::Vector3<T> q = x.template to_eigen<3>(iter[1]);
    Eigen::Vector3<T> r = x.template to_eigen<3>(iter[2]);
    Eigen::Vector3<T> s = x.template to_eigen<3>(iter[3]);

    // Compute dihedral angle using helper function
    T n0_norm, n1_norm;
    T theta = compute_dihedral_angle(p, q, r, s, n0_norm, n1_norm);

    if (n0_norm < T(1e-10) || n1_norm < T(1e-10)) {
        rest_angle(eh) = T(0);
        edge_area(eh)  = T(0);
        return;
    }

    rest_angle(eh) = theta;

    // Compute edge area: average area of two triangles / 3
    // Triangle areas
    T area0 = n0_norm * T(0.5);  // ||(x1-x0) x (x2-x0)|| / 2
    T area1 = n1_norm * T(0.5);  // ||(x3-x0) x (x1-x0)|| / 2

    // Average area divided by 3 (for the 3 edges)
    edge_area(eh) = (area0 + area1) / T(6.0);
}
