#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

/**
 * Discrete bending energy based on dihedral angles between adjacent triangles.
 * This follows the formulation from "Discrete quadratic curvature energies"
 * and similar papers on discrete shell energies.
 *
 * The energy is: E = k_b * sum_edges (area_edge / 3) * (theta - theta_rest)^2
 * where:
 *   - theta is the current dihedral angle
 *   - theta_rest is the rest dihedral angle
 *   - area_edge is the average area of the two adjacent triangles
 *   - k_b is the bending stiffness
 */
template <typename ProblemT, typename VAttrI, typename EAttrF, typename T>
void bending_energy(ProblemT&     problem,
                    const VAttrI& is_dbc,
                    const EAttrF& rest_angle,
                    const EAttrF& edge_area,
                    const T       bending_stiffness,
                    const T       h)  // time step
{
    const T h_sq = h * h;
    problem.template add_term<Op::EVDiamond, true>(
        [=] __device__(const auto& eh, const auto& iter, auto& obj) mutable {
            using ActiveT = ACTIVE_TYPE(eh);
            return ActiveT(0.0f);
        });

    // Bending energy operates on edges with their diamond (two adjacent faces)
    // problem.template add_term<Op::EVDiamond, true>(
    //     [=] __device__(const auto& eh, const auto& iter, auto& obj) mutable {
    //         using ActiveT = ACTIVE_TYPE(eh);

    //         if (iter.size() != 4) {
    //             // boundary edge, no bending energy
    //             return ActiveT(0.0f);
    //         }

    //         // Check if any vertex is Dirichlet BC
    //         if (is_dbc(iter[0]) || is_dbc(iter[1]) || is_dbc(iter[2]) ||
    //             is_dbc(iter[3])) {
    //             return ActiveT(0.0f);
    //         }

    //         // Get vertex positions
    //         // Edge goes from p to r. q and s are opposite to the edge
    //         Eigen::Vector3<ActiveT> p =
    //             iter_val<ActiveT, 3>(eh, iter, obj, 0);
    //         Eigen::Vector3<ActiveT> q =
    //             iter_val<ActiveT, 3>(eh, iter, obj, 1);
    //         Eigen::Vector3<ActiveT> r =
    //             iter_val<ActiveT, 3>(eh, iter, obj, 2);
    //         Eigen::Vector3<ActiveT> s =
    //             iter_val<ActiveT, 3>(eh, iter, obj, 3);

    //         // Compute edge vector
    //         Eigen::Vector3<ActiveT> e = r - p;
    //         ActiveT                 e_norm = e.norm();

    //         if (e_norm < ActiveT(1e-10)) {
    //             return ActiveT(0.0f);  // degenerate edge
    //         }

    //         // Compute normals of the two triangles
    //         Eigen::Vector3<ActiveT> n0 = (r - p).cross(q - p);
    //         ActiveT                 n0_norm = n0.norm();

    //         Eigen::Vector3<ActiveT> n1 = (s - p).cross(r - p);
    //         ActiveT                 n1_norm = n1.norm();

    //         if (n0_norm < ActiveT(1e-10) || n1_norm < ActiveT(1e-10)) {
    //             return ActiveT(0.0f);  // degenerate triangle
    //         }

    //         // Normalize the normals
    //         Eigen::Vector3<ActiveT> n0_normalized = n0 / n0_norm;
    //         Eigen::Vector3<ActiveT> n1_normalized = n1 / n1_norm;

    //         // Compute dihedral angle using atan2 for robustness
    //         ActiveT cos_theta = n0_normalized.dot(n1_normalized);

    //         // Clamp to avoid numerical issues with acos
    //         cos_theta = (cos_theta < ActiveT(-1.0)) ? ActiveT(-1.0) : cos_theta;
    //         cos_theta = (cos_theta > ActiveT(1.0)) ? ActiveT(1.0) : cos_theta;

    //         // Use cross product to get the sign
    //         Eigen::Vector3<ActiveT> n_cross = n0_normalized.cross(n1_normalized);
    //         ActiveT sign = (n_cross.dot(e) >= ActiveT(0)) ? ActiveT(1.0) : ActiveT(-1.0);

    //         ActiveT sin_theta_sq = ActiveT(1.0) - cos_theta * cos_theta;
    //         // Clamp to avoid sqrt of negative due to numerical errors
    //         sin_theta_sq = (sin_theta_sq < ActiveT(0)) ? ActiveT(0) : sin_theta_sq;
    //         ActiveT sin_theta = sign * sqrt(sin_theta_sq);
    //         ActiveT theta = atan2(sin_theta, cos_theta);

    //         // Get rest angle
    //         T theta_rest = rest_angle(eh);

    //         // Compute angle difference
    //         ActiveT d_theta = theta - theta_rest;

    //         // Bending energy: k_b * area * (theta - theta_rest)^2 * h^2
    //         // where area is the average area of the two triangles divided by 3
    //         // (distributed equally among the 3 edges)
    //         T area = edge_area(eh);

    //         // Skip if area is too small (degenerate configuration)
    //         if (area < T(1e-12)) {
    //             return ActiveT(0.0f);
    //         }

    //         ActiveT E = T(0.5) * bending_stiffness * area * d_theta * d_theta * h_sq;

    //         // Debug: check for misbehaving energy values
    //         if constexpr (std::is_same_v<ActiveT, T>) {
    //             // ActiveT is just T (float), no AD
    //             if (isnan(E) || isinf(E) || E < T(0)) {
    //                 printf("Bending energy misbehaves: E=%f, area=%f, d_theta=%f, theta=%f, theta_rest=%f\n",
    //                        E, area, d_theta, theta, theta_rest);
    //             }
    //         } else {
    //             // ActiveT is Scalar type with AD
    //             if (isnan(E.val()) || isinf(E.val()) || E.val() < T(0)) {
    //                 printf("Bending energy misbehaves: E=%f, area=%f, d_theta=%f, theta=%f, theta_rest=%f, n0_norm=%f, n1_norm=%f\n",
    //                        E.val(), area, d_theta.val(), theta.val(), theta_rest,
    //                        n0_norm.val(), n1_norm.val());
    //             }
    //         }

    //         return E;
    //     });
}

/**
 * Initialize rest angles and edge areas for bending energy
 */
template <typename VAttrT, typename EAttrF>
void init_bending(RXMeshStatic& rx,
                  const VAttrT& x,
                  EAttrF&       rest_angle,
                  EAttrF&       edge_area)
{
    using T = typename VAttrT::Type;

    constexpr uint32_t blockThreads = 256;

    rx.run_query_kernel<Op::EVDiamond, blockThreads>(
        [=] __device__(const EdgeHandle& eh, const VertexIterator& iter) mutable {
            if (iter.size() != 4) {
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

            // Compute edge vector
            Eigen::Vector3<T> e = r - p;
            T                 e_norm = e.norm();

            if (e_norm < T(1e-10)) {
                rest_angle(eh) = T(0);
                edge_area(eh)  = T(0);
                return;
            }

            // Compute normals of the two triangles
            Eigen::Vector3<T> n0 = (r - p).cross(q - p);
            T                 n0_norm = n0.norm();

            Eigen::Vector3<T> n1 = (s - p).cross(r - p);
            T                 n1_norm = n1.norm();

            if (n0_norm < T(1e-10) || n1_norm < T(1e-10)) {
                rest_angle(eh) = T(0);
                edge_area(eh)  = T(0);
                return;
            }

            n0 /= n0_norm;
            n1 /= n1_norm;

            // Compute dihedral angle
            T cos_theta = n0.dot(n1);
            cos_theta = (cos_theta < T(-1.0)) ? T(-1.0) : cos_theta;
            cos_theta = (cos_theta > T(1.0)) ? T(1.0) : cos_theta;

            Eigen::Vector3<T> n_cross = n0.cross(n1);
            T sign = (n_cross.dot(e) >= T(0)) ? T(1.0) : T(-1.0);

            T sin_theta = sign * sqrt(T(1.0) - cos_theta * cos_theta);
            T theta = atan2(sin_theta, cos_theta);

            rest_angle(eh) = theta;

            // Compute edge area: average area of two triangles / 3
            // Triangle areas
            T area0 = n0_norm * T(0.5);  // ||(x1-x0) x (x2-x0)|| / 2
            T area1 = n1_norm * T(0.5);  // ||(x3-x0) x (x1-x0)|| / 2

            // Average area divided by 3 (for the 3 edges)
            edge_area(eh) = (area0 + area1) / T(6.0);
        });
}