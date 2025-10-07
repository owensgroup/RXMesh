#pragma once

#include "rxmesh/diff/util.h"
#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename VAttrT, typename FAttrT1, typename FAttrT2>
void init_volume_inverse_b(RXMeshStatic& rx,
                           const VAttrT& x,
                           FAttrT1&      volume,
                           FAttrT2&      inv_b)
{
    using T = typename VAttrT::Type;

    rx.run_query_kernel<Op::FV, 256>(
        [x, volume, inv_b] __device__(const FaceHandle      f,
                                      const VertexIterator& vv) mutable {
            Eigen::Vector3<T> x0 = x.template to_eigen<3>(vv[0]);
            Eigen::Vector3<T> x1 = x.template to_eigen<3>(vv[1]);
            Eigen::Vector3<T> x2 = x.template to_eigen<3>(vv[2]);

            Eigen::Vector3<T> e0 = x2 - x0;
            Eigen::Vector3<T> e1 = x1 - x0;

            Eigen::Vector3<T> n = (e0).cross(e1);
            n.normalize();

            Eigen::Matrix<T, 3, 3> tb = col_mat(e0, e1, n);

            volume(f) = T(0.5) * tb.col(0).cross(tb.col(1)).norm();


            // pseudo-inverse
            // inv_b(f) = (((tb.transpose()) * (tb)).inverse()) *
            // (tb.transpose());

            inv_b(f) = tb.inverse();

            
        });
}