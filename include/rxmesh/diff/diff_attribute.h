#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/diff/scalar.h"

namespace rxmesh {

class RXMeshStatic;

template <typename T, int Size, bool WithHessian, typename HandleT>
class DiffAttribute : public Attribute<Scalar<T, Size, WithHessian>, HandleT>
{
   public:
    using PassiveType = T;
    using ScalarType  = Scalar<T, Size, WithHessian>;


    DiffAttribute() : Attribute<ScalarType, HandleT>()
    {
    }

    explicit DiffAttribute(const char*         name,
                           uint32_t            num_attributes,  // not used
                           locationT           location,
                           layoutT             layout,  // not used
                           const RXMeshStatic* rxmesh)
        : Attribute<ScalarType, HandleT>(name,
                                         num_attributes,
                                         location,
                                         layout,
                                         rxmesh)
    {
    }

   private:
};

template <typename T, int Size, bool WithHessian>
using DiffVertexAttribute = DiffAttribute<T, Size, WithHessian, VertexHandle>;

template <typename T, int Size, bool WithHessian>
using DiffEdgeAttribute = DiffAttribute<T, Size, WithHessian, EdgeHandle>;

template <typename T, int Size, bool WithHessian>
using DiffFaceAttribute = DiffAttribute<T, Size, WithHessian, FaceHandle>;

}  // namespace rxmesh