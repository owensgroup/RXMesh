#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/diff/scalar.h"

namespace rxmesh {

class RXMeshStatic;

template <typename T, int Size, bool WithHessian, typename HandleT>
class DiffAttribute : public Attribute<Scalar<T, Size, WithHessian>, HandleT>
{
   public:
    using PassiveType       = T;
    using ScalarType        = Scalar<T, Size, WithHessian>;
    using BaseAttributeType = Attribute<T, VertexHandle>;


    DiffAttribute() : Attribute<ScalarType, HandleT>()
    {
    }

    explicit DiffAttribute(const char*   name,
                           uint32_t      num_attributes,  // not used
                           locationT     location,
                           layoutT       layout,  // not used
                           RXMeshStatic* rxmesh)
        : Attribute<ScalarType, HandleT>(name,
                                         num_attributes,
                                         location,
                                         layout,
                                         rxmesh)
    {
    }


    /**
     * @brief return the value of the Scalar type as an attribute. The only
     * reason we do this is for visualization. Thus, the return attributes is
     * defined only on the host
     */
    std::shared_ptr<BaseAttributeType> to_passive()
    {
        auto ret = this->m_rxmesh->template add_vertex_attribute<T>(
            std::string("rx:") + std::string(this->m_name), 1, HOST);
        this->m_rxmesh->for_each_vertex(HOST, [&](VertexHandle vh) {
            (*ret)(vh) = this->operator()(vh).val;
        });

        return ret;
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