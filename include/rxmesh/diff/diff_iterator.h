#pragma once

#include "rxmesh/diff/diff_handle.h"

#include "rxmesh/iterator.cuh"
#include "rxmesh/types.h"

namespace rxmesh {

/**
 * @brief Helper struct to get the iterator type based on a query operation
 */
template <Op op>
struct IteratorType
{
    using type = void;
};

template <>
struct IteratorType<Op::VV>
{
    using type = VertexIterator;
};

template <>
struct IteratorType<Op::VE>
{
    using type = EdgeIterator;
};

template <>
struct IteratorType<Op::VF>
{
    using type = FaceIterator;
};


template <>
struct IteratorType<Op::EV>
{
    using type = VertexIterator;
};

template <>
struct IteratorType<Op::EVDiamond>
{
    using type = VertexIterator;
};

template <>
struct IteratorType<Op::EE>
{
    using type = EdgeIterator;
};

template <>
struct IteratorType<Op::EF>
{
    using type = FaceIterator;
};


template <>
struct IteratorType<Op::FV>
{
    using type = VertexIterator;
};

template <>
struct IteratorType<Op::FE>
{
    using type = EdgeIterator;
};

template <>
struct IteratorType<Op::FF>
{
    using type = FaceIterator;
};


/**
 * @brief 
 * @tparam T 
 * @tparam DiffHandleT 
 * @tparam IteratorT 
 * @tparam PassiveT 
 * @tparam VariableDim 
 * @param handle 
 * @param iter 
 * @param attr 
 * @param index 
 * @return 
 */
template <typename T,
          int VariableDim,
          typename DiffHandleT,
          typename IteratorT,
          typename PassiveT>
__device__ __inline__ Eigen::Vector<T, VariableDim> iter_val(
    const DiffHandleT& handle,  // used to get the scalar type
    const IteratorT&   iter,
    const Attribute<PassiveT, typename IteratorT::Handle>& attr,
    int                                                    index)
{
    Eigen::Vector<T, VariableDim> ret;

    assert(index < iter.size());

    assert(VariableDim == attr.get_num_attributes());


    // val
    for (int j = 0; j < VariableDim; ++j) {
        if constexpr (DiffHandleT::IsActive) {
            ret[j].val = attr(iter[index], j);
        } else {
            ret[j] = attr(iter[index], j);
        }
    }

    // init grad
    if constexpr (DiffHandleT::IsActive) {
        for (int j = 0; j < VariableDim; ++j) {
            ret[j].grad[index * VariableDim + j] = 1;
        }
    }

    return ret;
}

}  // namespace rxmesh