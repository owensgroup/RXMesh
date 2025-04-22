#pragma once

#include "rxmesh/diff/diff_handle.h"

#include "rxmesh/iterator.cuh"
#include "rxmesh/types.h"

namespace rxmesh {

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


template <typename T, int VariableDim, typename DiffHandleT, typename PassiveT>
__device__ __inline__ Eigen::Vector<T, VariableDim> iter_val(
    const DiffHandleT&                                       handle,
    const Attribute<PassiveT, typename DiffHandleT::Handle>& attr)
{
    Eigen::Vector<T, VariableDim> ret;

    assert(VariableDim == attr.get_num_attributes());


    // val
    for (int j = 0; j < VariableDim; ++j) {
        if constexpr (DiffHandleT::IsActive) {
            ret[j].val = attr(handle, j);
        } else {
            ret[j] = attr(handle, j);
        }
    }

    // init grad
    if constexpr (DiffHandleT::IsActive) {
        for (int j = 0; j < VariableDim; ++j) {
            ret[j].grad[j] = 1;
        }
    }

    return ret;
}

}  // namespace rxmesh