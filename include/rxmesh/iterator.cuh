#pragma once
#include <stdint.h>
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/patch_stash.cuh"
#include "rxmesh/types.h"
#include "rxmesh/util/bitmask_util.h"

namespace rxmesh {

template <typename HandleT>
struct Iterator
{
    using LocalT = typename HandleT::LocalT;
    using Handle = HandleT;

    __device__ __inline__ Iterator()
        : m_context(Context()),
          m_local_id(INVALID16),
          m_patch_output(nullptr),
          m_patch_id(INVALID32),
          m_output_owned_bitmask(nullptr),
          m_output_lp_hashtable(LPHashTable()),
          m_s_table(nullptr),
          m_patch_stash(PatchStash()),
          m_begin(0),
          m_end(0),
          m_current(0),
          m_shift(0)
    {
    }

    __device__ __inline__ Iterator(const Context& context,
                                   const uint16_t local_id,
                                   const uint32_t patch_id)
        : m_context(context),
          m_local_id(local_id),
          m_patch_output(nullptr),
          m_patch_id(patch_id),
          m_output_owned_bitmask(nullptr),
          m_output_lp_hashtable(LPHashTable()),
          m_s_table(nullptr),
          m_patch_stash(PatchStash()),
          m_begin(0),
          m_end(0),
          m_current(0),
          m_shift(0)
    {
    }

    __device__ __inline__ Iterator(const Context&     context,
                                   const uint16_t     local_id,
                                   const LocalT*      patch_output,
                                   const uint16_t*    patch_offset,
                                   const uint32_t     offset_size,
                                   const uint32_t     patch_id,
                                   const uint32_t*    output_owned_bitmask,
                                   const LPHashTable& output_lp_hashtable,
                                   const LPPair*      s_table,
                                   const PatchStash   patch_stash,
                                   int                shift = 0)
        : m_context(context),
          m_local_id(local_id),
          m_patch_output(patch_output),
          m_patch_id(patch_id),
          m_output_owned_bitmask(output_owned_bitmask),
          m_output_lp_hashtable(output_lp_hashtable),
          m_s_table(s_table),
          m_patch_stash(patch_stash),
          m_shift(shift)
    {
        set(local_id, offset_size, patch_offset);
    }

    Iterator(const Iterator& orig) = default;


    __device__ __inline__ uint16_t size() const
    {
        return m_end - m_begin;
    }

    __device__ __inline__ HandleT operator[](const uint16_t i) const
    {
        if (i + m_begin >= m_end) {
            return HandleT();
        }
        assert(m_patch_output);
        assert(i + m_begin < m_end);
        uint16_t lid = (m_patch_output[m_begin + i].id) >> m_shift;
        if (lid == INVALID16) {
            return HandleT();
        }

        if (detail::is_owned(lid, m_output_owned_bitmask)) {
            HandleT ret(m_patch_id, lid);
            return ret;
        } else {
            assert(m_s_table);
            LPPair lp = m_output_lp_hashtable.find(lid, m_s_table);
            if (lp.is_sentinel()) {
                return HandleT();
            }
            return HandleT(m_patch_stash.get_patch(lp),
                           {lp.local_id_in_owner_patch()});

            // return m_context.get_owner_handle(ret, nullptr, m_s_table);
        }
    }

    __device__ __inline__ uint16_t local(const uint16_t i) const
    {
        if (i + m_begin >= m_end) {
            return INVALID16;
        }
        assert(m_patch_output);
        assert(i + m_begin < m_end);
        uint16_t lid = (m_patch_output[m_begin + i].id) >> m_shift;
        return lid;
    }

    __device__ __inline__ HandleT back() const
    {
        return ((*this)[size() - 1]);
    }

    __device__ __inline__ HandleT front() const
    {
        return ((*this)[0]);
    }


   private:
    const Context&    m_context;
    uint16_t          m_local_id;
    const LocalT*     m_patch_output;
    const uint32_t    m_patch_id;
    const uint32_t*   m_output_owned_bitmask;
    const LPHashTable m_output_lp_hashtable;
    const LPPair*     m_s_table;
    const PatchStash  m_patch_stash;
    uint16_t          m_begin;
    uint16_t          m_end;
    uint16_t          m_current;
    int               m_shift;

    __device__ void set(const uint16_t  local_id,
                        const uint32_t  offset_size,
                        const uint16_t* patch_offset)
    {
        m_current = 0;
        if (offset_size == 0) {
            m_begin = patch_offset[m_local_id];
            m_end   = patch_offset[m_local_id + 1];
        } else {
            m_begin = m_local_id * offset_size;
            m_end   = (m_local_id + 1) * offset_size;
        }
        assert(m_end > m_begin);
    }
};

using VertexIterator = Iterator<VertexHandle>;
using EdgeIterator   = Iterator<EdgeHandle>;
using DEdgeIterator  = Iterator<DEdgeHandle>;
using FaceIterator   = Iterator<FaceHandle>;


/**
 * @brief Helper struct to get the iterator type based on a query operation
 */
template <Op op>
struct IteratorType
{
    using type = void;
};

template <>
struct IteratorType<Op::V>
{
    using type = VertexIterator;
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
struct IteratorType<Op::E>
{
    using type = VertexIterator;
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
struct IteratorType<Op::F>
{
    using type = VertexIterator;
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


}  // namespace rxmesh