#pragma once
#include <stdint.h>
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/patch_stash.cuh"
#include "rxmesh/util/bitmask_util.h"

namespace rxmesh {

template <typename HandleT>
struct Iterator
{
    using LocalT = typename HandleT::LocalT;

    __device__ Iterator()
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

    __device__ Iterator(const Context& context,
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

    __device__ Iterator(const Context&     context,
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


    __device__ uint16_t size() const
    {
        return m_end - m_begin;
    }

    __device__ HandleT operator[](const uint16_t i) const
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
        HandleT ret(m_patch_id, lid);

        if (detail::is_owned(lid, m_output_owned_bitmask)) {
            return ret;
        } else {
            return m_context.get_owner_handle(ret, nullptr, m_s_table);
        }
    }

    __device__ uint16_t local(const uint16_t i) const
    {
        if (i + m_begin >= m_end) {
            return INVALID16;
        }
        assert(m_patch_output);
        assert(i + m_begin < m_end);
        uint16_t lid = (m_patch_output[m_begin + i].id) >> m_shift;
        return lid;
    }

    __device__ HandleT back() const
    {
        return ((*this)[size() - 1]);
    }

    __device__ HandleT front() const
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

}  // namespace rxmesh