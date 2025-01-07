#if 0
#pragma once

#include "rxmesh/diff/diff_handle.h"

namespace rxmesh {

template <typename DiffHandle, int StencilSize>
struct DiffIterator : public Iterator<typename DiffHandle::HandleT>
{
    using HandleT = typename DiffHandle::HandleT;
    using LocalT  = typename HandleT::LocalT;

    __device__ DiffIterator() : Iterator<HandleT>()
    {
    }

    DiffIterator(const Iterator<HandleT>& rhs) : Iterator<HandleT>(rhs)
    {
    }

    __device__ DiffIterator(const Context& context,
                            const uint16_t local_id,
                            const uint32_t patch_id)
        : Iterator<HandleT>(context, local_id, patch_id)
    {
    }

    __device__ DiffIterator(const Context&     context,
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
        : Iterator<HandleT>(context,
                            local_id,
                            patch_output,
                            patch_offset,
                            offset_size,
                            patch_id,
                            output_owned_bitmask,
                            output_lp_hashtable,
                            s_table,
                            patch_stash,
                            shift)
    {
    }

    DiffIterator(const DiffIterator&) = default;

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

    __device__ HandleT back() const
    {
        return ((*this)[size() - 1]);
    }

    __device__ HandleT front() const
    {
        return ((*this)[0]);
    }


    __device__ bool operator==(const Iterator& rhs) const
    {
        return rhs.m_local_id == m_local_id && rhs.m_patch_id == m_patch_id &&
               rhs.m_current == m_current;
    }

    __device__ bool operator!=(const Iterator& rhs) const
    {
        return !(*this == rhs);
    }
};

template <bool IsActive>
using DiffVertexIterator = DiffIterator<DiffVertexHandle<IsActive>>;

template <bool IsActive>
using DiffEdgeIterator = DiffIterator<DiffVertexHandle<IsActive>>;

template <bool IsActive>
using DiffFaceIterator = DiffIterator<DiffVertexHandle<IsActive>>;

}  // namespace rxmesh
#endif