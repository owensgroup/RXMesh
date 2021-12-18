#pragma once
#include <stdint.h>
#include "rxmesh/types.h"

namespace rxmesh {

template <typename HandleT>
struct Iterator
{
    using LocalT = typename HandleT::LocalT;

    __device__ Iterator(const uint16_t  local_id,
                        const LocalT*   patch_output,
                        const uint16_t* patch_offset,
                        const uint32_t  offset_size,
                        const uint32_t  patch_id,
                        int             shift = 0)
        : m_patch_output(patch_output),
          m_patch_offset(patch_offset),
          m_patch_id(patch_id),
          m_shift(shift)
    {
        set(local_id, offset_size);
    }

    Iterator(const Iterator& orig) = default;


    __device__ uint16_t size() const
    {
        return m_end - m_begin;
    }

    __device__ HandleT operator[](const uint16_t i) const
    {
        assert(m_patch_output);
        assert(i + m_begin < m_end);
        return {m_patch_id, ((m_patch_output[m_begin + i].id) >> m_shift)};
    }

    __device__ HandleT operator*() const
    {
        assert(m_patch_output);        
        return ((*this)[m_current]);
    }

    __device__ HandleT back() const
    {
        return ((*this)[size() - 1]);
    }

    __device__ HandleT front() const
    {
        return ((*this)[0]);
    }

    __device__ Iterator& operator++()
    {
        // pre
        m_current = (m_current + 1) % size();
        return *this;
    }
    __device__ const Iterator operator++(int)
    {
        // post
        Iterator pre(*this);
        m_current = (m_current + 1) % size();
        return pre;
    }

    __device__ Iterator& operator--()
    {
        // pre
        m_current = (m_current == 0) ? size() - 1 : m_current - 1;
        return *this;
    }

    __device__ const Iterator operator--(int)
    {
        // post
        Iterator pre(*this);
        m_current = (m_current == 0) ? size() - 1 : m_current - 1;
        return pre;
    }

    __device__ bool operator==(const Iterator& rhs) const
    {
        return rhs.m_local_id == m_local_id && rhs.m_current == m_current;
    }

    __device__ bool operator!=(const Iterator& rhs) const
    {
        return !(*this == rhs);
    }


   private:
    const LocalT*   m_patch_output;
    const uint16_t* m_patch_offset;
    const uint32_t  m_patch_id;
    uint16_t        m_local_id;
    uint16_t        m_begin;
    uint16_t        m_end;
    uint16_t        m_current;
    int             m_shift;

    __device__ void set(const uint16_t local_id, const uint32_t offset_size)
    {
        m_current  = 0;
        m_local_id = local_id;
        if (offset_size == 0) {
            m_begin = m_patch_offset[m_local_id];
            m_end   = m_patch_offset[m_local_id + 1];
        } else {
            m_begin = m_local_id * offset_size;
            m_end   = (m_local_id + 1) * offset_size;
        }
        assert(m_end > m_begin);
    }
};

using VertexIterator = Iterator<VertexHandle>;
using EdgeIterator   = Iterator<EdgeHandle>;
using FaceIterator   = Iterator<FaceHandle>;

}  // namespace rxmesh