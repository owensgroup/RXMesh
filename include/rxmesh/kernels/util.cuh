#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace rxmesh {

template <typename attrT>
__global__ void memcopy(attrT*         d_dest,
                        const attrT*   d_src,
                        const uint32_t length)
{
    const uint32_t stride = blockDim.x * gridDim.x;
    uint32_t       i      = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length) {
        d_dest[i] = d_src[i];
        i += stride;
    }
}


template <typename attrT>
__global__ void memset(attrT* d_dest, const attrT val, const uint32_t length)
{
    const uint32_t stride = blockDim.x * gridDim.x;
    uint32_t       i      = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length) {
        d_dest[i] = val;
        i += stride;
    }
}

__device__ __forceinline__ float atomicMin(float* address, float val)
{
    // from
    // https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
    int ret = __float_as_int(*address);
    while (val < __int_as_float(ret)) {
        int old = ret;
        if ((ret = ::atomicCAS((int*)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ float atomicMax(float* address, float val)
{
    // from
    // https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret)) {
        int old = ret;
        if ((ret = ::atomicCAS((int*)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ uint16_t atomicAdd(uint16_t* address, uint16_t val)
{
    // Taken from
    // https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh#L36
    size_t    offset        = (size_t)address & 2;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    bool      is_32_align   = offset;
    uint32_t  old           = *address_as_ui;
    uint32_t  old_bytes;
    uint32_t  newval;
    uint32_t  assumed;

    do {
        assumed   = old;
        old_bytes = is_32_align ? old >> 16 : old & 0xffff;
        // preserve size in initial cast. Casting directly to uint32_t pads
        // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
        newval = static_cast<uint16_t>(val + old_bytes);
        newval = is_32_align ? (old & 0xffff) | (newval << 16) :
                               (old & 0xffff0000) | newval;
        old    = ::atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
    return (is_32_align) ? uint16_t(old >> 16) : uint16_t(old & 0xffff);
}


__device__ __forceinline__ uint16_t atomicMin(uint16_t* address, uint16_t val)
{
    // take from
    // https://github.com/pytorch/pytorch/blob/8b29b7953a46fbab9363294214f7689d04df0a85/aten/src/ATen/cuda/Atomic.cuh#L104
    size_t    offset        = (size_t)address & 2;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    bool      is_32_align   = offset;
    uint32_t  old           = *address_as_ui;
    uint32_t  old_bytes;
    uint32_t  newval;
    uint32_t  assumed;

    do {
        assumed   = old;
        old_bytes = is_32_align ? old >> 16 : old & 0xffff;
        newval    = std::min(val, static_cast<uint16_t>(old_bytes));
        newval    = is_32_align ? (old & 0xffff) | (newval << 16) :
                                  (old & 0xffff0000) | newval;
        old       = ::atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
    return is_32_align ? (old >> 16) : (old & 0xffff);
}

__device__ __forceinline__ uint8_t atomicAdd(uint8_t* address, uint8_t val)
{
    // Taken from
    // https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh#L14
    size_t    offset        = (size_t)address & 3;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    uint32_t  old           = *address_as_ui;
    uint32_t  shift         = offset * 8;
    uint32_t  old_byte;
    uint32_t  newval;
    uint32_t  assumed;

    do {
        assumed  = old;
        old_byte = (old >> shift) & 0xff;
        // preserve size in initial cast. Casting directly to uint32_t pads
        // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
        newval = static_cast<uint8_t>(val + old_byte);
        newval = (old & ~(0x000000ff << shift)) | (newval << shift);
        old    = ::atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);

    return uint8_t((old >> shift) & 0xff);
}

namespace detail {
template <typename T_output, typename T_input>
__forceinline__ __device__ T_output type_reinterpret(T_input value)
{
    static_assert(sizeof(T_output) == sizeof(T_input),
                  "type_reinterpret for different size");
    return *(reinterpret_cast<T_output*>(&value));
}
}  // namespace detail

/**
 * atomicCAS() on unsigned short int for SM <7.0
 */
__device__ __forceinline__ unsigned short int atomicCAS(
    unsigned short int* address,
    unsigned short int  compare,
    unsigned short int  val)
{
#if __CUDA_ARCH__ >= 700
    return ::atomicCAS(address, compare, val);
#else
    // Taken from
    // https://github.com/rapidsai/cudf/blob/89b802e6cecffe2425048f1f70cd682b865730b8/cpp/include/cudf/detail/utilities/device_atomics.cuh
    using T_int       = unsigned int;
    using T_int_short = unsigned short int;

    bool   is_32_align = (reinterpret_cast<size_t>(address) & 2) ? false : true;
    T_int* address_uint32 = reinterpret_cast<T_int*>(
        reinterpret_cast<size_t>(address) - (is_32_align ? 0 : 2));

    T_int       old = *address_uint32;
    T_int       assumed;
    T_int_short target_value;
    uint16_t    u_val = detail::type_reinterpret<uint16_t, T_int_short>(val);

    do {
        assumed = old;
        target_value =
            (is_32_align) ? T_int_short(old & 0xffff) : T_int_short(old >> 16);
        if (target_value != compare)
            break;

        T_int new_value = (is_32_align) ? (old & 0xffff0000) | u_val :
                                          (old & 0xffff) | (T_int(u_val) << 16);
        old             = ::atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return target_value;

#endif
}


__device__ __forceinline__ unsigned dynamic_smem_size()
{
    unsigned ret;
    asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}


/**
 * @brief read from global memory bypassing caches. It could/should be
 * implemented using
 * asm volatile ("ld.global.cg.s32 %0, [%1];\n" : "=r"(output) : "l"(ptr));
 * but this will requires specialization for different types. The code below
 * performs just the same (in terms of number of registers, clock cycles) and
 * does not need specialization for different types
 */
template <typename T>
__device__ __forceinline__ T atomic_read(T* ptr)
{
    __threadfence();
    return ::atomicAdd(ptr, T(0));
}


template <uint32_t blockThreads, typename T, typename SizeT>
__device__ __forceinline__ void fill_n(T* arr, const SizeT size, const T val)
{
    for (SizeT i = threadIdx.x; i < size; i += blockThreads) {
        arr[i] = val;
    }
}

template <typename T>
__device__ __forceinline__ void swap(T*& a, T*& b)
{
    T* temp = a;
    a       = b;
    b       = temp;
}

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret;
    asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}


template <typename FuncT>
__global__ void for_each_item(uint32_t length, FuncT func)
{
    const uint32_t stride = blockDim.x * gridDim.x;
    uint32_t       i      = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length) {
        func(i);
        i += stride;
    }
}

}  // namespace rxmesh