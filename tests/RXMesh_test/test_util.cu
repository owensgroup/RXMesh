#include "gtest/gtest.h"

#include <algorithm>

#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/rxmesh_queries.cuh"
#include "rxmesh/kernels/shmem_allocator.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

template <uint32_t rowOffset, uint32_t blockThreads, uint32_t itemPerThread>
__global__ static void test_block_mat_transpose_kernel(uint16_t*      d_src,
                                                       const uint32_t num_rows,
                                                       const uint32_t num_cols,
                                                       uint16_t*      d_output,
                                                       uint32_t* d_row_bitmask)
{

    rxmesh::detail::block_mat_transpose<rowOffset, blockThreads, itemPerThread>(
        num_rows, num_cols, d_src, d_output, d_row_bitmask, 0);
}

template <uint32_t rowOffset, uint32_t blockThreads>
__global__ static void test_block_mat_transpose_kernel_shmem(
    uint16_t*      d_src,
    const uint32_t num_rows,
    const uint32_t num_cols,
    uint16_t*      d_output,
    uint32_t*      d_row_bitmask,
    uint32_t*      d_col_bitmask)
{
    using namespace rxmesh;

    ShmemAllocator shrd_alloc;

    uint16_t* s_temp_size  = shrd_alloc.alloc<uint16_t>(num_cols + 1);
    uint16_t* s_temp_local = shrd_alloc.alloc<uint16_t>(num_cols);

    rxmesh::detail::block_mat_transpose<rowOffset, blockThreads>(num_rows,
                                                                 num_cols,
                                                                 d_src,
                                                                 d_output,
                                                                 s_temp_size,
                                                                 s_temp_local,
                                                                 d_row_bitmask,
                                                                 d_col_bitmask,
                                                                 0);
}

template <typename T, uint32_t blockThreads>
__global__ static void test_block_exclusive_sum_kernel(T*             d_src,
                                                       const uint32_t size)
{
    rxmesh::detail::cub_block_exclusive_sum<T, blockThreads>(d_src, size);
}

template <typename T>
__global__ static void test_atomicMin_kernel(T* d_in, T* d_out)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    rxmesh::atomicMin(d_out, d_in[tid]);
}

template <typename T>
__global__ static void test_atomicAdd_kernel(T* d_val)
{
    rxmesh::atomicAdd(d_val, 1);
    /*__half* as_half = (__half*)(d_val);
    ::atomicAdd(as_half,1);
    __syncthreads();
    uint16_t val = uint16_t(as_half[0]);
    if(threadIdx.x == 0){
        d_val[0] = val;
    }*/
}

TEST(Util, Scan)
{
    using namespace rxmesh;

    constexpr uint32_t    blockThreads = 128;
    uint32_t              size         = 8144;
    std::vector<uint32_t> h_src(size, 1);
    uint32_t*             d_src = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_src, size * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(
        d_src, h_src.data(), size * sizeof(uint32_t), cudaMemcpyHostToDevice));

    test_block_exclusive_sum_kernel<uint32_t, blockThreads>
        <<<1, blockThreads>>>(d_src, size);

    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());

    CUDA_ERROR(cudaMemcpy(
        h_src.data(), d_src, size * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < h_src.size(); ++i) {
        EXPECT_EQ(h_src[i], i);
    }

    GPU_FREE(d_src);
}

template <typename T>
bool test_atomicAdd(const uint32_t threads = 1024)
{
    using namespace rxmesh;

    T  h_val = 0;
    T* d_val;

    CUDA_ERROR(cudaMalloc((void**)&d_val, sizeof(T)));
    CUDA_ERROR(cudaMemcpy(d_val, &h_val, sizeof(T), cudaMemcpyHostToDevice));


    test_atomicAdd_kernel<T><<<1, threads>>>(d_val);

    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());

    CUDA_ERROR(cudaMemcpy(&h_val, d_val, sizeof(T), cudaMemcpyDeviceToHost));


    // check
    bool passed = true;
    if (h_val != static_cast<T>(threads)) {
        passed = false;
    }
    GPU_FREE(d_val);

    return passed;
}

template <typename T>
bool test_atomicMin(const uint32_t threads = 1024)
{
    using namespace rxmesh;

    T* d_out;
    T* d_in;
    CUDA_ERROR(cudaMalloc((void**)&d_out, sizeof(T)));
    CUDA_ERROR(cudaMalloc((void**)&d_in, threads * sizeof(T)));
    if constexpr (sizeof(T) == 1) {
        CUDA_ERROR(cudaMemset(d_out, INVALID8, sizeof(T)));
    }
    if constexpr (sizeof(T) == 2) {
        CUDA_ERROR(cudaMemset(d_out, INVALID16, sizeof(T)));
    }
    if constexpr (sizeof(T) == 4) {
        CUDA_ERROR(cudaMemset(d_out, INVALID32, sizeof(T)));
    }
    if constexpr (sizeof(T) == 8) {
        CUDA_ERROR(cudaMemset(d_out, INVALID64, sizeof(T)));
    }

    std::vector<T> h_in(threads);
    fill_with_random_numbers(h_in.data(), h_in.size());
    std::transform(h_in.cbegin(), h_in.cend(), h_in.begin(), [](T c) {
        return 5 * (c + 1);
    });
    CUDA_ERROR(cudaMemcpy(
        d_in, h_in.data(), threads * sizeof(T), cudaMemcpyHostToDevice));


    test_atomicMin_kernel<<<1, threads>>>(d_in, d_out);

    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());

    T h_out;
    CUDA_ERROR(cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));

    std::sort(h_in.begin(), h_in.end());


    // check
    bool passed = h_in[0] == h_out;

    GPU_FREE(d_in);
    GPU_FREE(d_out);

    return passed;
}
TEST(Util, AtomicMin)
{
    EXPECT_TRUE(test_atomicMin<uint16_t>()) << "uint16_t failed";
}

TEST(Util, AtomicAdd)
{
    EXPECT_TRUE(test_atomicAdd<uint16_t>()) << "uint16_t failed";
    EXPECT_TRUE(test_atomicAdd<uint8_t>()) << "uint8_t failed";
}


TEST(Util, Align)
{
    using Type             = float;
    const size_t num_bytes = sizeof(Type) * 1024;
    const size_t alignment = 128;

    Type* ptr = (Type*)malloc(num_bytes);

    char* ptr_mis_aligned = reinterpret_cast<char*>(ptr) + 1;

    Type* ptr_aligned = reinterpret_cast<Type*>(ptr_mis_aligned);
    rxmesh::align(alignment, ptr_aligned);

    void*       ptr_aligned_gt = reinterpret_cast<void*>(ptr_mis_aligned);
    std::size_t spc            = num_bytes;
    void*       ret = std::align(alignment, sizeof(char), ptr_aligned_gt, spc);

    free(ptr);
    EXPECT_NE(ret, nullptr);
    EXPECT_EQ(ptr_aligned, ptr_aligned_gt);
}

TEST(Util, BlockMatrixTranspose)
{
    constexpr uint32_t numRows   = 542;
    constexpr uint32_t numCols   = 847;
    constexpr uint32_t rowOffset = 3;

    using namespace rxmesh;
    // The matrix is numRows X numCols where every rows has rowOffset
    // non-zero elements. The matrix passed to the kernel contains the
    // column ids only and we also pass the rowOffset. The transposed matrix
    // is stored in the source as row ids and the offset is stored in the
    // h_res_offset.

    const uint32_t        arr_size = numRows * rowOffset;
    std::vector<uint16_t> h_src(arr_size);
    std::vector<uint16_t> row(numCols);
    fill_with_sequential_numbers(row.data(), static_cast<uint32_t>(row.size()));
    random_shuffle(row.data(), static_cast<uint32_t>(row.size()));

    for (uint32_t s = 0; s < h_src.size(); s += rowOffset) {
        // prevent duplication in the same row
        for (uint32_t i = 0; i < rowOffset; ++i) {
            h_src[s + i] = row[i];
        }
        random_shuffle(row.data(), static_cast<uint32_t>(row.size()));
    }


    // const uint32_t threads = numRows*rowOffset;
    // We try to divide the number of non-zero elements equally between
    // threads. However, it may not aligned perfectly. So we need to pad
    // h_src with INVALID32 since this will be part of the sorting in
    // the transpose kernel. Also, d_offset should be large enough to
    // align with the padding.

    const uint32_t threads         = 256;
    const uint32_t item_per_thread = DIVIDE_UP(numRows * rowOffset, threads);
    const uint32_t blocks          = 1;


    if (item_per_thread * threads > numRows * rowOffset) {
        for (uint32_t i = numRows * rowOffset; i < item_per_thread * threads;
             ++i) {
            h_src.push_back(INVALID16);
        }
    }

    uint32_t* d_bitmask    = nullptr;
    uint32_t  bitmask_size = DIVIDE_UP(std::max(numRows, numCols), 32);
    CUDA_ERROR(cudaMalloc((void**)&d_bitmask, bitmask_size * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_bitmask, 0xFF, bitmask_size * sizeof(uint32_t)));

    uint16_t *d_src, *d_offset;
    CUDA_ERROR(cudaMalloc((void**)&d_src, h_src.size() * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_offset, h_src.size() * sizeof(uint16_t)));
    CUDA_ERROR(cudaMemcpy(d_src,
                          h_src.data(),
                          h_src.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));


    // test_block_mat_transpose_kernel<rowOffset, threads, item_per_thread>
    //     <<<blocks, threads, 0>>>(d_src, numRows, numCols, d_offset,
    //     d_bitmask);

    const size_t shmem = 2 * (numCols + 1) * sizeof(uint16_t) +
                         2 * ShmemAllocator::default_alignment;
    test_block_mat_transpose_kernel_shmem<rowOffset, threads>
        <<<blocks, threads, shmem>>>(
            d_src, numRows, numCols, d_offset, d_bitmask, d_bitmask);

    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());

    std::vector<uint16_t> h_res(arr_size);
    std::vector<uint16_t> h_res_offset(numCols);

    CUDA_ERROR(cudaMemcpy(h_res.data(),
                          d_offset,
                          arr_size * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));

    CUDA_ERROR(cudaMemcpy(h_res_offset.data(),
                          d_src,
                          numCols * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));

    std::vector<uint16_t> gold_res(arr_size);
    std::vector<uint16_t> gold_res_offset(arr_size);
    std::fill_n(gold_res_offset.data(), numCols, 0);
    std::fill_n(gold_res.data(), numRows * rowOffset, INVALID16);
    // count
    for (uint32_t i = 0; i < arr_size; ++i) {
        gold_res_offset[h_src[i]]++;
    }
    // offset
    uint32_t prv       = gold_res_offset[0];
    gold_res_offset[0] = 0;
    for (uint32_t i = 1; i < numCols; ++i) {
        uint16_t cur       = gold_res_offset[i];
        gold_res_offset[i] = gold_res_offset[i - 1] + prv;
        prv                = cur;
    }
    // fill in
    for (uint32_t i = 0; i < arr_size; ++i) {
        uint16_t col   = h_src[i];
        uint32_t row   = i / rowOffset;
        uint16_t start = gold_res_offset[col];
        uint16_t end   = (col == numCols - 1) ? numRows * rowOffset :
                                                gold_res_offset[col + 1];
        for (uint32_t j = start; j < end; ++j) {
            if (gold_res[j] == INVALID16) {
                gold_res[j] = row;
                break;
            }
        }
    }


    for (uint32_t i = 0; i < numCols; ++i) {
        uint32_t start = h_res_offset[i];
        uint32_t end =
            (i == numCols - 1) ? numRows * rowOffset : h_res_offset[i + 1];
        std::sort(h_res.data() + start, h_res.data() + end);
    }


    // compare
    bool is_offset_okay = compare<uint16_t, uint16_t>(
        h_res_offset.data(), gold_res_offset.data(), numCols, false);
    bool is_value_okay = compare<uint16_t, uint16_t>(
        h_res.data(), gold_res.data(), arr_size, false);

    GPU_FREE(d_src);
    GPU_FREE(d_offset);
    GPU_FREE(d_bitmask);

    EXPECT_TRUE(is_offset_okay);
    EXPECT_TRUE(is_value_okay);
}