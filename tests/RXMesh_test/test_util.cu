#include "gtest/gtest.h"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/rxmesh_queries.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

template <uint32_t rowOffset, uint32_t blockThreads, uint32_t itemPerThread>
__global__ static void k_test_block_mat_transpose(uint16_t*      d_src,
                                                  const uint32_t num_rows,
                                                  const uint32_t num_cols,
                                                  uint16_t*      d_output)
{

    rxmesh::block_mat_transpose<rowOffset, blockThreads, itemPerThread>(
        num_rows, num_cols, d_src, d_output);
}

template <typename T, uint32_t blockThreads>
__global__ static void k_test_block_exclusive_sum(T* d_src, const uint32_t size)
{
    rxmesh::cub_block_exclusive_sum<T, blockThreads>(d_src, size);
}

template <typename T>
__global__ static void k_test_atomicAdd(T* d_val)
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

    k_test_block_exclusive_sum<uint32_t, blockThreads>
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


    k_test_atomicAdd<T><<<1, threads>>>(d_val);

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

TEST(Util, AtomicAdd)
{
    EXPECT_TRUE(test_atomicAdd<uint16_t>()) << "uint16_t failed";
    EXPECT_TRUE(test_atomicAdd<uint8_t>()) << "uint8_t failed";
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

    uint16_t *d_src, *d_offset;
    CUDA_ERROR(cudaMalloc((void**)&d_src, h_src.size() * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_offset, h_src.size() * sizeof(uint16_t)));
    CUDA_ERROR(cudaMemcpy(d_src,
                          h_src.data(),
                          h_src.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));


    k_test_block_mat_transpose<rowOffset, threads, item_per_thread>
        <<<blocks, threads, numRows * rowOffset * sizeof(uint32_t)>>>(
            d_src, numRows, numCols, d_offset);

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
    bool passed = true;
    if (!compare<uint16_t, uint16_t>(
            h_res.data(), gold_res.data(), arr_size, false) ||
        !compare<uint16_t, uint16_t>(
            h_res_offset.data(), gold_res_offset.data(), numCols, false)) {
        passed = false;
    }

    GPU_FREE(d_src);
    GPU_FREE(d_offset);

    EXPECT_TRUE(passed);
}