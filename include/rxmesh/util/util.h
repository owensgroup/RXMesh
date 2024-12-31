#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <random>
#include "rxmesh/util/macros.h"

namespace rxmesh {

/**
 * @brief Parse for an option. Maninly used to parse user input from CMD
 */
inline char* get_cmd_option(char** begin, char** end, const std::string& option)
{
    // https://stackoverflow.com/a/868894/1608232
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

/**
 * @brief Check if an input string exists. Mainly used to check if input option
 * exists in CMD
 */
inline bool cmd_option_exists(char**             begin,
                              char**             end,
                              const std::string& option)
{
    // https://stackoverflow.com/a/868894/1608232
    return std::find(begin, end, option) != end;
}

/**
 * @brief Print current GPU memory usage
 */
inline void print_device_memory_usage()
{
    // print how much memory is available, used and free on the current device
    size_t free_t, total_t;
    CUDA_ERROR(cudaMemGetInfo(&free_t, &total_t));
    double free_m  = (double)free_t / (double)1048576.0;
    double total_m = (double)total_t / (double)1048576.0;
    double used_m  = total_m - free_m;
    RXMESH_TRACE(
        " device memory mem total = {} (B) [{} (MB)]", total_t, total_m);
    RXMESH_TRACE(" device memory free: {} (B) [{} (MB)]", free_t, free_m);
    RXMESH_TRACE(" device memory mem used: {} (MB)", used_m);
}


/**
 * @brief Find the index of an entry in a vector
 * @tparam T type of the entry and vector elements
 * @param entry to search for
 * @param vect input vector to search in
 * @return return the index of the entry or std::numeric_limits<uint32_t>::max()
 * if it is not found
 */
template <typename T>
inline uint32_t find_index(const T entry, const std::vector<T>& vect)
{
    // get index of entry in vector

    typename std::vector<T>::const_iterator it =
        std::find(vect.begin(), vect.end(), entry);
    if (it == vect.end()) {
        return std::numeric_limits<uint32_t>::max();
    }
    return uint32_t(it - vect.begin());
}

/**
 * @brief Find the index of an entry an array given its size
 * @tparam T type of the entry and array elements
 * @param entry to search for
 * @param arr input array to search in
 * @param arr_size size of the input array (arr)
 * @return return the index of the entry or std::numeric_limits<uint32_t>::max()
 * if it is not found
 */
template <typename T>
inline T find_index(const T* arr, const T arr_size, const T entry)
{
    // get index of entry in array
    const T* begin = arr;
    const T* end   = arr + arr_size;
    const T* it    = std::find(begin, end, entry);
    if (it == end) {
        return std::numeric_limits<T>::max();
    }
    return it - begin;
}

/**
 * @brief Shuffle the content of an input array randomly
 */
template <typename T>
inline void random_shuffle(T*             d_in,
                           const uint32_t end,
                           const uint32_t start = 0)
{
    std::random_device rd;
    std::mt19937       g(rd());
    std::shuffle(d_in + start, d_in + end, g);
}

/**
 * @brief Fill in an array with sequential numbers
 */
template <typename T>
inline void fill_with_sequential_numbers(T*             arr,
                                         const uint32_t size,
                                         const T        start = 0)
{
    std::iota(arr, arr + size, start);
}


/**
 * @brief Fill in an array with random numbers
 */
template <typename T>
inline void fill_with_random_numbers(T* arr, const uint32_t size)
{
    fill_with_sequential_numbers(arr, size);
    random_shuffle(arr, size);
}


/**
 * @brief Compare the content of two input arrays
 */
template <typename T, typename dataT>
bool compare(const dataT* gold,
             const dataT* arr,
             const T      size,
             const bool   verbose = false,
             const dataT  tol     = 10E-5)
{

    bool result = true;
    for (T i = 0; i < size; i++) {
        if (std::abs(double(gold[i]) - double(arr[i])) > tol) {
            if (verbose) {
                RXMESH_WARN("compare() mismatch at {} gold = {} arr = {} ",
                            i,
                            gold[i],
                            arr[i]);
                result = false;
            } else {
                // it is not verbose, don't bother running through all entires
                return false;
            }
        }
    }
    return result;
}

/**
 * @brief Copy the content of one vector to another
 */
template <typename T>
void copy(const std::vector<T>& src, std::vector<T>& tar, int tar_start = 0)
{
    std::copy(src.begin(), src.end(), tar.data() + tar_start);
}


/**
 * @brief Compute the average and standard deviation of an input array
 */
template <typename T>
inline void compute_avg_stddev_max_min(const T* arr,
                                       uint32_t size,
                                       double&  avg,
                                       double&  stddev,
                                       T&       max,
                                       T&       min)
{
    max = std::numeric_limits<T>::min();
    min = std::numeric_limits<T>::max();

    if (size == 1) {
        avg    = arr[0];
        max    = arr[0];
        min    = arr[0];
        stddev = 0;
        return;
    }
    avg = 0;
    // compute avg
    for (uint32_t i = 0; i < size; i++) {
        avg += arr[i];

        max = std::max(max, arr[i]);
        min = std::min(min, arr[i]);
    }
    avg /= size;

    // compute stddev
    double sum = 0;
    for (uint32_t i = 0; i < size; i++) {
        double diff = double(arr[i]) - avg;
        sum += diff * diff;
    }
    stddev = std::sqrt(double(sum) / double(size - 1));
    return;
}

/**
 * @brief binary search in a vector (has to be sorted --- not checked)
 */
template <typename T>
inline size_t binary_search(const std::vector<T>& list,
                            const T               target,
                            const size_t          start,
                            const size_t          end)
{
    // lookup where target in list. List is supposed to be sorted since
    // we are using binary search
    assert(list.size() >= end);
    assert(end >= start);


    if (end - start < 20) {
        // linear search for small searches
        for (size_t i = start; i < end; ++i) {
            if (list[i] == target) {
                return i;
            }
        }
    } else {
        // binary search
        auto loc =
            std::lower_bound(list.begin() + start, list.begin() + end, target);
        if (loc != (list.begin() + end) && (target == *loc)) {
            return loc - list.begin();
        }
    }

    return std::numeric_limits<size_t>::max();
}


/**
 * @brief in-place remove duplicates from sorted vector
 * requires one pass over all elements in sort_vec
 * it also resize sort_vec to contain only the unique values
 */
template <typename T>
inline void inplace_remove_duplicates_sorted(std::vector<T>& sort_vec)
{
    if (sort_vec.size() == 0) {
        return;
    }

    // leave the first value
    uint32_t next_unique_id = 1;
    T        prev_value     = sort_vec.front();
    for (uint32_t i = 1; i < sort_vec.size(); ++i) {
        T curr_val = sort_vec[i];
        if (curr_val != prev_value) {
            sort_vec[next_unique_id++] = curr_val;
            prev_value                 = curr_val;
        }
    }

    sort_vec.resize(next_unique_id);
}

/**
 * @brief Given the vertex coordinates and face indices, shuffle the input mesh
 * randomly --- both vertices and face indices
 */
template <typename T>
inline void shuffle_obj(std::vector<std::vector<uint32_t>>& Faces,
                        std::vector<std::vector<T>>&        Verts)
{
    // shuffle verts
    {
        std::vector<uint32_t> rand(Verts.size());
        fill_with_sequential_numbers(rand.data(), rand.size());
        random_shuffle(rand.data(), rand.size());

        for (auto& f : Faces) {
            for (uint32_t i = 0; i < f.size(); ++i) {
                f[i] = rand[f[i]];
            }
        }

        std::vector<std::vector<T>> verts_old(Verts);
        for (uint32_t v = 0; v < Verts.size(); ++v) {
            for (uint32_t i = 0; i < Verts[v].size(); ++i) {
                Verts[rand[v]][i] = verts_old[v][i];
            }
        }
    }

    // shuffle faces
    {
        std::vector<uint32_t> rand(Faces.size());
        fill_with_sequential_numbers(rand.data(), rand.size());
        random_shuffle(rand.data(), rand.size());

        std::vector<std::vector<uint32_t>> faces_old(Faces);
        for (uint32_t f = 0; f < Faces.size(); ++f) {
            for (uint32_t i = 0; i < Faces[f].size(); ++i) {
                // Verts[rand[v]][i] = verts_old[v][i];
                Faces[rand[f]][i] = faces_old[f][i];
            }
        }
    }
}


/**
 * @brief Remove the extension of an input file path
 */
inline std::string remove_extension(const std::string& filename)
{  // https://stackoverflow.com/a/6417908/1608232
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos)
        return filename;
    return filename.substr(0, lastdot);
}

/**
 * @brief Extract file path given its full path
 */
inline std::string extract_file_name(const std::string& full_path)
{
    // given full path, we extract the file name without extension
    std::string filename  = remove_extension(full_path);
    size_t      lastslash = filename.find_last_of("/\\");

    return filename.substr(lastslash + 1);
}

/**
 * @brief given an initial number of bytes, increase this number such that it
 * multiple of alignment
 */
__device__ __host__ __inline__ uint32_t expand_to_align(
    uint32_t init_bytes,
    uint32_t alignment = 128)
{
    uint32_t remainder = init_bytes % alignment;
    if (remainder == 0) {
        return init_bytes;
    }
    return init_bytes + alignment - remainder;
};

/**
 * @brief find the next multiple of 32
 * https://codegolf.stackexchange.com/a/17852
 */
__device__ __host__ __inline__ uint16_t round_to_next_multiple_32(uint16_t num)
{
    if (num % 32 != 0) {
        return (num | 31) + 1;
    } else {
        return num;
    }
}

/**
 * @brief Cast a uint32_t to an int, throwing an exception if the value is too
 * large to fit in an int.
 */
__host__ __inline__ bool arr_check_uint32_to_int_cast(const uint32_t* arr,
                                                      size_t          size)
{
    static_assert(sizeof(int) >= sizeof(uint32_t),
                  "int must be at least 32 bits wide");
    static_assert(std::is_same<int, std::int32_t>::value,
                  "int must be exactly 32 bits");

    for (size_t i = 0; i < size; ++i) {
        if (arr[i] > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            return false;  // Unsafe to cast
        }
    }
    return true;  // Safe to cast
}

namespace detail {

/**
 * @brief hash function that takes a pair of vertices and returns a unique
 * values. Used for storing vertex-edge relation in std map
 */
struct edge_key_hash
{
    // www.techiedelight.com/use-std-pair-key-std-unordered_map-cpp/
    template <class T>
    inline std::size_t operator()(const std::pair<T, T>& e_key) const
    {
        return std::hash<T>()(e_key.first * 8191 + e_key.second * 11003);
    }
};

/**
 * @brief return consistent edge key given two vertices
 */
inline std::pair<uint32_t, uint32_t> edge_key(const uint32_t v0,
                                              const uint32_t v1)
{
    uint32_t i = std::max(v0, v1);
    uint32_t j = std::min(v0, v1);
    return std::make_pair(i, j);
}
}  // namespace detail

/**
 * @brief given a pointer, this function returns a pointer to the first location
 * at the boundary of a given alignment size. This what std:align does but it
 * does not work with CUDA so this a stripped down version of it.
 * @tparam T type of the pointer
 * @param byte_alignment number of bytes to get the pointer to be aligned to
 * @param ptr input/output pointer pointing at first usable location. On return,
 * it will be properly aligned to the beginning of the first element that is
 * aligned to alignment
 */
template <typename T>
__device__ __host__ __inline__ void align(const std::size_t byte_alignment,
                                          T*&               ptr) noexcept
{
    const uint64_t intptr    = reinterpret_cast<uint64_t>(ptr);
    const uint64_t remainder = intptr % byte_alignment;
    if (remainder == 0) {
        return;
    }
    const uint64_t aligned = intptr + byte_alignment - remainder;
    ptr                    = reinterpret_cast<T*>(aligned);
}

/**
 * @brief get cuSparse/cuSolver data type for T
 */
template <typename T>
__host__ __inline__ cudaDataType_t cuda_type()
{
    if (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    } else if (std::is_same_v<T, double>) {
        return CUDA_R_64F;
    } else if (std::is_same_v<T, cuComplex>) {
        return CUDA_C_32F;
    } else if (std::is_same_v<T, cuDoubleComplex>) {
        return CUDA_C_64F;
    } else if (std::is_same_v<T, int8_t>) {
        return CUDA_R_8I;
    } else if (std::is_same_v<T, uint8_t>) {
        return CUDA_R_8U;
    } else if (std::is_same_v<T, int16_t>) {
        return CUDA_R_16I;
    } else if (std::is_same_v<T, uint16_t>) {
        return CUDA_R_16U;
    } else if (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
        return CUDA_R_32I;
    } else if (std::is_same_v<T, uint32_t>) {
        return CUDA_R_32U;
    } else if (std::is_same_v<T, int64_t>) {
        return CUDA_R_64I;
    } else if (std::is_same_v<T, uint64_t>) {
        return CUDA_R_64U;
    } else {
        RXMESH_ERROR(
            "Unsupported type. Sparse/Dense Matrix in RXMesh can support "
            "different data type but for the solver, only float, double, "
            "cuComplex, and cuDoubleComplex are supported");    
        //to silence compiler warning 
        return CUDA_R_32F;    
    }
}
}  // namespace rxmesh