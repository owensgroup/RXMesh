#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <random>
#include "rxmesh/util/macros.h"

namespace rxmesh {

/**
 * get_cmd_option()
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
 * cmd_option_exists()
 */
inline bool cmd_option_exists(char**             begin,
                              char**             end,
                              const std::string& option)
{
    // https://stackoverflow.com/a/868894/1608232
    return std::find(begin, end, option) != end;
}

/**
 * print_device_memory_usage()
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
 * find_index()
 */
template <typename T>
inline uint32_t find_index(const T entery, const std::vector<T>& vect)
{
    // get index of entry in vector

    typename std::vector<T>::const_iterator it =
        std::find(vect.begin(), vect.end(), entery);
    if (it == vect.end()) {
        return std::numeric_limits<uint32_t>::max();
    }
    return uint32_t(it - vect.begin());
}

/**
 * find_index()
 */
template <typename T>
inline T find_index(const T* arr, const T arr_size, const T val)
{
    // get index of entry in array
    const T* begin = arr;
    const T* end   = arr + arr_size;
    const T* it    = std::find(begin, end, val);
    if (it == end) {
        return std::numeric_limits<T>::max();
    }
    return it - begin;
}

/**
 * random_shuffle()
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
 * fill_with_sequential_numbers()
 */
template <typename T>
inline void fill_with_sequential_numbers(T*             arr,
                                         const uint32_t size,
                                         const T        start = 0)
{
    std::iota(arr, arr + size, start);
}

/**
 * compare()
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
 * copy()
 */
template <typename T>
void copy(const std::vector<T>& src, std::vector<T>& tar, int tar_start = 0)
{
    std::copy(src.begin(), src.end(), tar.data() + tar_start);
}

/**
 * compute_avg_stddev()
 */
template <typename T>
inline void compute_avg_stddev(const T* arr,
                               uint32_t size,
                               double&  avg,
                               double&  stddev)
{
    if (size == 1) {
        avg    = arr[0];
        stddev = 0;
        return;
    }
    avg = 0;
    // compute avg
    for (uint32_t i = 0; i < size; i++) {
        avg += arr[i];
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
 * compute_avg_stddev_max_min_rs()
 * computes the average and stddev where the input is running sum (output of
 * exclusive sum) the input size is actually size + 1
 */
template <typename T>
inline void compute_avg_stddev_max_min_rs(const T* arr_rs,
                                          uint32_t size,
                                          double&  avg,
                                          double&  stddev,
                                          T&       max,
                                          T&       min)
{
    uint32_t* arr = (uint32_t*)malloc(size * sizeof(uint32_t));
    max           = std::numeric_limits<T>::min();
    min           = std::numeric_limits<T>::max();
    for (uint32_t i = 0; i < size; i++) {
        // arr[i] = arr_rs[i + 1] - arr_rs[i];
        uint32_t start = (i == 0) ? 0 : arr_rs[i - 1];
        uint32_t end   = arr_rs[i];
        arr[i]         = end - start;
        max            = std::max(max, arr[i]);
        min            = std::min(min, arr[i]);
    }

    compute_avg_stddev(arr, size, avg, stddev);

    free(arr);
}

/**
 * binary_search()
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
 * inplace_remove_duplicates_sorted()
 * in-place remove duplicates from sorted vector
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
 * shuffle_obj()
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
 * remove_extension()
 */
inline std::string remove_extension(const std::string& filename)
{  // https://stackoverflow.com/a/6417908/1608232
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos)
        return filename;
    return filename.substr(0, lastdot);
}

/**
 * extract_file_name()
 */
inline std::string extract_file_name(const std::string& full_path)
{
    // given full path, we extract the file name without extension
    std::string filename  = remove_extension(full_path);
    size_t      lastslash = filename.find_last_of("/\\");

    return filename.substr(lastslash + 1);
}

/**
 * in_place_matrix_transpose()
 */
template <class RandomIterator>
void in_place_matrix_transpose(RandomIterator first,
                               RandomIterator last,
                               uint64_t       m)
{
    // in-place matrix transpose represented as row-major format with m
    // number for columns
    // https://stackoverflow.com/a/9320349/1608232
    const uint64_t mn1 = (last - first - 1);
    const uint64_t n   = (last - first) / m;

    std::vector<bool> visited(last - first, false);

    RandomIterator cycle = first;
    while (++cycle != last) {
        if (visited[cycle - first]) {
            continue;
        }
        uint64_t a = cycle - first;
        do {
            a = (a == mn1) ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = true;
        } while ((first + a) != cycle);
    }
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
 * @param v0 
 * @param v1
 * @return
 */
inline std::pair<uint32_t, uint32_t> edge_key(const uint32_t v0,
                                              const uint32_t v1)
{
    uint32_t i = std::max(v0, v1);
    uint32_t j = std::min(v0, v1);
    return std::make_pair(i, j);
}
}  // namespace detail
}  // namespace rxmesh