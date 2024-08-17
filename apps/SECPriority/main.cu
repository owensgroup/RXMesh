#include <cuco/priority_queue.cuh>
#include <cuco/detail/pair.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <map>
#include <vector>

#include <iostream>
#include <random>

using namespace cuco;
namespace cg = cooperative_groups;

// grab some bits from priority queue tests and benchmarks

// -- simulate reading the mesh, computing edge length
// -- cuco:pair<float, uint32_t>
//
// setup pair_less template
//
// setup device function to pop items from queue
//

template <typename T>
struct pair_less 
{
    __host__ __device__ bool operator()(const T& a, const T& b) const
    {
        return a.first < b.first;
    }
};

template <typename PairType, typename OutputIt>
void generate_kv_pairs_uniform(OutputIt output_begin, OutputIt output_end)
{
  std::random_device rd;
  std::mt19937 gen{rd()};

  const auto num_keys = std::distance(output_begin, output_end);
  for(auto i = 0; i < num_keys; i++)
  {
    output_begin[i] = {static_cast<typename PairType::first_type>(gen()),
                       static_cast<typename PairType::second_type>(i)};
  }
}

void sp_pair()
{
  // Setup the cuco::priority_queue
  const size_t insertion_size = 200;
  const size_t deletion_size  = 100;
  using PairType              = cuco::pair<float, uint32_t>;
  using Compare               = pair_less<PairType>;

  cuco::priority_queue<PairType, Compare> pq(insertion_size);

  // Generate data for the queue
  std::vector<PairType> h_pairs(insertion_size);
  generate_kv_pairs_uniform<PairType>(h_pairs.begin(), h_pairs.end());

  for(auto i = 0; i < h_pairs.size(); i++)
  {
    std::cout << "Priority: " << h_pairs[i].first 
              << "\tID: " << h_pairs[i].second << "\n";
  }

  // Fill the priority queue
  thrust::device_vector<PairType> d_pairs(h_pairs);
  pq.push(d_pairs.begin(), d_pairs.end());
  cudaDeviceSynchronize();

  // Pop the priority queue
  thrust::device_vector<PairType> d_popped(deletion_size);
  pq.pop(d_popped.begin(), d_popped.end());
  cudaDeviceSynchronize();

  std::cout << "-----After Pop-----\n";
  thrust::host_vector<PairType> h_popped(d_popped);
  for(auto i = 0; i < h_popped.size(); i++)
  {
    std::cout << "Priority: " << h_popped[i].first 
              << "\tID: " << h_popped[i].second << "\n";
  }
}

int main(int argc, char* argv[])
{
  sp_pair();

  return 0;
}