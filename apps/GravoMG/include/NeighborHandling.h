#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include "cuda_runtime.h"

struct Mutex
{
#ifdef __CUDA_ARCH__
#if (__CUDA_ARCH__ < 700)
#error ShmemMutex requires compiling with sm70 or higher since it relies on Independent Thread Scheduling
#endif
#endif

    __device__ Mutex() : m_mutex(0)
    {
    }
    __device__ void lock()
    {
#ifdef __CUDA_ARCH__
        assert(&m_mutex);
        __threadfence();
        while (::atomicCAS(&m_mutex, 0, 1) != 0) {
            __threadfence();
        }
        __threadfence();
#endif
    }
    __device__ void unlock()
    {
#ifdef __CUDA_ARCH__
        assert(&m_mutex);
        __threadfence();
        ::atomicExch(&m_mutex, 0);
        __threadfence();
#endif
    }
    int m_mutex;
};

// Node structure for linked list of neighbors, storing a chunk of 64 neighbors
struct NeighborNode
{
    static const int CHUNK_SIZE = 64;
    int              neighbors[CHUNK_SIZE];
    int           count;  // Number of neighbors currently stored in this chunk
    NeighborNode* next;

    __device__ NeighborNode() : count(0), next(nullptr)
    {
    }

    // Add a neighbor to the chunk, returns true if successfully added, false if
    // full
    __device__ bool addNeighbor(int neighbor)
    {
        if (count < CHUNK_SIZE) {
            neighbors[count++] = neighbor;
            return true;
        }
        return false;
    }

    // Check if a neighbor exists in this chunk
    __device__ bool containsNeighbor(int neighbor)
    {
        for (int i = 0; i < count; ++i) {
            if (neighbors[i] == neighbor) {
                return true;
            }
        }
        return false;
    }
};

/**
 * \brief used to store neighbors per vertex in parallel
 */
class VertexNeighbors
{
   public:
    __device__ VertexNeighbors() : head(nullptr), tail(nullptr), mutex()
    {
        // cudaMallocManaged(mutex, sizeof(Mutex));
        cudaMallocManaged(&neighborsList, sizeof(int) * 64);
    }

    // Add a neighbor to the list
    __device__ void addNeighbor(int neighbor)
    {
        mutex.lock();
        if (containsNeighbor(neighbor)) {
            mutex.unlock();
            return;  // Neighbor already exists, do nothing
        }
        if (!tail || !tail->addNeighbor(neighbor)) {
            // Create a new node if the current tail is full or doesn't exist
            NeighborNode* newNode = new NeighborNode();
            if (tail) {
                tail->next = newNode;
            } else {
                head = newNode;
            }
            tail = newNode;
            tail->addNeighbor(neighbor);  // Add the neighbor to the new node
            // printf("\nneighbor %d added",neighbor);
        }
        mutex.unlock();
    }

    // Retrieve all neighbors
    __device__ int getNumberOfNeighbors(int max_neighbors = 0) const
    {
        int           count   = 0;  // Track the number of neighbors added
        NeighborNode* current = head;

        // Traverse the linked list
        while (current) {
            for (int i = 0; i < current->count; ++i) {
                ++count;
            }
            current = current->next;  // Move to the next node in the list
        }

        return count;  // Return the total number of neighbors added
    }

    __device__ void getNeighbors(int* thread_neighbors) const
    {
        int           count   = 0;  // Track the number of neighbors added
        NeighborNode* current = head;

        // Traverse the linked list
        while (current) {
            for (int i = 0; i < current->count; ++i) {
                thread_neighbors[count] = current->neighbors[i];
                ++count;
            }
            current = current->next;  // Move to the next node in the list
        }
    }


    // Destructor to clean up memory
    __device__ ~VertexNeighbors()
    {
        NeighborNode* current = head;
        while (current) {
            NeighborNode* temp = current;
            current            = current->next;
            delete temp;
        }
    }

   private:
    int*          neighborsList;
    NeighborNode* head;
    NeighborNode* tail;
    mutable Mutex mutex;

    // Check if a neighbor already exists in the list
    __device__ bool containsNeighbor(int neighbor) const
    {
        NeighborNode* current = head;
        while (current) {
            if (current->containsNeighbor(neighbor)) {
                return true;
            }
            current = current->next;
        }
        return false;
    }
};