#pragma once

#include <chrono>
#include <map>
#include "rxmesh/util/macros.h"


namespace rxmesh {

struct GPUTimer
{
    GPUTimer(cudaStream_t stream = NULL) : m_stream(stream)
    {
        CUDA_ERROR(cudaEventCreate(&m_start));
        CUDA_ERROR(cudaEventCreate(&m_stop));
    }
    ~GPUTimer()
    {
        CUDA_ERROR(cudaEventDestroy(m_start));
        CUDA_ERROR(cudaEventDestroy(m_stop));
    }
    void start()
    {
        CUDA_ERROR(cudaEventRecord(m_start, m_stream));
    }
    void stop()
    {
        CUDA_ERROR(cudaEventRecord(m_stop, m_stream));
    }
    float elapsed_millis()
    {
        float elapsed = 0;
        CUDA_ERROR(cudaEventSynchronize(m_stop));
        CUDA_ERROR(cudaEventElapsedTime(&elapsed, m_start, m_stop));
        return elapsed;
    }

   private:
    cudaEvent_t  m_start, m_stop;
    cudaStream_t m_stream;
};


struct CPUTimer
{
    CPUTimer()
    {
    }
    ~CPUTimer()
    {
    }
    void start()
    {
        m_start = std::chrono::high_resolution_clock::now();
    }
    void stop()
    {
        m_stop = std::chrono::high_resolution_clock::now();
    }
    float elapsed_millis()
    {
        return std::chrono::duration<float, std::milli>(m_stop - m_start)
            .count();
    }

   private:
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_stop;
};

template <typename TimerT>
struct Timers
{
    Timers()  = default;
    ~Timers() = default;

    void add(std::string name)
    {
        m_timers.insert(std::make_pair(name, std::make_shared<TimerT>()));
        m_total_time.insert(std::make_pair(name, 0));
    }

    void start(std::string name)
    {
        m_timers.at(name)->start();
    }

    void stop(std::string name)
    {
        m_timers.at(name)->stop();

        float new_time =
            m_total_time.at(name) + m_timers.at(name)->elapsed_millis();

        m_total_time.insert_or_assign(name, new_time);

    }

    float elapsed_millis(std::string name)
    {
        return m_total_time.at(name);
    }

    std::unordered_map<std::string, std::shared_ptr<TimerT>> m_timers;
    std::unordered_map<std::string, float>                   m_total_time;
};
}  // namespace rxmesh