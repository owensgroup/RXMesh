#pragma once

#include <numeric>

template <typename T>
struct Simulation
{
   public:
    /// simulation time step size
    ///
    T m_dt;

    /// end time
    ///
    T m_max_t;

    /// current simulation time
    ///
    T m_curr_t;

    /// whether we're currently running an entire simulation
    ///
    bool m_running;

    /// whether we're currently running a single timestep of the simulation
    ///
    bool m_currently_advancing_simulation;


    /// Specify a target time step and optional end time
    ///
    Simulation(T in_dt, T in_max_t = std::numeric_limits<T>::max())
        : m_dt(in_dt),
          m_max_t(in_max_t),
          m_curr_t(0),
          m_running(false),
          m_currently_advancing_simulation(false)
    {
        assert(m_dt > 0);
        assert(m_max_t > 0);
    }

    bool done_simulation()
    {
        return m_curr_t > m_max_t - 1e-7;
    }
};
