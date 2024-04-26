#pragma once

#include <numeric>

template <typename T>
struct FrameStepper
{
    T   frame_length;  // length of a frame in seconds
    int frame_count;   // always the frame currently being processed
    T   current_time;  // current time within a frame
    int step_count;    // current step within the frame


    FrameStepper(T frame_len)
        : frame_length(frame_len),
          frame_count(0),
          current_time(0),
          step_count(0)
    {
    }

    /**
     * @brief adjust the timestep to land on a frame time, or to use more evenly
     * spaced steps if close to a frame time
     */
    T get_step_length(T max_step)
    {
        if (current_time + max_step > frame_length) {
            max_step = frame_length - current_time;
        }

        return max_step;
    }

    /**
     * @brief we're done when current time is very close or past the
     * frame_length
     */
    bool done_frame()
    {
        return current_time >= frame_length - 1e-7;
    }

    void advance_step(T step_length)
    {
        current_time += step_length;
        ++step_count;
    }

    void next_frame()
    {
        current_time = 0;
        step_count   = 0;
        ++frame_count;
    }

    int get_step_count()
    {
        return step_count;
    }

    int get_frame()
    {
        return frame_count;
    }

    T get_time()
    {
        return (frame_count - 1) * frame_length + current_time;
    }
};