/*
  windowing_buffer.cpp

  Implementation file for the CPU windowing buffer
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include "windowing_buffer.h"


/*******************************************************************************
 * Functions
 ******************************************************************************/

std::optional<std::vector<float>> WindowingBuffer::add_frame
(
  const CocoFrame& frame
)
{
  frames_.push_back(frame);

  if (frames_.size() > kWindowFrameCount)
  {
    frames_.pop_front();
  }

  if (frames_.size() < kWindowFrameCount)
  {
    return std::nullopt;
  }

  return to_window_tensor();
}

void WindowingBuffer::reset()
{
  frames_.clear();
}

std::vector<float> WindowingBuffer::to_window_tensor() const
{
  std::vector<float> tensor(kWindowElementCount, 0.f);
  int                t = 0;

  /* Person slot 1 (the second position on the M axis) is left at its
     zero-initialized value for every joint/channel below. */
  for (const CocoFrame& frame : frames_)
  {
    for (int v = 0; v < kNumJointsCoco17; ++v)
    {
      const CocoJoint& joint = frame[v];

      const int base =
        ((t * kNumJointsCoco17 + v) * kJointChannelsCoco) * kMaxPersons;

      /* Person slot 0: real data. Channel order is x, y, score. */
      tensor[base + 0 * kMaxPersons] = joint.x;
      tensor[base + 1 * kMaxPersons] = joint.y;
      tensor[base + 2 * kMaxPersons] = joint.score;
    }

    ++t;
  }

  return tensor;
}
