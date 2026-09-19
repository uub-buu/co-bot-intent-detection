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

  return to_window_tensor(std::vector<CocoFrame>(frames_.begin(), frames_.end()));
}

std::optional<std::vector<float>> WindowingBuffer::flush()
{
  std::vector<CocoFrame> padded_frames;

  /* Nothing accumulated, or already long enough that add_frame() covered it */
  if (frames_.empty() || (frames_.size() >= static_cast<size_t>(kWindowFrameCount)))
  {
    return std::nullopt;
  }

  padded_frames.reserve(kWindowFrameCount);
  for (int i = 0; i < kWindowFrameCount; ++i)
  {
    padded_frames.push_back(frames_[i % frames_.size()]);
  }

  return to_window_tensor(padded_frames);
}

void WindowingBuffer::reset()
{
  frames_.clear();
}

std::vector<float> WindowingBuffer::to_window_tensor
(
  const std::vector<CocoFrame>& frames
) const
{
  std::vector<float> tensor(kWindowElementCount, 0.f);
  int                t = 0;

  /* Person slot 1 stays zero-initialized for every joint/channel below. */
  for (const CocoFrame& frame : frames)
  {
    for (int v = 0; v < kNumJointsCoco17; ++v)
    {
      const CocoJoint& joint = frame[v];

      const int base =
        ((t * kNumJointsCoco17 + v) * kJointChannelsCoco) * kMaxPersons;

      /* Person slot 0: Channel order is x, y, score. */
      tensor[base + 0 * kMaxPersons] = joint.x;
      tensor[base + 1 * kMaxPersons] = joint.y;
      tensor[base + 2 * kMaxPersons] = joint.score;
    }

    ++t;
  }

  return tensor;
}
