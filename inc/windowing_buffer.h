#ifndef WINDOWING_BUFFER_H
#define WINDOWING_BUFFER_H

/*
  windowing_buffer.h

  Buffers COCO-17 joints into 100-frame windows matching Lite-STGCN's
  input: [1, 100, 17, 3, 2].

  Only one person is tracked; second person slot is zero-padded,
  matching PySKL's single-person convention.

  Output is a flat [T, V, C, M] buffer (N=1 implicit), ready to copy
  into an Snpe_ITensor.
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include <deque>
#include <optional>
#include <vector>

#include "joint_remap.h"


/*******************************************************************************
 * Data
 ******************************************************************************/

/* T: frames per window, fixed by the model's input tensor shape */
constexpr int kWindowFrameCount = 100;

/* C: channels per joint (x, y, score) */
constexpr int kJointChannelsCoco = 3;

/* M: max tracked persons. Only slot 0 is populated; slot 1 is zero-padded. */
constexpr int kMaxPersons = 2;

/* Total element count of one flattened window, in [T, V, C, M] order */
constexpr int kWindowElementCount =
  kWindowFrameCount * kNumJointsCoco17 * kJointChannelsCoco * kMaxPersons;


/*******************************************************************************
 * Type definitions
 ******************************************************************************/

class WindowingBuffer
{
  public:
    WindowingBuffer() = default;

    /*
     * add_frame
     *
     * Feeds in one frame. Returns a flattened [T, V, C, M] window once 100
     * frames have accumulated, else nullopt. Slides by one frame/call aftrer
     * that.
     */
    std::optional<std::vector<float>> add_frame(const CocoFrame& frame);

    /*
     * flush
     *
     * Call at end of stream for videos shorter than 100 frames, which
     * add_frame() never emits a window for. Pads to 100 by cyclically
     * repeating frames seen so far, matching PySKL's behaviour.
     */
    std::optional<std::vector<float>> flush();

    void reset();

  private:
    std::vector<float> to_window_tensor(const std::vector<CocoFrame>& frames) const;

    std::deque<CocoFrame> frames_;
};

#endif /* WINDOWING_BUFFER_H */
