#ifndef WINDOWING_BUFFER_H
#define WINDOWING_BUFFER_H

/*
  windowing_buffer.h

  Accumulates a stream of per-frame COCO-17 joint arrays into fixed-length
  windows matching the Lite-STGCN model's input tensor: 100 frames, 17 joints,
  3 channels (x, y, score), 2 person slots. Confirmed via snpe-dlc-info on
  lite_stgcn_hmdb51.dlc: input "keypoint" shape [1, 100, 17, 3, 2].

  Only one person is tracked by this pipeline, so the second person slot is
  always zero-padded. This matches PySKL's own convention for single person
  clips.

  Output is a flat, [T, V, C, M] ordered float buffer (batch dimension N=1 is
  implicit, not included), ready to be copied straight into an Snpe_ITensor by
  the STGCN offload class, the same way PoseDSPOffload's preocess() builds its
  input tensor.
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

/*
 * C: channels per joint (x, y, score), fixed by the model's input tensor shape
 */
constexpr int kJointChannelsCoco = 3;

/*
 * M: max tracked persons, fixed by the model's input tensor shape. Only person
 * slot 0 is ever populated; slot 1 is always zero-padded.
 */
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
     * Feeds one frame's COCO-17 joints in. Returns a flattened, [T, V, C, M]
     * ordered window (size kWindowElementCount) once kWindowFrameCount
     * frames have accumulated, otherwise std::nullopt. Once full, the
     * window slides forward by one frame per call (the oldest frame is
     * dropped first), so a new window is produced on every call from then
     * on.
     */
    std::optional<std::vector<float>> add_frame(const CocoFrame& frame);

    /*
     * flush
     *
     * Call once at end-of-stream (source exhausted) to handle videos
     * shorter than kWindowFrameCount, which add_frame() never produces a
     * window for on its own. If any frames were accumulated but fewer
     * than kWindowFrameCount, pads up to kWindowFrameCount by cyclically
     * repeating the frames already seen (index modulo frame count) --
     * matching PySKL's UniformSampleFrames convention for short clips at
     * eval time -- and returns the resulting window. Returns std::nullopt
     * if no frames were ever accumulated, or if the stream was already
     * long enough that add_frame() already emitted every window (nothing
     * left to flush).
     */
    std::optional<std::vector<float>> flush();

    void reset();

  private:
    std::vector<float> to_window_tensor(const std::vector<CocoFrame>& frames) const;

    std::deque<CocoFrame> frames_;
};

#endif /* WINDOWING_BUFFER_H */
