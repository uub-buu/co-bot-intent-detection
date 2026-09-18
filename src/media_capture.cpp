/*
  media_capture.cpp

  We originally intended to be implemented on a real-time camera feed, but
  later scaled down to running on pre-recorded video files only.

  This reads frames on their own thread, and pushes them into a thread-safe
  queue. The consumer side of that queue runs the real pipeline: DSP pose
  estimation (PoseDSPOffload), MediaPipe -> COCO-17 joint remap, and the CPU
  windowing buffer. GPU STGCN offload stubbed.
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "joint_remap.h"
#include "pose_dsp_offload.h"
#include "windowing_buffer.h"
#include "stgcn_gpu_offload.h"
#include "normalization.h"
#include "class_labels.h"


/*******************************************************************************
 * Type definitions
 ******************************************************************************/

struct Frame
{
  cv::Mat image;
  int     index = 0;
  double  timestamp_ms = 0.0;
};

/*
 * ThreadSafeQueue
 *
 * Bounded producer/consumer queue between the capture thread and the
 * pose/remap/windowing pipeline.
 */
template <typename T>
class ThreadSafeQueue
{
  public:
    explicit ThreadSafeQueue(size_t max_size) : max_size_(max_size) {}

    /*
     * Blocks if the queue is full, so a slow consumer applies backpressure
     * to the capture thread instead of unbounded memory growth.
     */
    void push(T item)
    {
      std::unique_lock<std::mutex> lock(mutex_);

      not_full_.wait(
        lock, [this] { return queue_.size() < max_size_ || stopped_; }
      );

      if (stopped_)
      {
        return;
      }

      queue_.push_back(std::move(item));
      lock.unlock();
      not_empty_.notify_one();
    }

    /*
     * Blocks until an item is available or the queue is stopped, in which
     * case it returns std::nullopt.
     */
    std::optional<T> pop()
    {
      std::unique_lock<std::mutex> lock(mutex_);
      not_empty_.wait(
        lock, [this] { return !queue_.empty() || stopped_; }
      );

      if (queue_.empty())
      {
        return std::nullopt;
      }

      T item = std::move(queue_.front());
      queue_.pop_front();
      lock.unlock();
      not_full_.notify_one();
      return item;
    }

    void stop()
    {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
      }
      not_empty_.notify_all();
      not_full_.notify_all();
    }

  private:
    std::mutex              mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T>           queue_;
    size_t                  max_size_;
    bool                    stopped_ = false;
};

class VideoSource
{
  public:
    explicit VideoSource(const std::string& path)
    {
      opened_ = capture_.open(path);
    }

    bool is_opened() const { return opened_; }

    /*
     * Reads the next frame and converts it to RGB (OpenCV decodes as BGR by
     * default). Returns std::nullopt at end-of-stream or on a read error.
     */
    std::optional<cv::Mat> read_rgb_frame()
    {
      cv::Mat bgr_frame, rgb_frame;

      if (!capture_.read(bgr_frame) || bgr_frame.empty())
      {
        return std::nullopt;
      }

      cv::cvtColor(bgr_frame, rgb_frame, cv::COLOR_BGR2RGB);

      return rgb_frame;
    }

  private:
    cv::VideoCapture capture_;
    bool             opened_ = false;
};

/*
 * capture_loop
 *
 * Runs in its own thread, pushing frames into frame_queue until the source
 * runs out of frames or stop_requested is set.
 */
void capture_loop
(
  VideoSource&            source,
  ThreadSafeQueue<Frame>& frame_queue,
  std::atomic<bool>&      stop_requested
)
{
  std::chrono::steady_clock::time_point now;
  Frame                                 frame;
  double                                timestamp_ms;
  int                                   frame_index = 0;
  const auto                            start_time =
    std::chrono::steady_clock::now();

  while (!stop_requested.load())
  {
    auto maybe_frame = source.read_rgb_frame();
    if (!maybe_frame.has_value())
    {
      break;
    }

    now = std::chrono::steady_clock::now();
    timestamp_ms =
      std::chrono::duration<double, std::milli>(now - start_time).count();

    frame.image = std::move(*maybe_frame);
    frame.index = frame_index++;
    frame.timestamp_ms = timestamp_ms;

    frame_queue.push(frame);
  }

  frame_queue.stop();
  std::printf("Capture loop done. Total frames captured: %d\n", frame_index);
}

/*******************************************************************************
 * Functions
 ******************************************************************************/

/*
 * process_frame
 *
 * Runs one captured frame through the real pipeline: DSP pose estimation,
 * MediaPipe -> COCO-17 remap, and the CPU windowing buffer.
 */
void process_frame
(
  const Frame&     frame,
  PoseDSPOffload&  pose_model,
  WindowingBuffer& window_buffer,
  StgcnGpuOffload& stgcn_model
)
{
  JointFrame mediapipe_joints;
  const int  frame_width = frame.image.cols;
  const int  frame_height = frame.image.rows;

  if (!pose_model.estimate(frame.image, mediapipe_joints))
  {
    std::fprintf(
      stderr, "Pose estimation failed on frame %d\n", frame.index);
    return;
  }

  const CocoFrame raw_coco_frame = remap_mediapipe_to_coco17(mediapipe_joints);

  const CocoFrame norm_frame =
    normalize_coco_frame(raw_coco_frame, frame_width, frame_height);

  const std::optional<std::vector<float>> window =
    window_buffer.add_frame(norm_frame);

  if (window.has_value())
  {
    std::printf(
      "Frame %d: window ready (%zu floats) for GPU STGCN offload\n",
      frame.index, window->size());

    ClassificationResult result;

    if (stgcn_model.classify(*window, result))
    {
      std::printf(
        "Frame %d: predicted class=%d (%s) confidence=%.3f\n",
        frame.index, result.class_index, hmdb51_label(result.class_index),
        result.confidence);
    }
  }
}

/*
 * Main
 */
int main
(
  int    argc,
  char **argv
)
{
  /* Sanity check(s) */
  if (argc < 2)
  {
    std::fprintf(
      stderr, "Usage: %s <video_path> [pose_model_path]\n", argv[0]);
    return 1;
  }

  const std::string pose_model_path =
    (argc >= 3) ? argv[2] : "model/dlc/pose_landmark_lite_quantized.dlc";

  std::optional<VideoSource> source;
  source.emplace(std::string(argv[1]));

  if (!source->is_opened())
  {
    std::fprintf(stderr, "Failed to open video source\n");
    return 1;
  }

  PoseDSPOffload pose_model(pose_model_path);
  if (!pose_model.is_ready())
  {
    std::fprintf(
      stderr, "Failed to load pose model: %s\n", pose_model_path.c_str());
    return 1;
  }

  WindowingBuffer window_buffer;

  const std::string stgcn_model_path =
      (argc >= 4) ? argv[3] : "model/dlc/lite_stgcn_hmdb51.dlc";

  StgcnGpuOffload stgcn_model(stgcn_model_path);

  if (!stgcn_model.is_ready())
  {
    std::fprintf(
        stderr, "Failed to load STGCN model: %s\n", stgcn_model_path.c_str());
    return 1;
  }
  /* caps how far capture can run ahead of processing */
  constexpr size_t kQueueCapacity = 8;
  ThreadSafeQueue<Frame> frame_queue(kQueueCapacity);
  std::atomic<bool> stop_requested{false};

  std::thread capture_thread(
    capture_loop, std::ref(*source),
    std::ref(frame_queue), std::ref(stop_requested));

  while (true)
  {
    std::optional<Frame> frame = frame_queue.pop();

    if (!frame.has_value())
    {
      break;
    }

    process_frame(*frame, pose_model, window_buffer, stgcn_model);
  }

  capture_thread.join();
  return 0;
}
