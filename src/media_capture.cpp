/*
  media_capture.cpp

  We originally intended to be implemented on a real-time camera feed, but
  later scaled down to running on pre-recorded video files only.

  This reads frames on their own thread, and pushes them into a thread-safe
  queue. The consumer side of that queue runs the real pipeline: DSP pose
  estimation (PoseDSPOffload), MediaPipe -> COCO-17 joint remap, the CPU
  windowing buffer, and GPU STGCN offload (StgcnGpuOffload).

  GPU inference and video decode are serialized against each other via
  gpu_video_mutex: running both concurrently on this board's hardware
  causes an unrecoverable lockup (confirmed via isolated testing -- GPU
  inference alone and video decode alone both work fine; only running
  them at the same time crashes the board). The same mutex also covers
  DSP pose inference: DSP execution was returning frozen/stale output
  regardless of input, suspected to be the same class of hardware
  contention with concurrent video decode (not yet isolated/confirmed the
  way the GPU case was -- this is the fix being tested). All three of
  video decode, DSP inference, and GPU inference are mutually exclusive
  with each other under this one mutex.
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
    /*
     * gpu_video_mutex must be held for the duration of every decode call
     * -- see the file-level comment for why video decode and GPU
     * inference cannot run concurrently on this board.
     */
    explicit VideoSource
    (
      const std::string& path,
      std::mutex&         gpu_video_mutex
    )
      : gpu_video_mutex_(gpu_video_mutex)
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
      cv::Mat                     bgr_frame, rgb_frame;
      std::lock_guard<std::mutex> lock(gpu_video_mutex_);

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
    std::mutex&      gpu_video_mutex_;
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
 * MediaPipe -> COCO-17 remap, the CPU windowing buffer, and GPU STGCN
 * offload once a window fills. gpu_video_mutex is held only around the
 * classify() call -- see the file-level comment for why.
 */
void process_frame
(
  const Frame&     frame,
  PoseDSPOffload&  pose_model,
  WindowingBuffer& window_buffer,
  StgcnGpuOffload& stgcn_model,
  std::mutex&      gpu_video_mutex
)
{
  JointFrame mediapipe_joints;
  const int  frame_width = frame.image.cols;
  const int  frame_height = frame.image.rows;
  bool       pose_ok;

  /*
   * gpu_video_mutex also covers DSP pose inference here, not just the
   * STGCN GPU call below -- DSP and video decode running concurrently on
   * this board is the suspected cause of frozen pose output (this was
   * never actually isolated/tested the way GPU+video-decode was; only
   * added after the freeze was traced back to DSP execution).
   */
  {
    std::lock_guard<std::mutex> lock(gpu_video_mutex);
    pose_ok = pose_model.estimate(frame.image, mediapipe_joints);
  }

  if (!pose_ok)
  {
    std::fprintf(
      stderr, "Pose estimation failed on frame %d\n", frame.index);
    return;
  }

  #ifdef DEBUG
  /*
   * Print a few raw landmark values every 15 frames, to
   * check whether pose output is suspiciously similar/near-constant
   * across different videos (would indicate the quantized pose model is
   * losing precision) versus genuinely varying with the person's actual
   * motion. Indices: 0=nose, 11=left_shoulder, 15=left_wrist,
   * 23=left_hip, 27=left_ankle.
   */
  if (0 == (frame.index % 15))
  {
    const cv::Scalar image_checksum = cv::sum(frame.image);

    std::printf(
      "Frame %d input image checksum (B, G, R sums): %.1f, %.1f, %.1f\n",
      frame.index, image_checksum[0], image_checksum[1], image_checksum[2]);
    std::printf("Frame %d landmarks (x, y, visibility):\n", frame.index);
    std::printf(
      "  nose:          %.4f, %.4f, %.4f\n",
      mediapipe_joints[0].x, mediapipe_joints[0].y, mediapipe_joints[0].visibility);
    std::printf(
      "  left_shoulder: %.4f, %.4f, %.4f\n",
      mediapipe_joints[11].x, mediapipe_joints[11].y, mediapipe_joints[11].visibility);
    std::printf(
      "  left_wrist:    %.4f, %.4f, %.4f\n",
      mediapipe_joints[15].x, mediapipe_joints[15].y, mediapipe_joints[15].visibility);
    std::printf(
      "  left_hip:      %.4f, %.4f, %.4f\n",
      mediapipe_joints[23].x, mediapipe_joints[23].y, mediapipe_joints[23].visibility);
    std::printf(
      "  left_ankle:    %.4f, %.4f, %.4f\n",
      mediapipe_joints[27].x, mediapipe_joints[27].y, mediapipe_joints[27].visibility);
  }
  #endif

  const CocoFrame raw_coco_frame = remap_mediapipe_to_coco17(mediapipe_joints);

  const CocoFrame norm_frame =
    normalize_coco_frame(raw_coco_frame, frame_width, frame_height);

  const std::optional<std::vector<float>> window =
    window_buffer.add_frame(norm_frame);

  if (window.has_value())
  {
    ClassificationResult result;
    bool                 classified;

    std::printf(
      "Frame %d: window ready (%zu floats) for GPU STGCN offload\n",
      frame.index, window->size());

    {
      std::lock_guard<std::mutex> lock(gpu_video_mutex);
      classified = stgcn_model.classify(*window, result);
    }

    if (classified)
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
    (argc >= 3) ? argv[2] : "model/dlc/pose_landmark_lite.dlc";

  /*
   * Serializes GPU inference against video decode -- concurrent use
   * crashes this board's hardware (see file-level comment).
   */
  std::mutex gpu_video_mutex;

  std::optional<VideoSource> source;
  source.emplace(std::string(argv[1]), gpu_video_mutex);

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
    (argc >= 4) ? argv[3] : "model/dlc/lite_stgcn_hmdb51_matmul.dlc";

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

    process_frame(*frame, pose_model, window_buffer, stgcn_model, gpu_video_mutex);
  }

  capture_thread.join();
  return 0;
}
