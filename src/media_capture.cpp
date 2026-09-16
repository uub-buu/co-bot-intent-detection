/*
  media_capture.cpp

  We originally intended to be implemented on a real-time camera feed, but later
  scaled down to running on pre-recorded video files only.

  This reads frames on their own thread, and pushes them into a thread-safe queue.
  The consumer side of that queue is where the FastRPC offload to the Hexagon DSP
  (pose estimation) plugs in (stubbed as of now).
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

#include <opencv2/opencv.hpp>


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
 * Bounded producer/consumer queue between the capture thread and DSP offload.
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
 * Runs in its own thread, pushing frames into frame_queue
 * until the source runs out of frames or stop_requested is set.
 */
void capture_loop
(
  VideoSource&            source,
  ThreadSafeQueue<Frame>& frame_queue,
  std::atomic<bool>&      stop_requested
)
{
  Frame        frame;
  int          frame_index = 0;
  const auto   start_time = std::chrono::steady_clock::now();

  while (!stop_requested.load())
  {
    auto maybe_frame = source.read_rgb_frame();
    if (!maybe_frame.has_value())
    {
      break;
    }

    const auto now = std::chrono::steady_clock::now();
    const double timestamp_ms =
      std::chrono::duration<double, std::milli>(now - start_time).count();

    frame.image = std::move(*maybe_frame);
    frame.index = frame_index++;
    frame.timestamp_ms = timestamp_ms;

    frame_queue.push(std::move(frame));
  }

  frame_queue.stop();
  std::printf("Capture loop done. Total frames captured: %d\n", frame_index);
}

/*******************************************************************************
 * Functions
 ******************************************************************************/

/*
 * TODO: FastRPC offload (stub as of now)
 */
void offload_to_pose_estimation
(
  const Frame& frame
)
{
  std::printf(
    "[stub] Frame %d (t = %.1fms, %dx%d) Ready for DSP pose offload\n",
    frame.index, frame.timestamp_ms, frame.image.cols, frame.image.rows);
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
      stderr, "Usage: %s <video_path>\n", argv[0]);
    return 1;
  }

  std::optional<VideoSource> source;
  source.emplace(std::string(argv[1]));

  if (!source->is_opened())
  {
    std::fprintf(stderr, "Failed to open video source\n");
    return 1;
  }

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

    offload_to_pose_estimation(*frame);
  }

  capture_thread.join();
  return 0;
}
