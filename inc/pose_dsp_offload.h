#ifndef POSE_DSP_OFFLOAD_H
#define POSE_DSP_OFFLOAD_H

/*
 * Using the Snpe_* APIs from the Qualcomm AI Runtime SDK
 *
 * Below numbers were with BlazePose's "lite" model - placeholder for now,
 * needs to be updated once we swith to the HMDB51 trained model.
 */

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include <array>
#include <string>

#include <opencv2/opencv.hpp>

#include "SNPE/SNPE.h"
#include "SNPE/SNPEBuilder.h"
#include "SNPE/SNPEUtil.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/ITensor.h"
#include "DlSystem/StringList.h"
#include "DlSystem/TensorMap.h"
#include "DlSystem/TensorShape.h"
#include "DlContainer/DlContainer.h"


/*******************************************************************************
 * Data
 ******************************************************************************/

/* Total landmarks in the Identity tensor */
constexpr int kNumLandmarksRaw = 39;

/* BlazePose/MediaPipe body landmarks (first 33 of the 39) */
constexpr int kNumLandmarksBody = 33;

/* x, y, z, visibility, presence per landmark */
constexpr int kLandmarkValueStride = 5;


/*******************************************************************************
 * Type definitions
 ******************************************************************************/

/* Based on BlazePose sample tflite */
struct Landmark
{
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
  float visibility = 0.f;
  float presence = 0.f;
};

using JointFrame = std::array<Landmark, kNumLandmarksBody>;

class PoseDSPOffload
{
  public:
    explicit PoseDSPOffload
    (
      const std::string& model_path,
      bool               prefer_dsp = true);

    ~PoseDSPOffload();

    PoseDSPOffload(const PoseDSPOffload&) = delete;
    PoseDSPOffload& operator=(const PoseDSPOffload&) = delete;

    bool is_ready() const
    {
      return snpe_handle_ != nullptr;
    }

    bool estimate
    (
      const cv::Mat& rgb_frame,
      JointFrame&    out_joints
    );

  private:
    Snpe_ITensor_Handle_t preprocess(const cv::Mat& rgb_frame);
    bool postprocess(Snpe_TensorMap_Handle_t output_map_handle, JointFrame& out_joints);

    Snpe_DlContainer_Handle_t container_handle_ = nullptr;
    Snpe_SNPEBuilder_Handle_t builder_handle_ = nullptr;
    Snpe_SNPE_Handle_t snpe_handle_ = nullptr;
    Snpe_TensorShape_Handle_t input_shape_handle_ = nullptr;
    std::string input_tensor_name_;
};

#endif /* POSE_DSP_OFFLOAD_H */
