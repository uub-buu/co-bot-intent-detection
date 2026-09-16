/*
  pose_dsp_offload.cpp

  Implementation file for offloading to DSP
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include <cmath>
#include <cstdio>
#include <cstring>

#include "pose_dsp_offload.h"


namespace
{
  Snpe_Runtime_t choose_runtime
  (
    bool prefer_dsp
  )
  {
    if (prefer_dsp && Snpe_Util_IsRuntimeAvailable(SNPE_RUNTIME_DSP))
    {
      return SNPE_RUNTIME_DSP;
    }

    std::fprintf(
      stderr,
      "[pose_dsp_offload] DSP runtime unavailable, falling back to CPU\n");

    return SNPE_RUNTIME_CPU;
  }
}

PoseDSPOffload::PoseDSPOffload
(
  const std::string& model_path,
  bool               prefer_dsp
)
{
  Snpe_Runtime_t            runtime = SNPE_RUNTIME_CPU;
  Snpe_StringList_Handle_t  input_names_handle = nullptr;
  Snpe_RuntimeList_Handle_t runtime_list_handle = nullptr;

  container_handle_ = Snpe_DlContainer_Open(model_path.c_str());

  /* Sanity check(s) */
  if (!container_handle_)
  {
    std::fprintf(
      stderr,
      "[pose_dsp_offload] Failed to open container: %s\n", model_path.c_str());

    return;
  }

  runtime = choose_runtime(prefer_dsp);
  runtime_list_handle = Snpe_RuntimeList_Create();
  Snpe_RuntimeList_Add(runtime_list_handle, runtime);

  builder_handle_ = Snpe_SNPEBuilder_Create(container_handle_);
  Snpe_SNPEBuilder_SetRuntimeProcessorOrder(builder_handle_, runtime_list_handle);
  Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(builder_handle_, false);
  snpe_handle_ = Snpe_SNPEBuilder_Build(builder_handle_);

  Snpe_RuntimeList_Delete(runtime_list_handle);

  if (!snpe_handle_)
  {
    std::fprintf(stderr, "[pose_dsp_offload] Failed to build SNPE instance\n");
    return;
  }

  input_names_handle = Snpe_SNPE_GetInputTensorNames(snpe_handle_);

  if (!input_names_handle || (0 == Snpe_StringList_Size(input_names_handle)))
  {
    std::fprintf(stderr, "[pose_dsp_offload] Model reports no input tensors\n");
    snpe_handle_ = nullptr;
    return;
  }

  input_tensor_name_ = Snpe_StringList_At(input_names_handle, 0);
  input_shape_handle_ = Snpe_SNPE_GetInputDimensions(snpe_handle_, input_tensor_name_.c_str());
  Snpe_StringList_Delete(input_names_handle);
}

PoseDSPOffload::~PoseDSPOffload()
{
  if (input_shape_handle_)
  {
    Snpe_TensorShape_Delete(input_shape_handle_);
  }

  if (snpe_handle_)
  {
    Snpe_SNPE_Delete(snpe_handle_);
  }

  if (builder_handle_)
  {
    Snpe_SNPEBuilder_Delete(builder_handle_);
  }

  if (container_handle_)
  {
    Snpe_DlContainer_Delete(container_handle_);
  }
}

Snpe_ITensor_Handle_t PoseDSPOffload::preprocess
(
  const cv::Mat& rgb_frame
)
{
  cv::Mat                resized, float_frame;
  void                  *tensor_data = nullptr;
  Snpe_ITensor_Handle_t  tensor_handle = nullptr;

  /* Dims are NHWC */
  const size_t rank = Snpe_TensorShape_Rank(input_shape_handle_);
  if (rank < 3)
  {
    std::fprintf(
      stderr, "[pose_dsp_offload] Unexpected input rank: %zu\n", rank);
    return nullptr;
  }

  const size_t dims_offset = (rank == 4) ? 1 : 0;

  const int target_height =
    static_cast<int>(Snpe_TensorShape_At(input_shape_handle_, dims_offset));

  const int target_width =
    static_cast<int>(Snpe_TensorShape_At(input_shape_handle_, dims_offset + 1));

  cv::resize(rgb_frame, resized, cv::Size(target_width, target_height));
  /* TODO: Check with model */
  resized.convertTo(float_frame, CV_32FC3, 1.0 / 255.0);

  tensor_handle = Snpe_Util_CreateITensor(input_shape_handle_);
  if (!tensor_handle)
  {
    std::fprintf(
      stderr, "[pose_dsp_offload] Failed to allocate input tensor\n");
    return nullptr;
  }

  tensor_data = Snpe_ITensor_GetData(tensor_handle);

  const size_t tensor_bytes =
    Snpe_ITensor_GetSize(tensor_handle) * sizeof(float);

  std::memcpy(tensor_data, float_frame.ptr<float>(0), tensor_bytes);

  return tensor_handle;
}

namespace {
  inline float sigmoid
  (
    float x
  )
  {
    return 1.0f / (1.0f + std::exp(-x));
  }
}

bool PoseDSPOffload::postprocess
(
  Snpe_TensorMap_Handle_t output_map_handle,
  JointFrame&             out_joints
)
{
  Snpe_ITensor_Handle_t candidate = nullptr;
  Snpe_ITensor_Handle_t landmark_tensor_handle = nullptr;

  Snpe_StringList_Handle_t output_names_handle =
    Snpe_TensorMap_GetTensorNames(output_map_handle);

  /* Sanity check(s) */
  if (!output_names_handle || (0 == Snpe_StringList_Size(output_names_handle)))
  {
    std::fprintf(stderr, "[pose_dsp_offload] No output tensors returned\n");
    return false;
  }

  /* TODO: Might have to update after Brian's model. Below is with BlazePose */
  constexpr size_t kExpectedLandmarkTensorSize =
    kNumLandmarksRaw * kLandmarkValueStride;

  const size_t num_outputs = Snpe_StringList_Size(output_names_handle);
  for (size_t i = 0; i < num_outputs; ++i)
  {
    const char* name = Snpe_StringList_At(output_names_handle, i);
    candidate = Snpe_TensorMap_GetTensor_Ref(output_map_handle, name);

    if (candidate &&
        kExpectedLandmarkTensorSize == Snpe_ITensor_GetSize(candidate))
    {
      landmark_tensor_handle = candidate;
      break;
    }
  }

  if (nullptr == landmark_tensor_handle)
  {
    std::fprintf(
      stderr,
      "[pose_dsp_offload] Could not find the %zu-element landmark tensor \
among %zu outputs\n",
      kExpectedLandmarkTensorSize, num_outputs);

    Snpe_StringList_Delete(output_names_handle);
    return false;
  }

  const float* data =
    static_cast<const float*>(Snpe_ITensor_GetData(landmark_tensor_handle));

  for (int joint = 0; joint < kNumLandmarksBody; ++joint)
  {
    const float* lm = data + joint * kLandmarkValueStride;
    Landmark& out = out_joints[joint];
    out.x = lm[0];
    out.y = lm[1];
    out.z = lm[2];
    out.visibility = sigmoid(lm[3]);
    out.presence = sigmoid(lm[4]);
  }

  Snpe_StringList_Delete(output_names_handle);
  return true;
}

bool PoseDSPOffload::estimate
(
  const cv::Mat& rgb_frame,
  JointFrame&     out_joints
)
{
  bool                    ret = false;
  Snpe_TensorMap_Handle_t input_map_handle = nullptr;
  Snpe_ITensor_Handle_t   input_tensor_handle = nullptr;
  Snpe_TensorMap_Handle_t output_map_handle = nullptr;

  if (!is_ready())
  {
    return ret;
  }

  input_tensor_handle = preprocess(rgb_frame);
  if (nullptr == input_tensor_handle)
  {
    return ret;
  }

  input_map_handle = Snpe_TensorMap_Create();
  Snpe_TensorMap_Add(
    input_map_handle, input_tensor_name_.c_str(), input_tensor_handle);

  output_map_handle = Snpe_TensorMap_Create();

  Snpe_SNPE_ExecuteITensors(snpe_handle_, input_map_handle, output_map_handle);

  ret = postprocess(output_map_handle, out_joints);

  Snpe_TensorMap_Delete(input_map_handle);
  Snpe_TensorMap_Delete(output_map_handle);
  Snpe_ITensor_Delete(input_tensor_handle);

  return ret;
}
