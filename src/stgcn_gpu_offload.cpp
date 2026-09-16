/*
  stgcn_gpu_offload.cpp

  Implementation file for offloading the Lite-STGCN action classifier
  to the Adreno GPU via SNPE.
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "stgcn_gpu_offload.h"


namespace
{
  Snpe_Runtime_t choose_runtime
  (
    bool prefer_gpu
  )
  {
    if (prefer_gpu && Snpe_Util_IsRuntimeAvailable(SNPE_RUNTIME_GPU))
    {
      return SNPE_RUNTIME_GPU;
    }

    std::fprintf(
      stderr,
      "[stgcn_gpu_offload] GPU runtime unavailable, falling back to CPU\n");

    return SNPE_RUNTIME_CPU;
  }
}

StgcnGpuOffload::StgcnGpuOffload
(
  const std::string& model_path,
  bool                prefer_gpu
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
      "[stgcn_gpu_offload] Failed to open container: %s\n", model_path.c_str());

    return;
  }

  /* Same pattern as PoseDSPOffload -- only the runtime constant differs */
  runtime = choose_runtime(prefer_gpu);
  runtime_list_handle = Snpe_RuntimeList_Create();
  Snpe_RuntimeList_Add(runtime_list_handle, runtime);

  builder_handle_ = Snpe_SNPEBuilder_Create(container_handle_);
  Snpe_SNPEBuilder_SetRuntimeProcessorOrder(builder_handle_, runtime_list_handle);
  Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(builder_handle_, false);
  snpe_handle_ = Snpe_SNPEBuilder_Build(builder_handle_);

  Snpe_RuntimeList_Delete(runtime_list_handle);

  if (!snpe_handle_)
  {
    std::fprintf(stderr, "[stgcn_gpu_offload] Failed to build SNPE instance\n");
    return;
  }

  input_names_handle = Snpe_SNPE_GetInputTensorNames(snpe_handle_);

  if (!input_names_handle || (0 == Snpe_StringList_Size(input_names_handle)))
  {
    std::fprintf(stderr, "[stgcn_gpu_offload] Model reports no input tensors\n");
    snpe_handle_ = nullptr;
    return;
  }

  input_tensor_name_ = Snpe_StringList_At(input_names_handle, 0);
  input_shape_handle_ = Snpe_SNPE_GetInputDimensions(snpe_handle_, input_tensor_name_.c_str());
  Snpe_StringList_Delete(input_names_handle);
}

StgcnGpuOffload::~StgcnGpuOffload()
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

Snpe_ITensor_Handle_t StgcnGpuOffload::preprocess
(
  const std::vector<SkeletonFrame>& clip
)
{
  Snpe_ITensor_Handle_t tensor_handle = nullptr;
  void                  *tensor_data = nullptr;

  /* Sanity check(s) -- unlike PoseDSPOffload::preprocess, there is no
     resize/normalize step here: the caller is responsible for handing us
     an already-normalized, already-sized clip. This class is a pure
     "flatten to tensor, run, read scores back" wrapper. */
  if (clip.size() != static_cast<size_t>(kClipLen))
  {
    std::fprintf(
      stderr,
      "[stgcn_gpu_offload] Expected clip of %d frames, got %zu\n",
      kClipLen, clip.size());
    return nullptr;
  }

  tensor_handle = Snpe_Util_CreateITensor(input_shape_handle_);
  if (!tensor_handle)
  {
    std::fprintf(
      stderr, "[stgcn_gpu_offload] Failed to allocate input tensor\n");
    return nullptr;
  }

  tensor_data = Snpe_ITensor_GetData(tensor_handle);
  float* dst = static_cast<float*>(tensor_data);

  /* Model input layout is (N=1, M=kNumPerson, T=kClipLen, V=kNumCocoJoints, C=3)
     -- matches the (1, 2, 100, 17, 3) shape the ONNX export used, flattened
     in row-major order. This loop order must match that exactly, or the
     model will silently receive garbage (wrong axes swapped). */
  size_t idx = 0;
  for (int m = 0; m < kNumPerson; ++m)
  {
    for (int t = 0; t < kClipLen; ++t)
    {
      const auto& persons = clip[t].persons;
      for (int v = 0; v < kNumCocoJoints; ++v)
      {
        const Joint& joint = persons[m][v];
        dst[idx++] = joint.x;
        dst[idx++] = joint.y;
        dst[idx++] = joint.confidence;
      }
    }
  }

  return tensor_handle;
}

bool StgcnGpuOffload::postprocess
(
  Snpe_TensorMap_Handle_t output_map_handle,
  ClassificationResult&   out_result
)
{
  Snpe_ITensor_Handle_t score_tensor_handle = nullptr;

  Snpe_StringList_Handle_t output_names_handle =
    Snpe_TensorMap_GetTensorNames(output_map_handle);

  /* Sanity check(s) */
  if (!output_names_handle || (0 == Snpe_StringList_Size(output_names_handle)))
  {
    std::fprintf(stderr, "[stgcn_gpu_offload] No output tensors returned\n");
    return false;
  }

  /* The exported ONNX graph has a single output ('action_scores', shape
     (1, kNumClasses)) -- unlike PoseDSPOffload, there's no need to scan
     for a specific size among multiple outputs, but we still verify the
     size defensively rather than assuming index 0 is correct. */
  const size_t num_outputs = Snpe_StringList_Size(output_names_handle);
  for (size_t i = 0; i < num_outputs; ++i)
  {
    const char* name = Snpe_StringList_At(output_names_handle, i);
    Snpe_ITensor_Handle_t candidate = Snpe_TensorMap_GetTensor_Ref(output_map_handle, name);

    if (candidate && static_cast<size_t>(kNumClasses) == Snpe_ITensor_GetSize(candidate))
    {
      score_tensor_handle = candidate;
      break;
    }
  }

  if (nullptr == score_tensor_handle)
  {
    std::fprintf(
      stderr,
      "[stgcn_gpu_offload] Could not find the %d-element score tensor among %zu outputs\n",
      kNumClasses, num_outputs);

    Snpe_StringList_Delete(output_names_handle);
    return false;
  }

  const float* scores =
    static_cast<const float*>(Snpe_ITensor_GetData(score_tensor_handle));

  /* argmax over raw logits */
  int   best_idx   = 0;
  float best_score = scores[0];
  for (int i = 1; i < kNumClasses; ++i)
  {
    if (scores[i] > best_score)
    {
      best_score = scores[i];
      best_idx   = i;
    }
  }

  /* Softmax, computed only for reporting a human-readable confidence --
     does not affect the argmax decision above (softmax is monotonic). */
  float sum_exp = 0.0f;
  for (int i = 0; i < kNumClasses; ++i)
  {
    sum_exp += std::exp(scores[i] - best_score);  // shift by max for stability
  }
  const float confidence = 1.0f / sum_exp;  // exp(best - best) / sum_exp == 1 / sum_exp

  out_result.class_index = best_idx;
  out_result.raw_score    = best_score;
  out_result.confidence   = confidence;

  Snpe_StringList_Delete(output_names_handle);
  return true;
}

bool StgcnGpuOffload::classify
(
  const std::vector<SkeletonFrame>& clip,
  ClassificationResult&             out_result
)
{
  bool                     ret = false;
  Snpe_TensorMap_Handle_t  input_map_handle = nullptr;
  Snpe_ITensor_Handle_t    input_tensor_handle = nullptr;
  Snpe_TensorMap_Handle_t  output_map_handle = nullptr;

  if (!is_ready())
  {
    return ret;
  }

  input_tensor_handle = preprocess(clip);
  if (nullptr == input_tensor_handle)
  {
    return ret;
  }

  input_map_handle = Snpe_TensorMap_Create();
  Snpe_TensorMap_Add(
    input_map_handle, input_tensor_name_.c_str(), input_tensor_handle);

  output_map_handle = Snpe_TensorMap_Create();

  Snpe_SNPE_ExecuteITensors(snpe_handle_, input_map_handle, output_map_handle);

  ret = postprocess(output_map_handle, out_result);

  Snpe_TensorMap_Delete(input_map_handle);
  Snpe_TensorMap_Delete(output_map_handle);
  Snpe_ITensor_Delete(input_tensor_handle);

  return ret;
}