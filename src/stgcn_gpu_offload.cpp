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

#include "windowing_buffer.h"
#include "stgcn_gpu_offload.h"


namespace
{
  bool is_gpu_available()
  {
    if (Snpe_Util_IsRuntimeAvailable(SNPE_RUNTIME_GPU))
    {
      return true;
    }

    std::fprintf(
      stderr,
      "[stgcn_gpu_offload] GPU runtime unavailable. Check that the \
Adreno GPU driver/runtime is present.\n");

    return false;
  }
}

StgcnGpuOffload::StgcnGpuOffload
(
  const std::string& model_path
)
{
  Snpe_StringList_Handle_t  input_names_handle = nullptr;
  Snpe_RuntimeList_Handle_t runtime_list_handle = nullptr;
  
  /* Read the .dlc file*/
  container_handle_t = Snpe_DlContainer_Open(model_path.c_str());

  /* Sanity check */
  if (!container_handle_t)
  {
    std::fprintf(
      stderr,
      "[stgcn_gpu_offload] Failed to open container: %s\n", model_path.c_str());

    return;
  }
  /* Sanity check */
  if (!is_gpu_available())
  {
    Snpe_DlContainer_Delete(container_handle_t);
    container_handle_t = nullptr;
    return;
  }
  /* create snpe runtime list with just GPU on it. This will allow builder to process*/
  runtime_list_handle = Snpe_RuntimeList_Create();
  Snpe_RuntimeList_Add(runtime_list_handle, SNPE_RUNTIME_GPU);

  builder_handle_t = Snpe_SNPEBuilder_Create(container_handle_t);
  Snpe_SNPEBuilder_SetRuntimeProcessorOrder(builder_handle_t, runtime_list_handle);
  Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(builder_handle_t, false);
  /* This command will build our container into a runnable model for the GPU*/
  snpe_handle_t = Snpe_SNPEBuilder_Build(builder_handle_t);
  // we do not need the list after builder is configured
  Snpe_RuntimeList_Delete(runtime_list_handle);
  
  /* Sanity check: Did containter build? will return nullptr if it did. */
  if (!snpe_handle_t)
  {
    std::fprintf(stderr, "[stgcn_gpu_offload] Failed to build SNPE instance!\n");
    return;
  }

  input_names_handle = Snpe_SNPE_GetInputTensorNames(snpe_handle_t);

  if (!input_names_handle || (0 == Snpe_StringList_Size(input_names_handle)))
  {
    std::fprintf(stderr, "[stgcn_gpu_offload] Model reports no input tensors\n");
    snpe_handle_t = nullptr;
    return;
  }

  input_tensor_name = Snpe_StringList_At(input_names_handle, 0);
  /* use the input name to get input shape from SNPE objectt*/
  input_shape_handle_t = Snpe_SNPE_GetInputDimensions(snpe_handle_t, input_tensor_name.c_str());
  Snpe_StringList_Delete(input_names_handle);
}

StgcnGpuOffload::~StgcnGpuOffload()
{
  if (input_shape_handle_t)
  {
    Snpe_TensorShape_Delete(input_shape_handle_t);
  }

  if (snpe_handle_t)
  {
    Snpe_SNPE_Delete(snpe_handle_t);
  }

  if (builder_handle_t)
  {
    Snpe_SNPEBuilder_Delete(builder_handle_t);
  }

  if (container_handle_t)
  {
    Snpe_DlContainer_Delete(container_handle_t);
  }
}

Snpe_ITensor_Handle_t StgcnGpuOffload::preprocess
(
  const std::vector<float>& window_tensor
)
{
  Snpe_ITensor_Handle_t tensor_handle = nullptr;
  void                  *tensor_data = nullptr;

  /* Sanity check: is clip 100 frames long*/
    if (window_tensor.size() != static_cast<size_t>(kWindowElementCount))
  {
    std::fprintf(
      stderr,
      "[stgcn_gpu_offload] Expected window tensor of %d elements, got %zu\n",
      kWindowElementCount, window_tensor.size());
    return nullptr;
  }


  tensor_handle = Snpe_Util_CreateITensor(input_shape_handle_t);
  /* Sanity Check*/
  if (!tensor_handle)
  {
    std::fprintf(
      stderr, "[stgcn_gpu_offload] Failed to allocate input tensor\n");
    return nullptr;
  }

  /* this will be our 1D array for VRAM for the GPU*/
  tensor_data = Snpe_ITensor_GetData(tensor_handle);
  std::memcpy(tensor_data, window_tensor.data(), window_tensor.size() * sizeof(float));

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

  /* Sanity check */
  if (!output_names_handle || (0 == Snpe_StringList_Size(output_names_handle)))
  {
    std::fprintf(stderr, "[stgcn_gpu_offload] No output tensors returned\n");
    return false;
  }

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

  if (!score_tensor_handle)
  {
    std::fprintf(
      stderr,
      "[stgcn_gpu_offload] Could not find the %d-element score tensor among %zu outputs\n",
      kNumClasses, num_outputs);

    Snpe_StringList_Delete(output_names_handle);
    return false;
  }

  const float* scores = static_cast<const float*>(Snpe_ITensor_GetData(score_tensor_handle));

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

  /* ssoftmax, computed only for human-readable confidence */
  float sum_exp = 0.0f;
  for (int i = 0; i < kNumClasses; ++i)
  {
    sum_exp += std::exp(scores[i] - best_score);
  }

  const float confidence = 1.0f / sum_exp;

  out_result.class_index = best_idx;
  out_result.raw_score    = best_score;
  out_result.confidence   = confidence;

  Snpe_StringList_Delete(output_names_handle);
  return true;
}

bool StgcnGpuOffload::classify
(
  const std::vector<float>& window_tensor,
  ClassificationResult&     out_result

)
{
  bool                     ret = false;
  Snpe_TensorMap_Handle_t  input_map_handle = nullptr;
  Snpe_TensorMap_Handle_t  output_map_handle = nullptr;
  Snpe_ITensor_Handle_t    input_tensor_handle = nullptr;

  if (!is_ready())
  {
    return ret;
  }

  input_tensor_handle = preprocess(window_tensor);
  if (!input_tensor_handle)
  {
    return ret;
  }

  input_map_handle = Snpe_TensorMap_Create();
  Snpe_TensorMap_Add(input_map_handle, input_tensor_name.c_str(), input_tensor_handle);

  output_map_handle = Snpe_TensorMap_Create();

  Snpe_SNPE_ExecuteITensors(snpe_handle_t, input_map_handle, output_map_handle);

  ret = postprocess(output_map_handle, out_result);

  Snpe_TensorMap_Delete(input_map_handle);
  Snpe_TensorMap_Delete(output_map_handle);
  Snpe_ITensor_Delete(input_tensor_handle);

  return ret;
}

