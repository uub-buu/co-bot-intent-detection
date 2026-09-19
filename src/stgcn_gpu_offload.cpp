/*
  stgcn_gpu_offload.cpp

  Implementation file for offloading the Lite-STGCN action classifier via
  SNPE. Tries GPU first, retries on CPU if the GPU build fails.
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "stgcn_gpu_offload.h"
#include "windowing_buffer.h"


bool StgcnGpuOffload::try_build
(
  Snpe_Runtime_t runtime
)
{
  Snpe_RuntimeList_Handle_t runtime_list_handle = nullptr;
  Snpe_StringList_Handle_t  input_names_handle = nullptr;

  runtime_list_handle = Snpe_RuntimeList_Create();
  Snpe_RuntimeList_Add(runtime_list_handle, runtime);

  builder_handle_t = Snpe_SNPEBuilder_Create(container_handle_t);
  Snpe_SNPEBuilder_SetRuntimeProcessorOrder(builder_handle_t, runtime_list_handle);
  Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(builder_handle_t, false);
  Snpe_SNPEBuilder_SetInitCacheMode(builder_handle_t, 0);
  Snpe_SNPEBuilder_SetPerformanceProfile(
    builder_handle_t, SNPE_PERFORMANCE_PROFILE_DEFAULT);
  Snpe_SNPEBuilder_SetDebugMode(builder_handle_t, 1);
  snpe_handle_t = Snpe_SNPEBuilder_Build(builder_handle_t);

  Snpe_RuntimeList_Delete(runtime_list_handle);

  /* Sanity check: Did the network build for this runtime? */
  if (!snpe_handle_t)
  {
    Snpe_SNPEBuilder_Delete(builder_handle_t);
    builder_handle_t = nullptr;
    return false;
  }

  active_runtime_ = runtime;

  input_names_handle = Snpe_SNPE_GetInputTensorNames(snpe_handle_t);

  if (!input_names_handle || (0 == Snpe_StringList_Size(input_names_handle)))
  {
    std::fprintf(stderr, "[stgcn_gpu_offload] Model reports no input tensors\n");
    Snpe_SNPE_Delete(snpe_handle_t);
    snpe_handle_t = nullptr;
    Snpe_SNPEBuilder_Delete(builder_handle_t);
    builder_handle_t = nullptr;
    return false;
  }

  input_tensor_name = Snpe_StringList_At(input_names_handle, 0);
  /* Get input shape using the input name */
  input_shape_handle_t = Snpe_SNPE_GetInputDimensions(snpe_handle_t, input_tensor_name.c_str());
  Snpe_StringList_Delete(input_names_handle);

  return true;
}

StgcnGpuOffload::StgcnGpuOffload
(
  const std::string& model_path
)
{
  #ifdef DEBUG_SNPE
  Snpe_Util_InitializeLoggingPath(
    SNPE_LOG_LEVEL_VERBOSE,
    "/home/ubuntu/cobid/co-bot-intent-detection/snpe_logs_gpu");
  #endif

  /* Read the .dlc file */
  container_handle_t = Snpe_DlContainer_Open(model_path.c_str());

  /* Sanity check */
  if (!container_handle_t)
  {
    std::fprintf(
      stderr,
      "[stgcn_gpu_offload] Failed to open container: %s\n", model_path.c_str());

    return;
  }

  /*
   * Tries GPU first, retries on CPU if the build fails; see file
   * header for why PSNPE isn't used instead.
   */
  if (!try_build(SNPE_RUNTIME_GPU))
  {
    std::fprintf(
      stderr,
      "[stgcn_gpu_offload] GPU build failed, retrying on CPU\n");

    if (!try_build(SNPE_RUNTIME_CPU))
    {
      std::fprintf(stderr, "[stgcn_gpu_offload] Failed to build SNPE instance on CPU either!\n");
      Snpe_DlContainer_Delete(container_handle_t);
      container_handle_t = nullptr;
      return;
    }
  }
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
  void                  *tensor_data = nullptr;
  Snpe_ITensor_Handle_t  tensor_handle = nullptr;

  tensor_handle = Snpe_Util_CreateITensor(input_shape_handle_t);
  if (!tensor_handle)
  {
    std::fprintf(
      stderr, "[stgcn_gpu_offload] Failed to allocate input tensor\n");
    return nullptr;
  }

  tensor_data = Snpe_ITensor_GetData(tensor_handle);

  const size_t tensor_bytes =
    Snpe_ITensor_GetSize(tensor_handle) * sizeof(float);

  std::memcpy(tensor_data, window_tensor.data(), tensor_bytes);

  return tensor_handle;
}

bool StgcnGpuOffload::postprocess
(
  Snpe_TensorMap_Handle_t output_map_handle,
  ClassificationResult&   out_result
)
{
  Snpe_ITensor_Handle_t candidate = nullptr;

  Snpe_StringList_Handle_t output_names_handle =
    Snpe_TensorMap_GetTensorNames(output_map_handle);

  /* Sanity check(s) */
  if (!output_names_handle || (0 == Snpe_StringList_Size(output_names_handle)))
  {
    std::fprintf(stderr, "[stgcn_gpu_offload] No output tensors returned\n");
    return false;
  }

  const char* output_name = Snpe_StringList_At(output_names_handle, 0);
  candidate = Snpe_TensorMap_GetTensor_Ref(output_map_handle, output_name);

  if (!candidate || (Snpe_ITensor_GetSize(candidate) != static_cast<size_t>(kNumClasses)))
  {
    std::fprintf(
      stderr, "[stgcn_gpu_offload] Unexpected output size: %zu (expected %d)\n",
      candidate ? Snpe_ITensor_GetSize(candidate) : 0, kNumClasses);

    Snpe_StringList_Delete(output_names_handle);
    return false;
  }

  const float* data = static_cast<const float*>(Snpe_ITensor_GetData(candidate));

  float best_score = data[0];
  int   best_idx = 0;
  float sum_exp = 0.f;

  for (int i = 1; i < kNumClasses; ++i)
  {
    if (data[i] > best_score)
    {
      best_score = data[i];
      best_idx = i;
    }
  }

  /*
   * Softmax over the raw logits, shifted by the max for numeric stability.
   * Only the argmax class's probability is used.
   */
  for (int i = 0; i < kNumClasses; ++i)
  {
    sum_exp += std::exp(data[i] - best_score);
  }

  out_result.class_index = best_idx;
  out_result.raw_score = best_score;
  out_result.confidence = 1.0f / sum_exp;

  Snpe_StringList_Delete(output_names_handle);
  return true;
}

bool StgcnGpuOffload::classify
(
  const std::vector<float>& window_tensor,
  ClassificationResult&     out_result
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

  /* Sanity check(s) */
  if (window_tensor.size() != static_cast<size_t>(kWindowElementCount))
  {
    std::fprintf(
      stderr,
      "[stgcn_gpu_offload] Unexpected window size: %zu (expected %d)\n",
      window_tensor.size(), kWindowElementCount);
    return false;
  }

  #ifdef DEBUG
  /* Dumps calibration input for the first few calls */
  {
    static int dump_count = 0;
    if (dump_count < 10)
    {
      /* TODO: Hard-coded for uub-buu's setup */
      std::string path = "/home/buu/workspace/WES237B/co-bot-intent-detection/calib_" +
                          std::to_string(dump_count) + ".raw";
      FILE* f = std::fopen(path.c_str(), "wb");
      if (f)
      {
        std::fwrite(window_tensor.data(), sizeof(float), window_tensor.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "[stgcn_gpu_offload] Dumped %s\n", path.c_str());
      }
      ++dump_count;
    }
  }
  #endif

  input_tensor_handle = preprocess(window_tensor);
  if (nullptr == input_tensor_handle)
  {
    return ret;
  }

  input_map_handle = Snpe_TensorMap_Create();
  Snpe_TensorMap_Add(
    input_map_handle, input_tensor_name.c_str(), input_tensor_handle);

  output_map_handle = Snpe_TensorMap_Create();

  Snpe_SNPE_ExecuteITensors(snpe_handle_t, input_map_handle, output_map_handle);

  ret = postprocess(output_map_handle, out_result);

  Snpe_TensorMap_Delete(input_map_handle);
  Snpe_TensorMap_Delete(output_map_handle);
  Snpe_ITensor_Delete(input_tensor_handle);

  return ret;
}
