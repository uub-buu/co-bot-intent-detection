#ifndef STGCN_GPU_OFFLOAD_H
#define STGCN_GPU_OFFLOAD_H

/*
  stgcn_gpu_offload.h

  SNPE offload for the Lite-STGCN classifier. Uses plain SNPEBuilder,
  not PSNPE; PSNPE's GPU path fails silently on this board with no
  error, while plain SNPEBuilder+GPU validates ~94% of ops.

  Builds GPU first. If it fails (currently ReduceSum_Einsum_8_1, part
  of the graph conv's Einsum decomposition, unsupported on GPU), retries
  on CPU. Loses GPU speed on the ops that do validate, gains a build
  that succeeds.

  Assumes input is already COCO-17 ordered. Caller must remap
  MediaPipe's 33 landmarks before calling this.
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "SNPE/SNPE.h"
#include "SNPE/SNPEUtil.h"
#include "SNPE/SNPEBuilder.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/ITensor.h"
#include "DlSystem/StringList.h"
#include "DlSystem/TensorMap.h"
#include "DlSystem/TensorShape.h"
#include "DlContainer/DlContainer.h"


/*******************************************************************************
 * Data
 ******************************************************************************/

constexpr int kNumCocoJoints = 17;   /* graph_cfg layout='coco' */
constexpr int kClipLen       = 100;  /* clip_len in UniformSample */
constexpr int kNumPerson     = 2;    /* FormatGCNInput(num_person=2) */
constexpr int kNumClasses    = 51;   /* HMDB51 */


/*******************************************************************************
 * Type definitions
 ******************************************************************************/

/*
 * ClassificationResult
 */
struct ClassificationResult
{
  int   class_index = -1;   /* HMDB51 class index */
  float raw_score   = 0.0f; /* raw logit before softmax */
  float confidence  = 0.0f; /* softmax probability of class_index */
};

class StgcnGpuOffload
{
  public:
    explicit StgcnGpuOffload
    (
      const std::string& model_path
    );

    ~StgcnGpuOffload();

    StgcnGpuOffload(const StgcnGpuOffload&) = delete;
    StgcnGpuOffload& operator=(const StgcnGpuOffload&) = delete;

    bool is_ready() const
    {
      return snpe_handle_t != nullptr;
    }

    Snpe_Runtime_t active_runtime() const
    {
      return active_runtime_;
    }

    /* classify: Returns false if the model isn't ready or the clip size is wrong. */
    bool classify
    (
      const std::vector<float>& window_tensor,
      ClassificationResult&     out_result
    );

  private:
    /*
     * try_build
     *
     * Builds snpe_handle_t for the given runtime. Cleans up partial state on
     * failure so a retry starts clean (faced GPU issue otherwise!).
     */
    bool try_build
    (
      Snpe_Runtime_t runtime
    );

    Snpe_ITensor_Handle_t preprocess(const std::vector<float>& window_tensor);
    bool postprocess(Snpe_TensorMap_Handle_t output_map_handle, ClassificationResult& out_result);

    Snpe_DlContainer_Handle_t container_handle_t    = nullptr;
    Snpe_SNPEBuilder_Handle_t builder_handle_t     = nullptr;
    Snpe_SNPE_Handle_t        snpe_handle_t       = nullptr;
    Snpe_TensorShape_Handle_t input_shape_handle_t  = nullptr;
    std::string               input_tensor_name;
    Snpe_Runtime_t            active_runtime_ = SNPE_RUNTIME_CPU;
};

#endif /* STGCN_GPU_OFFLOAD_H */
