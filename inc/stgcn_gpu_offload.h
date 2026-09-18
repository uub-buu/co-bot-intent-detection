#ifndef STGCN_GPU_OFFLOAD_H
#define STGCN_GPU_OFFLOAD_H

/*
  stgcn_gpu_offload.h

  SNPE offload for the Lite-STGCN action classifier. Uses the plain
  SNPEBuilder API (not PSNPE -- PSNPE's GPU path fails silently on this
  board with no diagnosable error, despite plain SNPEBuilder + GPU
  correctly validating ~94% of this model's ops; see stgcn_gpu_offload.cpp
  for the full history).

  Tries to build for GPU first; if the build fails (as it currently does,
  due to one unsupported op -- ReduceSum_Einsum_8_1, part of the graph
  convolution's Einsum decomposition -- that the GPU backend cannot
  validate), retries the build targeting CPU instead of giving up. This
  trades away GPU acceleration on the ops that DO validate, in exchange for
  a build that reliably succeeds.

  IMPORTANT ASSUMPTION (confirm with pose-estimation stage owner):
  This class expects its input already remapped to COCO-17 joint order,
  i.e. whatever comes out of BlazePose/MediaPipe (33 landmarks) must be
  reduced/reordered to the 17-joint COCO layout the Lite-STGCN was trained
  on BEFORE it reaches this class. That remapping is not this class's
  responsibility -- see the SkeletonFrame comment below.
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

constexpr int kNumCocoJoints = 17;   // graph_cfg layout='coco'
constexpr int kClipLen       = 100;  // clip_len in UniformSample
constexpr int kNumPerson     = 2;    // FormatGCNInput(num_person=2)
constexpr int kNumClasses    = 51;   // HMDB51


/*******************************************************************************
 * Type definitions
 ******************************************************************************/

/*
 * ClassificationResult
 */
struct ClassificationResult
{
  int   class_index  = -1; // classification identifier index defied by HMDB51
  float raw_score    = 0.0f;  // raw un-normalized score
  float confidence   = 0.0f;  // softmax-derived confidence
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

    /*
     * Reports which runtime the model actually ended up built for, after
     * the GPU-then-CPU-retry logic in the constructor runs. Useful for
     * logging/reporting, since a "ready" instance may be running on either
     * runtime depending on whether the GPU build succeeded.
     */
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
     * Attempts to build snpe_handle_t for the given runtime. Returns true
     * on success. On failure, cleans up whatever partial state it created
     * (builder_handle_t) so a subsequent retry with a different runtime
     * starts clean.
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
