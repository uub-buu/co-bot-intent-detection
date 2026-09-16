/*
  stgcn_gpu_offload.h

  SNPE GPU offload for the Lite-STGCN action classifier.

  Counterpart to pose_dsp_offload.h/.cpp -- same SNPE Builder pattern,
  different runtime target (SNPE_RUNTIME_GPU instead of SNPE_RUNTIME_DSP)
  and a different model (Lite-STGCN action classifier instead of BlazePose
  pose estimator).

  IMPORTANT ASSUMPTION (confirm with pose-estimation stage owner):
  This class expects its input already remapped to COCO-17 joint order,
  i.e. whatever comes out of BlazePose/MediaPipe (33 landmarks) must be
  reduced/reordered to the 17-joint COCO layout the Lite-STGCN was trained
  on BEFORE it reaches this class. That remapping is not this class's
  responsibility -- see the SkeletonFrame comment below.
*/

#ifndef STGCN_GPU_OFFLOAD_H
#define STGCN_GPU_OFFLOAD_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "SNPE/SNPE.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/ITensor.h"
#include "DlSystem/TensorShape.h"
#include "DlContainer/DlContainer.h"

/*******************************************************************************
 * Constants -- must match the training config exactly
 * (configs/stgcn/stgcn_lite_hmdb51_hrnet/j.py)
 ******************************************************************************/
constexpr int kNumCocoJoints = 17;   // graph_cfg layout='coco'
constexpr int kClipLen       = 100;  // clip_len in UniformSample
constexpr int kNumPerson     = 2;    // FormatGCNInput(num_person=2)
constexpr int kNumClasses    = 51;   // HMDB51

/*
 * Joint
 * One COCO-17 joint's (x, y, confidence), already in the same convention
 * the model was trained on: pixel-space x/y, confidence in [0, 1].
 */
struct Joint
{
  float x          = 0.0f;
  float y          = 0.0f;
  float confidence = 0.0f;
};

/*
 * SkeletonFrame
 * One frame's joints, for up to kNumPerson people. If fewer than
 * kNumPerson people were detected, the unused slots should be left
 * zero-initialized -- this mirrors FormatGCNInput's zero-padding
 * behavior from the training pipeline.
 */
struct SkeletonFrame
{
  std::array<std::array<Joint, kNumCocoJoints>, kNumPerson> persons{};
  int num_detected_persons = 0;
};

/*
 * ClassificationResult
 */
struct ClassificationResult
{
  int   class_index  = -1;
  float raw_score    = 0.0f;  // pre-softmax logit
  float confidence   = 0.0f;  // post-softmax probability
};

class StgcnGpuOffload
{
  public:
    StgcnGpuOffload
    (
      const std::string& model_path,
      bool                prefer_gpu = true
    );

    ~StgcnGpuOffload();

    bool is_ready() const { return snpe_handle_ != nullptr; }

    /*
     * classify
     *
     * clip: exactly kClipLen SkeletonFrame entries, already in temporal
     * order and already normalized (see PreNormalize2D-equivalent note
     * in the .cpp -- normalization is NOT done inside this class, to
     * keep it a pure "tensor in, scores out" wrapper like PoseDSPOffload
     * keeps decode/resize outside of ExecuteITensors).
     *
     * Returns false if the model isn't ready or the clip size is wrong.
     */
    bool classify
    (
      const std::vector<SkeletonFrame>& clip,
      ClassificationResult&             out_result
    );

  private:
    Snpe_ITensor_Handle_t preprocess(const std::vector<SkeletonFrame>& clip);
    bool postprocess(Snpe_TensorMap_Handle_t output_map_handle, ClassificationResult& out_result);

    Snpe_DlContainer_Handle_t container_handle_    = nullptr;
    Snpe_SNPEBuilder_Handle_t builder_handle_      = nullptr;
    Snpe_SNPE_Handle_t        snpe_handle_         = nullptr;
    Snpe_TensorShape_Handle_t input_shape_handle_  = nullptr;
    std::string               input_tensor_name_;
};

#endif  // STGCN_GPU_OFFLOAD_H