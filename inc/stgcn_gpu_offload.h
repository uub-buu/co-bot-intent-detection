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
#include "SNPE/SNPEUtil.h"
#include "SNPE/SNPEBuilder.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/ITensor.h"
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
    StgcnGpuOffload
    (
      const std::string& model_path
    );

    ~StgcnGpuOffload();

    bool is_ready() const { return snpe_handle_t != nullptr; }

    /* classify: Returns false if the model isn't ready or the clip size is wrong. */
    bool classify
    (
      const std::vector<float>& window_tensor,
      ClassificationResult&     out_result
    );

  private:
    /* pass in results from windowing_buffer::add_frame*/
    Snpe_ITensor_Handle_t preprocess(const std::vector<float>& window_tensor);
    bool postprocess(Snpe_TensorMap_Handle_t output_map_handle, ClassificationResult& out_result);
    
    Snpe_DlContainer_Handle_t container_handle_t    = nullptr;
    Snpe_SNPEBuilder_Handle_t builder_handle_t     = nullptr;
    Snpe_SNPE_Handle_t        snpe_handle_t       = nullptr;
    Snpe_TensorShape_Handle_t input_shape_handle_t  = nullptr;
    std::string               input_tensor_name;
};

#endif /* STGCN_GPU_OFFLOAD_H */
