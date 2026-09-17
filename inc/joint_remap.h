#ifndef JOINT_REMAP_H
#define JOINT_REMAP_H

/*
  joint_remap.h

  Maps MediaPipe/BlazePose's 33-landmark body topology down to COCO's
  17-keypoint layout, maching with the Lite-STGCN model.
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include <array>

#include "pose_dsp_offload.h"


/*******************************************************************************
 * Data
 ******************************************************************************/

/* Number of joints in COCO's keypoint layout */
constexpr int kNumJointsCoco17 = 17;


/*******************************************************************************
 * Type definitions
 ******************************************************************************/

/*
 * One COCO-format joint: x, y, confidence score - matches the STGCN model's
 * expected [x, y, score] per-joint channel layout.
 */
struct CocoJoint
{
  float x = 0.f;
  float y = 0.f;
  float score = 0.f;
};

using CocoFrame = std::array<CocoJoint, kNumJointsCoco17>;


/*******************************************************************************
 * Functions
 ******************************************************************************/

/*
 * remap_mediapipe_to_coco17
 *
 * Converts one frame of MediaPipe's 33-landmark output into COCO's 17 point
 * layout, in COCO's standard joint order.
 */
CocoFrame remap_mediapipe_to_coco17
(
  const JointFrame& mediapipe_joints
);

#endif /* JOINT_REMAP_H */
