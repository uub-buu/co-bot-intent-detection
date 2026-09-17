/*
  joint_remap.cpp

  Implementation file for MediaPipe33 to COCO17 joint remapping
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include "joint_remap.h"


namespace
{
  /*
   * MediaPipe landmark index for each COCO joint, in COCO's standard order:
   * nose, L/R eye, L/R ear, L/R shoulder, L/R elbow, L/R wrist, L/R hip,
   * L/R knee, L/R ankle.
   *
   * MediaPipe's eyes are each split into inner/center/outer landmarks. The
   * center landmark (index 2 / 5) is used as the closest match to COCO's single
   * eye point per side.
   */
  constexpr int kMediaPipeIndexForCocoJoint[kNumJointsCoco17] =
  {
    0,    /* nose            */
    2,    /* left_eye        */
    5,    /* right_eye       */
    7,    /* left_ear        */
    8,    /* right_ear       */
    11,   /* left_shoulder   */
    12,   /* right_shoulder  */
    13,   /* left_elbow      */
    14,   /* right_elbow     */
    15,   /* left_wrist      */
    16,   /* right_wrist     */
    23,   /* left_hip        */
    24,   /* right_hip       */
    25,   /* left_knee       */
    26,   /* right_knee      */
    27,   /* left_ankle      */
    28,   /* right_ankle     */
  };
}


/*******************************************************************************
 * Functions
 ******************************************************************************/

CocoFrame remap_mediapipe_to_coco17
(
  const JointFrame& mediapipe_joints
)
{
  CocoFrame coco_joints;

  for (int coco_idx = 0; coco_idx < kNumJointsCoco17; ++coco_idx)
  {
    const int       mp_idx = kMediaPipeIndexForCocoJoint[coco_idx];
    const Landmark& lm = mediapipe_joints[mp_idx];
    CocoJoint&      out = coco_joints[coco_idx];

    /*
     * visibility is the closest analog to COCO's per-joint confidence
     * score - see pose_dsp_offload.h for how it's derived (sigmoid of
     * the model's raw output).
     * */
    out.x = lm.x;
    out.y = lm.y;
    out.score = lm.visibility;
  }

  return coco_joints;
}
