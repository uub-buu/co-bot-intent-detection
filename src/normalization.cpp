/*
  normalization.cpp

  Implementation file for CocoFrame normalization
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include "normalization.h"


/*******************************************************************************
 * Functions
 ******************************************************************************/

CocoFrame normalized_coco_frame
(
  const CocoFrame& frame,
  int               frame_width,
  int               frame_height
)
{
  CocoFrame out = frame;

  const float half_width  = static_cast<float>(frame_width) / 2.0f;
  const float half_height = static_cast<float>(frame_height) / 2.0f;

  for (int v = 0; v < kNumJointsCoco17; ++v)
  {
    out[v].x = (frame[v].x - half_width) / half_width;
    out[v].y = (frame[v].y - half_height) / half_height;
  }

  return out;
}