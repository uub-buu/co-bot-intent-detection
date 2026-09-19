#ifndef NORMALIZATION_H
#define NORMALIZATION_H

/*
  normalization.h

  Rescales CocoFrame x,y to [-1, 1] relative to frame size. Matches
  training-time PreNormalize2D.
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include "joint_remap.h"


/*******************************************************************************
 * Functions
 ******************************************************************************/

/*
 * normalize_coco_frame
 *
 * NOTE: frame_width / frame_height are the origianl video's resolution
 *
 */
CocoFrame normalize_coco_frame
(
  const CocoFrame& frame,
  int              frame_width,
  int              frame_height
);

#endif /* NORMALIZATION_H */
