#ifndef NORMALIZE_COCO_FRAME_H
#define NORMALIZE_COCO_FRAME_H

/*
  normalize_coco_frame.h

  Rescales a CocoFrame's (x, y) values into a [-1, 1] range relative to
  the source frame's dimensions, matching training-time preprocessing.
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
 * frame_width / frame_height: the ORIGINAL video frame's dimensions (not
 * the model's resized input dimensions -- see the open question above).
 *
 * confidence (score) values are passed through unchanged; only x/y are
 * transformed.
 */
CocoFrame normalized_coco_frame
(
  const CocoFrame& frame,
  int               frame_width,
  int               frame_height
);

#endif /* NORMALIZE_COCO_FRAME_H */