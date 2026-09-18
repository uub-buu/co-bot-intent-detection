#ifndef CLASS_LABELS_H
#define CLASS_LABELS_H

/*
  class_labels.h

  Maps the STGCN model's output class index to its human-readable HMDB51
  action name. Order is alphabetical, taken directly from
  pyskl/tools/data/label_map/hmdb51.txt -- this is the authoritative source,
  not an assumed/guessed ordering.
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include "stgcn_gpu_offload.h"


/*******************************************************************************
 * Functions
 ******************************************************************************/

/*
 * hmdb51_label
 *
 * Returns the action name for a class index in [0, kNumClasses). Returns
 * "unknown" for an out-of-range index rather than crashing, since a bad
 * index here means something upstream went wrong and this is a display/
 * logging path, not something that should bring the program down.
 */
const char* hmdb51_label(int class_index);

#endif /* CLASS_LABELS_H */
