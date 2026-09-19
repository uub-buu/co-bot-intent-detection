#ifndef CLASS_LABELS_H
#define CLASS_LABELS_H

/*
  class_labels.h

  Maps STGCN output index to HMDB51 label. Order matches
  pyskl/tools/data/label_map/hmdb51.txt exactly.
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include "stgcn_gpu_offload.h"


/*******************************************************************************
 * Functions
 ******************************************************************************/

/*
 * hmdb51_label. Returns label for index in [0, kNumClasses); returns
 * "unknown" if out of range instead of crashing.
 */
const char* hmdb51_label
(
  int class_index
);

#endif /* CLASS_LABELS_H */
