/*
  class_labels.cpp

  Implementation file for the HMDB51 class-index-to-label lookup table
*/

/*******************************************************************************
 * Includes
 ******************************************************************************/

#include "class_labels.h"


namespace
{
  /*
   * Alphabetical order, matching pyskl/tools/data/label_map/hmdb51.txt
   * exactly. Index 9 is "dribble"; index 19 is "jump".
   */
  const char* kHmdb51Labels[kNumClasses] =
  {
    "brush_hair",     "cartwheel", "catch",     "chew",           "clap",
    "climb",          "climb_stairs", "dive",   "draw_sword",     "dribble",
    "drink",          "eat",       "fall_floor", "fencing",       "flic_flac",
    "golf",           "handstand", "hit",       "hug",            "jump",
    "kick",           "kick_ball", "kiss",      "laugh",          "pick",
    "pour",           "pullup",    "punch",     "push",           "pushup",
    "ride_bike",      "ride_horse", "run",      "shake_hands",    "shoot_ball",
    "shoot_bow",      "shoot_gun", "sit",       "situp",          "smile",
    "smoke",          "somersault", "stand",    "swing_baseball", "sword",
    "sword_exercise", "talk",      "throw",     "turn",           "walk",
    "wave",
  };
}

const char* hmdb51_label(int class_index)
{
  if (class_index < 0 || class_index >= kNumClasses)
  {
    return "unknown";
  }

  return kHmdb51Labels[class_index];
}
