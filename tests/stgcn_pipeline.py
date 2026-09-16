"""
Full pipeline: video capture -> MediaPipe pose -> remap to COCO-17 ->
normalize -> sliding window buffer -> ONNX inference -> predicted action.

This is a DEV-MACHINE PROTOTYPE, not the final RB3 deployment code.
It uses onnxruntime (Python) to validate the pipeline logic -- the actual
RB3 deployment will run the quantized .dlc through the SNPE C++ API instead.

Requirements:
    pip install mediapipe opencv-python onnxruntime numpy
"""

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from collections import deque

# ---------------------------------------------------------------------------
# 1. MediaPipe (33 landmarks) -> COCO-17 remapping
#    MediaPipe Pose landmark indices: https://developers.google.com/mediapipe
#    COCO-17 order is what your Lite-STGCN was trained on (see project notes).
# ---------------------------------------------------------------------------
MEDIAPIPE_TO_COCO17 = [
    0,   # nose
    2,   # left eye
    5,   # right eye
    7,   # left ear
    8,   # right ear
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
]
NUM_COCO_JOINTS = 17

# ---------------------------------------------------------------------------
# 2. HMDB51 class names.
#    IMPORTANT: this is the standard alphabetical HMDB51 class list. Verify
#    this matches the label indices in your actual training pickle before
#    trusting predictions -- if pyskl's annotation file orders labels
#    differently, this list will silently produce WRONG class names even
#    though the numeric prediction itself is correct.
# ---------------------------------------------------------------------------
HMDB51_CLASSES = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb",
    "climb_stairs", "dive", "draw_sword", "dribble", "drink", "eat",
    "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit",
    "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick", "pour",
    "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse", "run",
    "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp",
    "smile", "smoke", "somersault", "stand", "swing_baseball", "sword",
    "sword_exercise", "talk", "throw", "turn", "walk", "wave",
]
assert len(HMDB51_CLASSES) == 51

# ---------------------------------------------------------------------------
# 3. Model / pipeline constants (must match training config exactly)
# ---------------------------------------------------------------------------
CLIP_LEN = 100      # frames per clip -- matches clip_len=100 in your configs
NUM_PERSON = 2      # matches FormatGCNInput(num_person=2)
ONNX_MODEL_PATH = "./model/onnx/lite_stgcn_hmdb51.onnx"  # path to your exported model


def extract_and_remap_landmarks(mp_results, frame_width, frame_height):
    """
    Pull MediaPipe's 33 landmarks out of a single frame's results and
    remap them to a (17, 3) array in COCO-17 order: (x_pixels, y_pixels,
    confidence). Returns None if no person was detected in this frame.
    """
    if mp_results.pose_landmarks is None:
        return None

    landmarks = mp_results.pose_landmarks.landmark  # 33 entries
    coco_kp = np.zeros((NUM_COCO_JOINTS, 3), dtype=np.float32)

    for coco_idx, mp_idx in enumerate(MEDIAPIPE_TO_COCO17):
        lm = landmarks[mp_idx]
        coco_kp[coco_idx, 0] = lm.x * frame_width   # MediaPipe outputs normalized [0,1]
        coco_kp[coco_idx, 1] = lm.y * frame_height  # scale back to pixel space
        coco_kp[coco_idx, 2] = lm.visibility        # MediaPipe's confidence analogue

    return coco_kp


def pre_normalize_2d(keypoint_seq, img_shape):
    """
    Best-effort match to pyskl's PreNormalize2D behavior: rescale pixel-space
    keypoints into a normalized range relative to the frame dimensions.

    NOTE: this is inferred from pyskl/mmaction2's documented behavior
    ("normalize the range of keypoint values" relative to img_shape), not
    copied from literal source. Validate against a known training sample
    before trusting this for real accuracy numbers.

    keypoint_seq: (T, V, 3) array, x/y in pixel space, confidence unchanged
    img_shape: (height, width) of the source frames
    """
    h, w = img_shape
    normalized = keypoint_seq.copy()
    normalized[..., 0] = (keypoint_seq[..., 0] - w / 2) / (w / 2)
    normalized[..., 1] = (keypoint_seq[..., 1] - h / 2) / (h / 2)
    # confidence (channel 2) is left unchanged
    return normalized


def build_model_input(frame_buffer):
    """
    frame_buffer: deque of up to CLIP_LEN arrays, each (17, 3) -- one
    detected person's keypoints per frame (or None for missed frames).

    Returns a (1, NUM_PERSON, CLIP_LEN, 17, 3) float32 array ready for
    the ONNX model, with the second "person" slot zero-padded (since
    MediaPipe's base Pose solution only tracks one person at a time).
    """
    frames = list(frame_buffer)

    # Pad on the left with zero-frames if we don't have CLIP_LEN yet
    while len(frames) < CLIP_LEN:
        frames.insert(0, np.zeros((NUM_COCO_JOINTS, 3), dtype=np.float32))

    # Replace any missed-detection frames (None) with zeros too
    frames = [f if f is not None else np.zeros((NUM_COCO_JOINTS, 3), dtype=np.float32)
              for f in frames]

    person_0 = np.stack(frames, axis=0)             # (CLIP_LEN, 17, 3)
    person_1 = np.zeros_like(person_0)               # second person slot: zero-padded

    clip = np.stack([person_0, person_1], axis=0)    # (NUM_PERSON, CLIP_LEN, 17, 3)
    clip = clip[np.newaxis, ...].astype(np.float32)   # (1, NUM_PERSON, CLIP_LEN, 17, 3)
    return clip


def run_pipeline(video_source=0):
    """
    video_source: 0 for default webcam, or a path to a video file.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name

    frame_buffer = deque(maxlen=CLIP_LEN)

    print("Starting pipeline. Press 'q' to quit.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        mp_results = pose.process(frame_rgb)
        coco_kp = extract_and_remap_landmarks(mp_results, w, h)
        frame_buffer.append(coco_kp)

        # Only run inference once we have a full window of frames
        if len(frame_buffer) == CLIP_LEN:
            model_input = build_model_input(frame_buffer)
            model_input[0] = pre_normalize_2d(model_input[0].reshape(-1, NUM_COCO_JOINTS, 3),
                                               (h, w)).reshape(model_input[0].shape)

            outputs = session.run(None, {input_name: model_input})
            scores = outputs[0][0]  # shape (51,)
            pred_idx = int(np.argmax(scores))
            pred_label = HMDB51_CLASSES[pred_idx]
            confidence = float(scores[pred_idx])

            cv2.putText(
                frame_bgr, f"{pred_label} ({confidence:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )

        cv2.imshow("Lite-STGCN action recognition", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":

    run_pipeline(video_source='./video_data/golfing.mp4')  # 0 = default webcam; or pass a video file path