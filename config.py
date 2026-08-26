"""
Central configuration for the Study Focus Detector.

Every tunable value used by the detectors, timers, alert system and UI lives
here so nothing is hardcoded elsewhere in the codebase.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FLIP_HORIZONTAL = True  # mirror the feed so it feels natural to the user

# ----------------------------------------------------------------------------
# Eyes closed detection (Eye Aspect Ratio)
# ----------------------------------------------------------------------------
# EAR drops sharply when eyes close. This threshold was tuned against
# MediaPipe FaceMesh eye landmarks; below it we consider the eye "closed".
EAR_THRESHOLD = 0.21
# How long (seconds) the eyes must remain continuously closed before the
# EYES_CLOSED distraction is confirmed. Normal blinks (~0.1-0.4s) are ignored.
EYES_CLOSED_DURATION = 2.0

# ----------------------------------------------------------------------------
# Looking away detection (head pose + iris position)
# ----------------------------------------------------------------------------
# Head yaw/pitch (degrees) beyond which the head is considered turned away
# from the screen.
HEAD_YAW_THRESHOLD = 25.0
HEAD_PITCH_THRESHOLD = 20.0
# Horizontal/vertical iris offset (normalized, relative to eye corners)
# beyond which the gaze itself is considered off-screen, independent of
# head pose. Small saccades / normal eye movement stay under this.
IRIS_H_OFFSET_THRESHOLD = 0.35
IRIS_V_OFFSET_THRESHOLD = 0.35
# How long (seconds) the "looking away" condition (head pose OR gaze) must
# persist continuously before LOOKING_AWAY is confirmed.
LOOKING_AWAY_DURATION = 3.0

# ----------------------------------------------------------------------------
# Face missing / covered detection
# ----------------------------------------------------------------------------
# How long (seconds) the face must be continuously absent/obstructed before
# FACE_NOT_VISIBLE is confirmed.
FACE_MISSING_DURATION = 2.0
# Number of consecutive frames without a face detection that are tolerated
# as a "grace period" before we even start counting toward the duration
# above. Prevents a single dropped frame from starting a false timer.
FACE_MISSING_GRACE_FRAMES = 3
# If a hand is detected overlapping the last known face region while the
# face is not visible, we classify it as the user covering their own face
# (still reported as FACE_NOT_VISIBLE, but shown distinctly in the UI).
ENABLE_HAND_COVER_DETECTION = True

# ----------------------------------------------------------------------------
# Phone detection (YOLO / Ultralytics)
# ----------------------------------------------------------------------------
PHONE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
# COCO class id for "cell phone" is 67 in the standard 80-class COCO set
# that the stock Ultralytics COCO-pretrained models use.
PHONE_CLASS_ID = 67
PHONE_CONFIDENCE_THRESHOLD = 0.45
# How often (seconds) to run the (relatively expensive) YOLO inference.
# Detection state is held between inferences so the UI still updates every
# frame.
PHONE_DETECTION_INTERVAL = 0.2
# Single inference cycles are noisy (motion blur, angle, brief occlusion by
# a hand). Results are smoothed over this many recent inference cycles --
# e.g. 5 cycles at a 0.2s interval covers roughly the last 1 second -- so a
# lone missed cycle doesn't reset the PHONE_DETECTED timer.
PHONE_SMOOTHING_WINDOW = 5
# Minimum fraction of the recent cycles (above) that must have detected the
# phone for it to still count as "in use" on the current frame.
PHONE_SMOOTHING_MIN_RATIO = 0.4
# How long (seconds) a phone must be continuously detected near the user
# before PHONE_DETECTED is confirmed.
PHONE_DETECTION_DURATION = 3.0
# A phone box is only considered "in use" if it is reasonably large relative
# to the frame (i.e. close to the camera / in the user's hands) rather than
# a phone sitting far away in the background. Expressed as a minimum
# fraction of the frame area.
PHONE_MIN_AREA_RATIO = 0.01

# ----------------------------------------------------------------------------
# Alerts / audio
# ----------------------------------------------------------------------------
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
# Alert sound files are cycled round-robin in this order, e.g. for 3 files:
# alert1 -> alert2 -> alert3 -> alert1 -> alert2 -> ...
ALERT_SOUND_FILES = [
    os.path.join(AUDIO_DIR, "alert1.mp3"),
    os.path.join(AUDIO_DIR, "alert2.mp3"),
    os.path.join(AUDIO_DIR, "alert3.mp3"),
]
# Minimum seconds between two alert sounds being played, regardless of how
# many distraction events occur in that window.
ALERT_COOLDOWN = 5.0

# ----------------------------------------------------------------------------
# Debug / UI
# ----------------------------------------------------------------------------
DEBUG = True
SHOW_FACE_LANDMARKS = True
SHOW_EYE_LANDMARKS = True
SHOW_HEAD_ORIENTATION = True
SHOW_PHONE_BOXES = True
SHOW_CONFIDENCE_VALUES = True

WINDOW_NAME = "Study Focus Detector"
