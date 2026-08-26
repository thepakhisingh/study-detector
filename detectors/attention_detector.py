"""
"Looking away" detection combining two independent signals so a person
isn't flagged just for a quick glance:

  1. Head pose (yaw/pitch) estimated with solvePnP against a generic 3D
     face model -- catches the head turning away from the screen.
  2. Iris position within the eye socket -- catches the eyes looking
     off-screen even while the head stays fairly still (e.g. glancing
     sideways without turning the head).

Either signal exceeding its threshold counts as "looking away" for that
frame; the caller (main.py) is responsible for requiring this to persist
for LOOKING_AWAY_DURATION seconds via the TimerManager before treating it
as a confirmed distraction.
"""

import math

import cv2
import numpy as np

# Generic 3D face model points (arbitrary units, right-handed, origin at
# nose tip) paired with their MediaPipe FaceMesh landmark indices. This is
# the standard 6-point model widely used for monocular head pose
# estimation via solvePnP.
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip           -> landmark 1
    (0.0, -330.0, -65.0),     # Chin               -> landmark 152
    (-225.0, 170.0, -135.0),  # Left eye left corner  -> landmark 33
    (225.0, 170.0, -135.0),   # Right eye right corner -> landmark 263
    (-150.0, -150.0, -125.0), # Left mouth corner  -> landmark 61
    (150.0, -150.0, -125.0),  # Right mouth corner -> landmark 291
], dtype=np.float64)

_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# Iris landmark indices available when FaceMesh is created with
# refine_landmarks=True. Center point plus eye-corner references used to
# compute the iris's normalized position within each eye socket.
_RIGHT_IRIS_CENTER = 468
_LEFT_IRIS_CENTER = 473
_RIGHT_EYE_CORNERS = (33, 133)   # (outer, inner)
_LEFT_EYE_CORNERS = (362, 263)   # (inner, outer)
_RIGHT_EYE_LIDS = (159, 145)     # (top, bottom)
_LEFT_EYE_LIDS = (386, 374)      # (top, bottom)


def _rotation_matrix_to_euler_angles(rmat):
    """Decompose a rotation matrix into (pitch, yaw, roll) in degrees."""
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0.0

    return np.degrees([x, y, z])  # pitch, yaw, roll


def _iris_offset(landmarks, iris_idx, corner_idx, lid_idx, frame_w, frame_h):
    iris = landmarks.landmark[iris_idx]
    c1 = landmarks.landmark[corner_idx[0]]
    c2 = landmarks.landmark[corner_idx[1]]
    top = landmarks.landmark[lid_idx[0]]
    bottom = landmarks.landmark[lid_idx[1]]

    ix, iy = iris.x * frame_w, iris.y * frame_h
    x1, x2 = c1.x * frame_w, c2.x * frame_w
    y1, y2 = top.y * frame_h, bottom.y * frame_h

    eye_width = abs(x2 - x1)
    eye_height = abs(y2 - y1)

    if eye_width < 1e-3 or eye_height < 1e-3:
        return 0.0, 0.0

    # Normalize to [-0.5, 0.5], 0 = perfectly centered in the socket.
    h_offset = ((ix - min(x1, x2)) / eye_width) - 0.5
    v_offset = ((iy - min(y1, y2)) / eye_height) - 0.5
    return h_offset, v_offset


class AttentionResult:
    __slots__ = (
        "looking_away", "yaw", "pitch", "roll",
        "iris_h_offset", "iris_v_offset",
        "nose_point_2d", "pose_axis_2d",
    )

    def __init__(self, looking_away, yaw, pitch, roll, iris_h_offset, iris_v_offset,
                 nose_point_2d, pose_axis_2d):
        self.looking_away = looking_away
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.iris_h_offset = iris_h_offset
        self.iris_v_offset = iris_v_offset
        self.nose_point_2d = nose_point_2d
        self.pose_axis_2d = pose_axis_2d


class AttentionDetector:
    def __init__(self, config):
        self.config = config

    def process(self, landmarks, frame_w, frame_h):
        cfg = self.config

        image_points = np.array([
            (landmarks.landmark[idx].x * frame_w, landmarks.landmark[idx].y * frame_h)
            for idx in _LANDMARK_IDS
        ], dtype=np.float64)

        focal_length = frame_w
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        yaw = pitch = roll = 0.0
        nose_point_2d = (int(image_points[0][0]), int(image_points[0][1]))
        pose_axis_2d = nose_point_2d

        success, rotation_vec, _translation_vec = cv2.solvePnP(
            _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        head_pose_away = False
        if success:
            rmat, _ = cv2.Rodrigues(rotation_vec)
            pitch, yaw, roll = _rotation_matrix_to_euler_angles(rmat)
            head_pose_away = (
                abs(yaw) > cfg.HEAD_YAW_THRESHOLD or abs(pitch) > cfg.HEAD_PITCH_THRESHOLD
            )

            axis_end, _ = cv2.projectPoints(
                np.array([(0.0, 0.0, 500.0)]), rotation_vec, _translation_vec,
                camera_matrix, dist_coeffs,
            )
            pose_axis_2d = (int(axis_end[0][0][0]), int(axis_end[0][0][1]))

        right_h, right_v = _iris_offset(
            landmarks, _RIGHT_IRIS_CENTER, _RIGHT_EYE_CORNERS, _RIGHT_EYE_LIDS, frame_w, frame_h
        )
        left_h, left_v = _iris_offset(
            landmarks, _LEFT_IRIS_CENTER, _LEFT_EYE_CORNERS, _LEFT_EYE_LIDS, frame_w, frame_h
        )
        iris_h_offset = (right_h + left_h) / 2.0
        iris_v_offset = (right_v + left_v) / 2.0

        gaze_away = (
            abs(iris_h_offset) > cfg.IRIS_H_OFFSET_THRESHOLD
            or abs(iris_v_offset) > cfg.IRIS_V_OFFSET_THRESHOLD
        )

        looking_away = head_pose_away or gaze_away

        return AttentionResult(
            looking_away=looking_away,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            iris_h_offset=iris_h_offset,
            iris_v_offset=iris_v_offset,
            nose_point_2d=nose_point_2d,
            pose_axis_2d=pose_axis_2d,
        )
