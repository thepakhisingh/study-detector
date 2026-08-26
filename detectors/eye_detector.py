"""
Eye Aspect Ratio (EAR) based eyes-closed detection using MediaPipe FaceMesh
landmarks.

EAR is the classic Soukupova & Cech metric: the ratio of the eye's vertical
opening to its horizontal width. It drops sharply towards 0 when the eye
closes and stays roughly constant while open, which makes it robust to
head size/distance from the camera.
"""

import math

# Six-point eye contours (outer corner, two upper lid points, inner corner,
# two lower lid points) for each eye, using MediaPipe FaceMesh's 468-point
# topology. These are the standard indices used for EAR calculation.
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _eye_points(landmarks, indices, frame_w, frame_h):
    pts = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        pts.append((lm.x * frame_w, lm.y * frame_h))
    return pts


def _ear_from_points(pts):
    # pts order: [outer, upper1, upper2, inner, lower2, lower1]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


class EyeResult:
    __slots__ = ("ear", "left_ear", "right_ear", "eyes_closed", "left_points", "right_points")

    def __init__(self, ear, left_ear, right_ear, eyes_closed, left_points, right_points):
        self.ear = ear
        self.left_ear = left_ear
        self.right_ear = right_ear
        self.eyes_closed = eyes_closed
        self.left_points = left_points
        self.right_points = right_points


class EyeDetector:
    def __init__(self, config):
        self.config = config

    def process(self, landmarks, frame_w, frame_h):
        left_points = _eye_points(landmarks, LEFT_EYE, frame_w, frame_h)
        right_points = _eye_points(landmarks, RIGHT_EYE, frame_w, frame_h)

        left_ear = _ear_from_points(left_points)
        right_ear = _ear_from_points(right_points)
        avg_ear = (left_ear + right_ear) / 2.0

        eyes_closed = avg_ear < self.config.EAR_THRESHOLD

        return EyeResult(
            ear=avg_ear,
            left_ear=left_ear,
            right_ear=right_ear,
            eyes_closed=eyes_closed,
            left_points=left_points,
            right_points=right_points,
        )
