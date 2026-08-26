"""
Face presence detection using MediaPipe FaceMesh, plus optional MediaPipe
Hands to distinguish "user is covering their face with a hand" from "face
is simply out of frame / not detected".
"""

import mediapipe as mp


class FaceResult:
    __slots__ = ("present", "in_grace_period", "landmarks", "bbox", "covered_by_hand")

    def __init__(self, present, in_grace_period, landmarks, bbox, covered_by_hand):
        self.present = present
        self.in_grace_period = in_grace_period
        self.landmarks = landmarks
        self.bbox = bbox
        self.covered_by_hand = covered_by_hand


def _landmarks_to_bbox(landmarks, frame_w, frame_h, padding=0.15):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding
    x_min = max(0.0, x_min - pad_x)
    x_max = min(1.0, x_max + pad_x)
    y_min = max(0.0, y_min - pad_y)
    y_max = min(1.0, y_max + pad_y)

    return (
        int(x_min * frame_w),
        int(y_min * frame_h),
        int(x_max * frame_w),
        int(y_max * frame_h),
    )


def _boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


class FaceDetector:
    def __init__(self, config):
        self.config = config
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # enables iris landmarks (indices 468-477)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.hands = None
        if config.ENABLE_HAND_COVER_DETECTION:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        self._missing_frame_count = 0
        self._last_bbox = None

    def process(self, frame_rgb):
        frame_h, frame_w = frame_rgb.shape[:2]
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            bbox = _landmarks_to_bbox(landmarks, frame_w, frame_h)
            self._last_bbox = bbox
            self._missing_frame_count = 0
            return FaceResult(
                present=True,
                in_grace_period=False,
                landmarks=landmarks,
                bbox=bbox,
                covered_by_hand=False,
            )

        # No face found this frame.
        self._missing_frame_count += 1
        in_grace_period = self._missing_frame_count <= self.config.FACE_MISSING_GRACE_FRAMES

        covered_by_hand = False
        if self.hands is not None and self._last_bbox is not None:
            hand_results = self.hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
                    ys = [lm.y * frame_h for lm in hand_landmarks.landmark]
                    hand_bbox = (min(xs), min(ys), max(xs), max(ys))
                    if _boxes_overlap(hand_bbox, self._last_bbox):
                        covered_by_hand = True
                        break

        return FaceResult(
            present=False,
            in_grace_period=in_grace_period,
            landmarks=None,
            bbox=None,
            covered_by_hand=covered_by_hand,
        )

    def close(self):
        self.face_mesh.close()
        if self.hands is not None:
            self.hands.close()
