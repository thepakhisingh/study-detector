"""
Mobile phone detection using an Ultralytics YOLO model.

YOLO inference is comparatively expensive, so it is only re-run every
PHONE_DETECTION_INTERVAL seconds; the most recent detections are cached and
returned on the frames in between so the UI still updates every frame.

A phone is only reported as "in use" (as opposed to sitting somewhere in
the background) when its bounding box covers at least PHONE_MIN_AREA_RATIO
of the frame, i.e. it's close to the camera / in the user's hands.

A single inference cycle is noisy -- motion blur, a slight angle change, or
a hand partially covering the phone can drop its confidence below the
threshold for one cycle even during continuous real use. If that raw
per-cycle result were fed directly into the PHONE_DETECTED timer, a single
missed cycle would reset the whole multi-second timer back to zero and the
condition could take much longer than PHONE_DETECTION_DURATION to ever
confirm. To avoid that, results are smoothed over a short rolling window of
recent inference cycles (PHONE_SMOOTHING_WINDOW) and "in use" is reported
as long as at least PHONE_SMOOTHING_MIN_RATIO of the recent cycles detected
it -- brief single-cycle misses no longer reset the timer, while a phone
that's genuinely not there for a sustained stretch still reports absent.
"""

import time
from collections import deque


class PhoneBox:
    __slots__ = ("x1", "y1", "x2", "y2", "confidence")

    def __init__(self, x1, y1, x2, y2, confidence):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.confidence = confidence


class PhoneDetector:
    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.model = None
        self._last_run = 0.0
        self._last_boxes = []
        self._recent_detections = deque(maxlen=config.PHONE_SMOOTHING_WINDOW)

        try:
            from ultralytics import YOLO
            self.model = YOLO(config.PHONE_MODEL_PATH)
        except Exception as exc:
            self.enabled = False
            print(f"[PhoneDetector] Disabled -- could not load YOLO model: {exc}")

    def process(self, frame_bgr):
        """Returns (phone_in_use: bool, boxes: list[PhoneBox])."""
        if not self.enabled:
            return False, []

        now = time.time()
        if now - self._last_run >= self.config.PHONE_DETECTION_INTERVAL:
            self._last_run = now
            self._last_boxes = self._run_inference(frame_bgr)

            frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
            cycle_detected = any(
                ((box.x2 - box.x1) * (box.y2 - box.y1)) / frame_area >= self.config.PHONE_MIN_AREA_RATIO
                for box in self._last_boxes
            )
            self._recent_detections.append(cycle_detected)

        if not self._recent_detections:
            return False, self._last_boxes

        hit_ratio = sum(self._recent_detections) / len(self._recent_detections)
        phone_in_use = hit_ratio >= self.config.PHONE_SMOOTHING_MIN_RATIO
        return phone_in_use, self._last_boxes

    def _run_inference(self, frame_bgr):
        cfg = self.config
        try:
            results = self.model.predict(
                frame_bgr,
                classes=[cfg.PHONE_CLASS_ID],
                conf=cfg.PHONE_CONFIDENCE_THRESHOLD,
                verbose=False,
            )
        except Exception as exc:
            print(f"[PhoneDetector] Inference failed, disabling phone detection: {exc}")
            self.enabled = False
            return []

        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                boxes.append(PhoneBox(x1, y1, x2, y2, confidence))
        return boxes
