"""
Study Focus Detector -- entry point.

Continuously reads the webcam feed and evaluates, every frame, whether the
user is:
  - looking away from the screen for too long,
  - keeping their eyes closed for too long,
  - not visible / covered for too long,
  - or using a phone for too long.

Each condition has its own independent debounce timer (utils/timer_manager)
so a brief flicker never triggers a false alert, and only a continuous
condition lasting past its configured threshold plays an alternating alert
sound (utils/alert_manager), subject to a cooldown.

Press 'q' or Esc to quit, 'd' to toggle the debug overlay.
"""

import time

import cv2

import config
from detectors.attention_detector import AttentionDetector
from detectors.eye_detector import EyeDetector
from detectors.face_detector import FaceDetector
from detectors.phone_detector import PhoneDetector
from utils.alert_manager import AlertManager
from utils.timer_manager import TimerManager
from utils import drawing

# Higher-priority conditions are shown/alerted first when more than one is
# true in the same frame (e.g. face missing takes precedence over a stale
# "looking away" reading computed on a previous frame's landmarks).
PRIORITY = ["FACE_NOT_VISIBLE", "PHONE_DETECTED", "LOOKING_AWAY", "EYES_CLOSED"]

DURATIONS = {
    "FACE_NOT_VISIBLE": config.FACE_MISSING_DURATION,
    "PHONE_DETECTED": config.PHONE_DETECTION_DURATION,
    "LOOKING_AWAY": config.LOOKING_AWAY_DURATION,
    "EYES_CLOSED": config.EYES_CLOSED_DURATION,
}


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def main():
    cap = open_camera(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[main] Could not open camera index {config.CAMERA_INDEX}.")
        return

    face_detector = FaceDetector(config)
    eye_detector = EyeDetector(config)
    attention_detector = AttentionDetector(config)
    phone_detector = PhoneDetector(config)
    alert_manager = AlertManager(config.ALERT_SOUND_FILES, config.ALERT_COOLDOWN)
    timers = TimerManager(DURATIONS)

    debug = config.DEBUG

    # "Locked in" tracking: how long the current unbroken FOCUSED streak has
    # run, plus the total time spent FOCUSED across the whole session.
    # focus_streak_start is None whenever status isn't FOCUSED right now.
    focus_streak_start = None
    total_focused_seconds = 0.0
    last_frame_time = None

    print("[main] Study Focus Detector running. Press 'q' or Esc to quit, 'd' to toggle debug overlay.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[main] Failed to read frame from camera, stopping.")
                break

            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            face_result = face_detector.process(frame_rgb)
            phone_in_use, phone_boxes = phone_detector.process(frame)

            eye_result = None
            attention_result = None

            face_missing_active = (not face_result.present) and (not face_result.in_grace_period)
            eyes_closed_active = False
            looking_away_active = False

            if face_result.present:
                eye_result = eye_detector.process(face_result.landmarks, frame_w, frame_h)
                attention_result = attention_detector.process(face_result.landmarks, frame_w, frame_h)
                eyes_closed_active = eye_result.eyes_closed
                looking_away_active = attention_result.looking_away

            now = time.time()
            elapsed_map = {}
            confirmed_now = []

            for name, active in (
                ("FACE_NOT_VISIBLE", face_missing_active),
                ("PHONE_DETECTED", phone_in_use),
                ("LOOKING_AWAY", looking_away_active),
                ("EYES_CLOSED", eyes_closed_active),
            ):
                elapsed, just_confirmed = timers.update(name, active, now=now)
                elapsed_map[name] = elapsed
                if just_confirmed:
                    confirmed_now.append(name)

            # Fire alerts for any condition that just crossed its threshold
            # this frame (there could theoretically be more than one).
            for name in confirmed_now:
                played = alert_manager.notify_distraction(name)
                suffix = f" (sound: {alert_manager.last_sound_file})" if played else " (sound skipped: cooldown/busy)"
                print(f"[main] Distraction confirmed: {name}{suffix}")

            # Pick the single condition to show as the current status,
            # preferring whichever confirmed/in-progress condition ranks
            # highest in PRIORITY.
            current_status = "FOCUSED"
            for name in PRIORITY:
                if timers.is_confirmed(name) or elapsed_map[name] > 0:
                    current_status = name
                    break

            status_elapsed = elapsed_map.get(current_status, 0.0)
            status_threshold = DURATIONS.get(current_status, 0.0)

            frame_dt = 0.0 if last_frame_time is None else now - last_frame_time
            last_frame_time = now

            if current_status == "FOCUSED":
                if focus_streak_start is None:
                    focus_streak_start = now
                total_focused_seconds += frame_dt
                locked_in_streak = now - focus_streak_start
            else:
                focus_streak_start = None
                locked_in_streak = 0.0

            drawing.draw_hud(
                frame,
                current_status,
                status_elapsed,
                status_threshold,
                alert_manager.cooldown_remaining(),
                alert_manager.distraction_count,
                locked_in_streak=locked_in_streak,
                total_focused=total_focused_seconds,
            )

            if debug:
                if face_result.bbox is not None:
                    drawing.draw_face_bbox(frame, face_result.bbox)
                if config.SHOW_FACE_LANDMARKS and face_result.landmarks is not None:
                    drawing.draw_face_landmarks(frame, face_result.landmarks)
                if config.SHOW_EYE_LANDMARKS and eye_result is not None:
                    drawing.draw_eye_landmarks(frame, eye_result)
                if config.SHOW_HEAD_ORIENTATION and attention_result is not None:
                    drawing.draw_head_orientation(frame, attention_result)
                if config.SHOW_PHONE_BOXES and phone_boxes:
                    drawing.draw_phone_boxes(frame, phone_boxes, config.SHOW_CONFIDENCE_VALUES)

                debug_lines = [
                    f"Face present: {face_result.present} (grace: {face_result.in_grace_period})",
                    f"Face covered by hand: {face_result.covered_by_hand}",
                ]
                if eye_result is not None:
                    debug_lines.append(f"EAR: {eye_result.ear:0.3f} (threshold {config.EAR_THRESHOLD})")
                if attention_result is not None:
                    debug_lines.append(
                        f"Yaw: {attention_result.yaw:0.1f} Pitch: {attention_result.pitch:0.1f}"
                    )
                    debug_lines.append(
                        f"Iris offset H:{attention_result.iris_h_offset:0.2f} V:{attention_result.iris_v_offset:0.2f}"
                    )
                debug_lines.append(f"Phone in use: {phone_in_use} ({len(phone_boxes)} box(es))")
                drawing.draw_debug_panel(frame, debug_lines)

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or Esc
                break
            if key == ord("d"):
                debug = not debug

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detector.close()
        alert_manager.shutdown()
        print(f"[main] Session ended. Total distractions: {alert_manager.distraction_count}")


if __name__ == "__main__":
    main()
