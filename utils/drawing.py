"""
All OpenCV overlay/HUD rendering lives here, kept separate from detection
and state-management logic.
"""

import cv2

STATUS_COLORS = {
    "FOCUSED": (80, 200, 80),
    "LOOKING_AWAY": (0, 165, 255),
    "EYES_CLOSED": (0, 165, 255),
    "FACE_NOT_VISIBLE": (0, 0, 255),
    "PHONE_DETECTED": (0, 0, 255),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX

LOCKED_IN_COLOR = (60, 230, 130)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def draw_hud(frame, status, elapsed, threshold, cooldown_remaining, distraction_count,
             locked_in_streak=0.0, total_focused=0.0):
    color = STATUS_COLORS.get(status, (255, 255, 255))
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 116), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    cv2.putText(frame, f"Status: {status}", (12, 26), FONT, 0.7, color, 2, cv2.LINE_AA)

    timer_text = f"Timer: {elapsed:0.1f}s / {threshold:0.1f}s" if status != "FOCUSED" else "Timer: --"
    cv2.putText(frame, timer_text, (12, 52), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    cooldown_text = f"Alert cooldown: {cooldown_remaining:0.1f}s" if cooldown_remaining > 0 else "Alert cooldown: ready"
    cv2.putText(frame, cooldown_text, (12, 74), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # "Locked in" = current unbroken FOCUSED streak. Only lights up green
    # while status == FOCUSED; any active distraction condition resets it,
    # which is enforced by the caller resetting locked_in_streak to 0.
    locked_in_text = f"Locked in: {format_duration(locked_in_streak)}" if status == "FOCUSED" else "Locked in: -- (not focused)"
    locked_in_color = LOCKED_IN_COLOR if status == "FOCUSED" else (140, 140, 140)
    cv2.putText(frame, locked_in_text, (12, 100), FONT, 0.6, locked_in_color, 2, cv2.LINE_AA)

    count_text = f"Distractions this session: {distraction_count}"
    (text_w, _), _ = cv2.getTextSize(count_text, FONT, 0.55, 1)
    cv2.putText(frame, count_text, (w - text_w - 12, 26), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    total_text = f"Total focused: {format_duration(total_focused)}"
    (text_w2, _), _ = cv2.getTextSize(total_text, FONT, 0.55, 1)
    cv2.putText(frame, total_text, (w - text_w2 - 12, 52), FONT, 0.55, LOCKED_IN_COLOR, 1, cv2.LINE_AA)


def draw_face_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks.landmark:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (90, 200, 250), -1)


def draw_eye_landmarks(frame, eye_result):
    for x, y in eye_result.left_points + eye_result.right_points:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)


def draw_head_orientation(frame, attention_result):
    cv2.line(frame, attention_result.nose_point_2d, attention_result.pose_axis_2d, (255, 0, 0), 2)
    text = f"yaw:{attention_result.yaw:0.1f} pitch:{attention_result.pitch:0.1f}"
    cv2.putText(
        frame, text,
        (attention_result.nose_point_2d[0] - 60, attention_result.nose_point_2d[1] + 30),
        FONT, 0.45, (255, 0, 0), 1, cv2.LINE_AA,
    )


def draw_phone_boxes(frame, boxes, show_confidence=True):
    for box in boxes:
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"phone {box.confidence:0.2f}" if show_confidence else "phone"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)), FONT, 0.5, (0, 0, 255), 2, cv2.LINE_AA)


def draw_face_bbox(frame, bbox, color=(80, 200, 80)):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


def draw_debug_panel(frame, lines):
    h, w = frame.shape[:2]
    y = h - 12 - 18 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y - 20), (330, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, y + i * 18), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
