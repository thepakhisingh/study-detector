# Study Focus Detector

A real-time webcam application that watches whether you're focused on
studying and plays an alternating alert sound when you're distracted for
too long.

It detects five distraction conditions, each with its own independent
timer:

| Condition           | How it's detected                                            | Trigger threshold |
|---------------------|---------------------------------------------------------------|--------------------|
| `LOOKING_AWAY`       | Head pose (yaw/pitch via solvePnP) **and** iris position within the eye socket | ~3s continuous |
| `EYES_CLOSED`        | Eye Aspect Ratio (EAR) from MediaPipe FaceMesh landmarks       | ~2s continuous |
| `FACE_NOT_VISIBLE`   | No face detected (with a short grace period + optional hand-cover check) | ~2s continuous |
| `PHONE_DETECTED`     | YOLOv8 (Ultralytics) "cell phone" detections large enough to be in-hand/in-use | ~3s continuous |
| `FOCUSED`            | None of the above are active                                  | -- |

## Project structure

```
study-detector/
├── main.py                      # entry point / main loop & state machine
├── config.py                    # all tunable settings
├── requirements.txt
├── detectors/
│   ├── face_detector.py         # MediaPipe FaceMesh + Hands (face presence / cover)
│   ├── eye_detector.py          # EAR calculation (eyes closed)
│   ├── attention_detector.py    # head pose + iris gaze (looking away)
│   └── phone_detector.py        # YOLOv8 phone detection
├── utils/
│   ├── timer_manager.py         # per-condition debounce timers
│   ├── alert_manager.py         # non-blocking, alternating, cooldown-limited audio
│   └── drawing.py               # HUD / debug overlay rendering
├── models/
│   └── yolov8n.pt               # downloaded automatically on first run
└── audio/
    ├── alert1.mp3
    ├── alert2.mp3
    └── alert3.mp3
```

## Setup

A virtual environment has already been created and the dependencies from
`requirements.txt` installed into it (`venv/`).

To set it up again from scratch:

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

### Audio files

Place three alert sound files at:

```
audio/alert1.mp3
audio/alert2.mp3
audio/alert3.mp3
```

Alerts cycle through them round-robin (`alert1 -> alert2 -> alert3 ->
alert1 -> ...`) each time a distraction is confirmed and the cooldown has
elapsed. If a file is missing, that turn is skipped silently (no crash)
and the rotation continues to the next file.

## Running

```powershell
venv\Scripts\python.exe main.py
```

Controls:
- `q` or `Esc` -- quit
- `d` -- toggle the debug overlay (landmarks, head orientation, phone boxes, confidence values)

The first run will auto-download the small YOLOv8n weights file
(`models/yolov8n.pt`, ~6MB) from Ultralytics if it isn't present, which
requires an internet connection. Phone detection is automatically disabled
(with a printed warning) if the model can't be loaded -- the rest of the
detector keeps working normally.

## Configuration

All thresholds and durations live in `config.py`: camera index, EAR
threshold, looking-away thresholds (head pose + iris offset), eyes-closed
duration, face-missing duration + grace frames, phone confidence
threshold + minimum area ratio + duration, alert cooldown, and debug
flags. Nothing is hardcoded elsewhere.

## How the state machine works

Every frame, each of the four distraction conditions is evaluated as a
simple boolean (e.g. "are eyes closed right now?"). That boolean feeds its
own `ConditionTimer` (see `utils/timer_manager.py`):

1. While the condition is `True`, its timer accumulates elapsed time.
2. If the condition becomes `False` at any point, its timer resets to zero
   immediately (no partial credit carried over).
3. Once elapsed time crosses the condition's configured duration, it is
   "confirmed" exactly once (a rising edge) -- this is what triggers the
   alert manager, not every subsequent frame the condition keeps being
   true.
4. The alert manager then plays the next sound in the alternating
   rotation, provided the cooldown has elapsed and nothing is currently
   playing (sounds never overlap).

When several conditions are true in the same frame, the HUD shows the
highest-priority one (`FACE_NOT_VISIBLE > PHONE_DETECTED > LOOKING_AWAY >
EYES_CLOSED`), but each still has its own independent timer running
underneath.

## Limitations / things to verify manually with a webcam

- Threshold values in `config.py` (EAR, head pose angles, iris offsets)
  were chosen from commonly-used defaults but may need small tuning for
  your specific camera, lighting, and face.
- Phone detection quality depends on the lightweight `yolov8n` model and
  camera angle; a larger Ultralytics model (e.g. `yolov8s.pt`) can be
  swapped in via `config.PHONE_MODEL_PATH` for better accuracy at the cost
  of speed.
- Audio playback, camera access, and the live GUI window all require a
  real desktop session with a webcam and speakers -- these cannot be
  verified in an automated/headless environment and should be tested by
  actually running the app.
