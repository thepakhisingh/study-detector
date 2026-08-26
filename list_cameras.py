"""
Utility: probe camera indices 0-9 and show a live preview of each so you
can identify which index corresponds to your real webcam, then set that
number as CAMERA_INDEX in config.py.

Run: venv\\Scripts\\python.exe list_cameras.py
Press any key to move to the next camera index, 'q' to stop early.
"""

import cv2

MAX_INDEX_TO_PROBE = 10


def main():
    found_any = False
    for index in range(MAX_INDEX_TO_PROBE):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue

        found_any = True
        print(f"Camera index {index}: OK ({frame.shape[1]}x{frame.shape[0]}). "
              f"Showing preview -- press any key for next, 'q' to quit.")

        window = f"Camera index {index} - press any key for next"
        cv2.imshow(window, frame)
        for _ in range(150):  # keep refreshing for ~5s so you can see motion
            ok, frame = cap.read()
            if ok:
                cv2.imshow(window, frame)
            key = cv2.waitKey(33) & 0xFF
            if key != 255:
                break
        cv2.destroyWindow(window)
        cap.release()

        if key == ord("q"):
            break

    if not found_any:
        print("No cameras found on indices 0-9.")
    else:
        print("\nSet CAMERA_INDEX in config.py to whichever index showed your real webcam.")


if __name__ == "__main__":
    main()
