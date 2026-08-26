"""
Non-blocking audio alerts.

Responsibilities:
  - Alternate between the configured alert sound files in round-robin order
    (e.g. with 3 files: alert1 -> alert2 -> alert3 -> alert1 -> ...).
  - Never play two sounds at once.
  - Enforce a cooldown between plays so a persisting distraction doesn't
    spam the speaker every frame.
  - Never block the webcam/detection loop -- playback happens via pygame's
    mixer, which runs on its own thread/hardware channel.
"""

import os
import time

import pygame


class AlertManager:
    def __init__(self, sound_files, cooldown):
        self.sound_files = list(sound_files)
        self.cooldown = cooldown
        self._index = 0
        self._last_played_at = 0.0
        self.distraction_count = 0
        self.last_reason = None
        self.last_sound_file = None

        self._mixer_ready = False
        try:
            pygame.mixer.init()
            self._mixer_ready = True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[AlertManager] Audio device unavailable, alerts will be silent: {exc}")

        missing = [f for f in self.sound_files if not os.path.exists(f)]
        if missing:
            print("[AlertManager] Warning: missing alert sound file(s):")
            for f in missing:
                print(f"    {f}")
            print("[AlertManager] Alerts referencing missing files will be skipped silently.")

    def _is_busy(self):
        if not self._mixer_ready:
            return False
        try:
            return pygame.mixer.music.get_busy()
        except Exception:
            return False

    def cooldown_remaining(self):
        remaining = self.cooldown - (time.time() - self._last_played_at)
        return max(0.0, remaining)

    def is_ready_to_play(self):
        return not self._is_busy() and self.cooldown_remaining() <= 0.0

    def notify_distraction(self, reason):
        """
        Register that a distraction condition was just confirmed (rising
        edge). Always increments the session distraction counter. Attempts
        to play the next alert sound in rotation, subject to the cooldown
        and "never overlap" rules. Returns True if a sound actually started
        playing.
        """
        self.distraction_count += 1
        self.last_reason = reason

        if not self._mixer_ready:
            return False
        if not self.is_ready_to_play():
            return False

        sound_file = self.sound_files[self._index % len(self.sound_files)]
        if not os.path.exists(sound_file):
            # Skip this slot but still advance rotation so a missing file
            # doesn't permanently jam the alternation order.
            self._index += 1
            return False

        try:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        except Exception as exc:
            print(f"[AlertManager] Failed to play {sound_file}: {exc}")
            return False

        self._index += 1
        self._last_played_at = time.time()
        self.last_sound_file = os.path.basename(sound_file)
        return True

    def shutdown(self):
        if self._mixer_ready:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
