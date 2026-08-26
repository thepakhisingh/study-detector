"""
Per-condition timers used to turn instantaneous per-frame detections (e.g.
"eyes are closed in this frame") into confirmed, debounced distraction
events (e.g. "eyes have been closed for 2+ continuous seconds").

Each tracked condition gets its own ConditionTimer so that, for example, an
in-progress LOOKING_AWAY timer is completely independent of the
EYES_CLOSED timer -- they start, accumulate and reset on their own.
"""

import time


class ConditionTimer:
    """Tracks how long a single boolean condition has been continuously true."""

    def __init__(self, duration):
        self.duration = duration
        self.start_time = None
        self.confirmed = False

    def update(self, active, now=None):
        """
        Feed the current truth value of the condition for this frame.

        Returns a tuple (elapsed_seconds, just_confirmed):
          - elapsed_seconds: how long the condition has been continuously
            true (0.0 if not currently active).
          - just_confirmed: True only on the single frame where the
            duration threshold was crossed (rising edge), so callers can
            fire a one-shot action (like playing an alert) instead of
            re-triggering every frame while the condition persists.
        """
        now = time.time() if now is None else now

        if not active:
            # Condition disappeared -- reset the timer entirely. A brief
            # flicker will simply restart counting from zero next time,
            # which is the desired "reset if it disappears" behaviour.
            self.start_time = None
            self.confirmed = False
            return 0.0, False

        if self.start_time is None:
            self.start_time = now

        elapsed = now - self.start_time

        if elapsed >= self.duration and not self.confirmed:
            self.confirmed = True
            return elapsed, True

        return elapsed, False

    def reset(self):
        self.start_time = None
        self.confirmed = False

    @property
    def is_active(self):
        return self.start_time is not None

    @property
    def is_confirmed(self):
        return self.confirmed


class TimerManager:
    """A named collection of ConditionTimer instances."""

    def __init__(self, durations):
        """durations: dict mapping condition name -> required duration (s)."""
        self._timers = {name: ConditionTimer(duration) for name, duration in durations.items()}

    def update(self, name, active, now=None):
        return self._timers[name].update(active, now=now)

    def elapsed(self, name):
        timer = self._timers[name]
        if timer.start_time is None:
            return 0.0
        return time.time() - timer.start_time

    def is_confirmed(self, name):
        return self._timers[name].is_confirmed

    def reset(self, name):
        self._timers[name].reset()

    def reset_all(self):
        for timer in self._timers.values():
            timer.reset()

    def names(self):
        return list(self._timers.keys())
