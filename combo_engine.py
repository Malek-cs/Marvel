import time

# Define combos: tuple of gesture names in order → combo label
COMBO_MAP = {
    ("thumbs_up",   "tongue_out"):   "combo_victory",
    ("hand_on_nose","tongue_out"):   "combo_cat",
    ("hand_on_head","two_hands"):    "combo_ko",
    ("smile",       "open_mouth"):   "combo_wow",
}

COMBO_WINDOW = 2.5   # seconds within which gestures must occur
COMBO_SHOW   = 3.0   # seconds to display combo result

class ComboEngine:
    def __init__(self):
        self._history   = []          # [(timestamp, gesture), ...]
        self._last      = None
        self._fired_at  = None
        self._fired_name = None

    def update(self, gesture: str | None) -> str | None:
        now = time.monotonic()

        # Only record leading edge of each new gesture
        if gesture and gesture != self._last:
            self._history.append((now, gesture))
        self._last = gesture

        # Prune old entries
        self._history = [(t, g) for t, g in self._history if now - t < COMBO_WINDOW]

        seq = tuple(g for _, g in self._history)
        for pattern, name in COMBO_MAP.items():
            n = len(pattern)
            if len(seq) >= n and seq[-n:] == pattern:
                self._history.clear()
                self._last = None
                self._fired_at   = now
                self._fired_name = name
                return name

        return None

    def active_combo(self) -> str | None:
        """Returns the active combo name if still within display window."""
        if self._fired_at and time.monotonic() - self._fired_at < COMBO_SHOW:
            return self._fired_name
        return None
