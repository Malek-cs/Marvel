HOLD_FRAMES = 10

class EmotionDetector:
    """
    Classifies face emotion from ratio dict produced by FaceDetector.get_ratios().
    Runs entirely on pre-computed values — no extra MediaPipe calls.
    """

    HAPPY_SMILE      = 0.40   # smile_ratio above this → happy
    SURPRISED_LIP    = 0.08   # lip_gap above this → part of surprised
    SURPRISED_BROW   = 0.28   # brow_avg y below this (brows raised) → surprised
    ANGRY_BROW       = 0.36   # brow_avg y above this (brows lowered) → angry
    ANGRY_SMILE_MAX  = 0.32   # smile_ratio must be below this for angry

    def __init__(self):
        self._counter = {}

    def classify(self, ratios: dict) -> str | None:
        if not ratios:
            self._counter.clear()
            return None

        smile  = ratios.get("smile_ratio", 0)
        lip    = ratios.get("lip_gap", 0)
        brow   = ratios.get("brow_avg_y", 0.33)

        if lip > self.SURPRISED_LIP and brow < self.SURPRISED_BROW:
            return self._confirm("surprised")
        if brow > self.ANGRY_BROW and smile < self.ANGRY_SMILE_MAX:
            return self._confirm("angry")
        if smile > self.HAPPY_SMILE:
            return self._confirm("happy")

        self._counter.clear()
        return None

    def _confirm(self, name: str) -> str | None:
        self._counter[name] = self._counter.get(name, 0) + 1
        for k in list(self._counter):
            if k != name:
                self._counter[k] = 0
        return name if self._counter[name] >= HOLD_FRAMES else None
