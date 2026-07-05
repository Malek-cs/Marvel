import os
import pickle
import numpy as np

SAVE_PATH      = "trained_gestures/model.pkl"
REQUIRED_FRAMES = 60

class GestureTrainer:
    def __init__(self):
        self._samples: dict[str, list] = {}   # {name: [vec, ...]}
        self._capturing_name  = None
        self._capture_buffer  = []
        self._load()

    # ── Capture ─────────────────────────────────────────────────────────────

    def start_capture(self, name: str):
        self._capturing_name = name
        self._capture_buffer = []
        print(f"[trainer] Capturing '{name}' — hold the pose...")

    def feed(self, hand_landmarks) -> int:
        """Call each frame during capture. Returns frames captured so far."""
        if hand_landmarks is None or self._capturing_name is None:
            return len(self._capture_buffer)
        vec = self._to_vec(hand_landmarks)
        self._capture_buffer.append(vec)
        return len(self._capture_buffer)

    def is_complete(self) -> bool:
        return len(self._capture_buffer) >= REQUIRED_FRAMES

    def save_and_train(self) -> str:
        """Persist samples and retrain. Returns gesture name saved."""
        name = self._capturing_name
        existing = self._samples.get(name, [])
        self._samples[name] = existing + self._capture_buffer
        self._capture_buffer = []
        self._capturing_name = None
        self._persist()
        return name

    # ── Internal ─────────────────────────────────────────────────────────────

    def _to_vec(self, hand_landmarks) -> np.ndarray:
        pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
        pts -= pts[0]                          # wrist at origin
        scale = np.max(np.abs(pts)) + 1e-6
        return (pts / scale).flatten()         # 42-dim normalised

    def _persist(self):
        os.makedirs("trained_gestures", exist_ok=True)
        with open(SAVE_PATH, "wb") as f:
            pickle.dump(self._samples, f)
        print(f"[trainer] Saved {len(self._samples)} gesture(s) to {SAVE_PATH}")

    def _load(self):
        if os.path.exists(SAVE_PATH):
            with open(SAVE_PATH, "rb") as f:
                self._samples = pickle.load(f)
            print(f"[trainer] Loaded samples: {list(self._samples)}")

    @property
    def samples(self) -> dict:
        return self._samples
