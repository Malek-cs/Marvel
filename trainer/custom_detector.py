import os
import pickle
import numpy as np

SAVE_PATH   = "trained_gestures/model.pkl"
MIN_CONF    = 0.60   # minimum KNN confidence to fire
HOLD_FRAMES = 8

class CustomGestureDetector:
    def __init__(self):
        self._model     = None
        self._classes   = []
        self._counter   = {}
        self.reload()

    def reload(self):
        """Call after training completes to pick up new data."""
        if not os.path.exists(SAVE_PATH):
            return
        try:
            from sklearn.neighbors import KNeighborsClassifier
            with open(SAVE_PATH, "rb") as f:
                samples: dict = pickle.load(f)
            if len(samples) < 2:
                print("[custom] Need at least 2 trained gestures to classify.")
                return
            X, y = [], []
            for name, vecs in samples.items():
                for v in vecs:
                    X.append(v)
                    y.append(name)
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X, y)
            self._model   = knn
            self._classes = list(samples.keys())
            print(f"[custom] Model ready: {self._classes}")
        except Exception as e:
            print(f"[custom] Load error: {e}")

    def detect(self, hand_landmarks) -> str | None:
        if self._model is None or hand_landmarks is None:
            return None
        vec   = self._to_vec(hand_landmarks).reshape(1, -1)
        proba = self._model.predict_proba(vec)[0]
        best  = int(np.argmax(proba))
        if proba[best] < MIN_CONF:
            self._counter.clear()
            return None
        name = self._model.classes_[best]
        return self._confirm(name)

    def _to_vec(self, hand_landmarks) -> np.ndarray:
        pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
        pts -= pts[0]
        scale = np.max(np.abs(pts)) + 1e-6
        return (pts / scale).flatten()

    def _confirm(self, name: str) -> str | None:
        self._counter[name] = self._counter.get(name, 0) + 1
        for k in list(self._counter):
            if k != name:
                self._counter[k] = 0
        return name if self._counter[name] >= HOLD_FRAMES else None
