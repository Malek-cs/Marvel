import mediapipe as mp
import numpy as np

HOLD_FRAMES = 8

# Eye landmarks
LEFT_EYE_TOP     = 159
LEFT_EYE_BOTTOM  = 145
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
LEFT_EYE_OUTER   = 33
LEFT_EYE_INNER   = 133
RIGHT_EYE_OUTER  = 362
RIGHT_EYE_INNER  = 263

# Mouth & face
UPPER_LIP    = 13
LOWER_LIP    = 14
TONGUE_TIP   = 17
MOUTH_LEFT   = 78     # left lip commissure
MOUTH_RIGHT  = 308    # right lip commissure
CHEEK_LEFT   = 234    # left cheekbone outer
CHEEK_RIGHT  = 454    # right cheekbone outer

# Nose & head
NOSE_TIP  = 1
FOREHEAD  = 10

# Brows (for emotion)
BROW_LEFT  = 70
BROW_RIGHT = 300

# Thresholds
WINK_THRESHOLD    = 0.15
TONGUE_THRESHOLD  = 0.04
SMILE_THRESHOLD   = 0.38   # mouth_width / face_width → smile
OPEN_MOUTH_TH     = 0.06   # lip_gap / h → open mouth


class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._counter        = {}
        self.mouth_center    = None
        self.nose_center     = None
        self.forehead_center = None
        self._ratios         = None   # for EmotionDetector

    def detect(self, rgb_frame, draw_frame):
        result = self.face_mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            self._counter.clear()
            self.mouth_center = self.nose_center = self.forehead_center = None
            self._ratios = None
            return None

        lm = result.multi_face_landmarks[0].landmark
        h, w = draw_frame.shape[:2]

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        self.mouth_center    = (pt(UPPER_LIP) + pt(LOWER_LIP)) / 2
        self.nose_center     = pt(NOSE_TIP)
        self.forehead_center = pt(FOREHEAD)

        left_ear  = self._ear(pt(LEFT_EYE_TOP),  pt(LEFT_EYE_BOTTOM),
                              pt(LEFT_EYE_OUTER), pt(LEFT_EYE_INNER))
        right_ear = self._ear(pt(RIGHT_EYE_TOP), pt(RIGHT_EYE_BOTTOM),
                              pt(RIGHT_EYE_OUTER), pt(RIGHT_EYE_INNER))

        lip_gap  = np.linalg.norm(pt(UPPER_LIP) - pt(LOWER_LIP)) / h
        tongue_y = lm[TONGUE_TIP].y

        mouth_w  = abs(lm[MOUTH_RIGHT].x - lm[MOUTH_LEFT].x)
        face_w   = abs(lm[CHEEK_RIGHT].x  - lm[CHEEK_LEFT].x)  + 1e-6
        smile_ratio = mouth_w / face_w

        brow_avg = (lm[BROW_LEFT].y + lm[BROW_RIGHT].y) / 2

        self._ratios = {
            "smile_ratio": smile_ratio,
            "lip_gap":     lip_gap,
            "brow_avg_y":  brow_avg,
            "left_ear":    left_ear,
            "right_ear":   right_ear,
        }

        return self._classify(left_ear, right_ear, lip_gap, tongue_y,
                               lm, smile_ratio)

    def get_ratios(self) -> dict | None:
        return self._ratios

    def _ear(self, top, bottom, outer, inner):
        return np.linalg.norm(top - bottom) / (np.linalg.norm(outer - inner) + 1e-6)

    def _classify(self, left_ear, right_ear, lip_gap, tongue_y, lm, smile_ratio):
        if left_ear < WINK_THRESHOLD and right_ear > WINK_THRESHOLD:
            return self._confirm("wink_left")
        if right_ear < WINK_THRESHOLD and left_ear > WINK_THRESHOLD:
            return self._confirm("wink_right")
        if lip_gap > TONGUE_THRESHOLD and tongue_y > lm[LOWER_LIP].y:
            return self._confirm("tongue_out")
        if lip_gap > OPEN_MOUTH_TH:
            return self._confirm("open_mouth")
        if smile_ratio > SMILE_THRESHOLD:
            return self._confirm("smile")

        self._counter.clear()
        return None

    def _confirm(self, name):
        self._counter[name] = self._counter.get(name, 0) + 1
        for k in list(self._counter):
            if k != name:
                self._counter[k] = 0
        return name if self._counter[name] >= HOLD_FRAMES else None
