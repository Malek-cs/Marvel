import cv2
import mediapipe as mp
import numpy as np

HOLD_FRAMES = 8

# FaceMesh landmark indices
LEFT_EYE_TOP     = 159
LEFT_EYE_BOTTOM  = 145
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
LEFT_EYE_OUTER   = 33
LEFT_EYE_INNER   = 133
RIGHT_EYE_OUTER  = 362
RIGHT_EYE_INNER  = 263
UPPER_LIP        = 13
LOWER_LIP        = 14
TONGUE_TIP       = 17
NOSE_TIP         = 1
FOREHEAD         = 10

WINK_THRESHOLD   = 0.15   # EAR below this = eye closed
TONGUE_THRESHOLD = 0.04
LOOK_THRESHOLD   = 0.50   # nose offset ratio to trigger look_right

class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._counter = {}
        self.mouth_center    = None
        self.nose_center     = None
        self.forehead_center = None

    def detect(self, rgb_frame, draw_frame):
        result = self.face_mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            self._counter.clear()
            self.mouth_center    = None
            self.nose_center     = None
            self.forehead_center = None
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

        look_dir = self._head_yaw(lm)

        # live debug: show yaw value on frame so you can tune LOOK_THRESHOLD
        cv2.putText(draw_frame, f"yaw:{look_dir:+.2f} thr:{LOOK_THRESHOLD}",
                    (10, draw_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        return self._classify(left_ear, right_ear, lip_gap, tongue_y, lm, look_dir)

    def _ear(self, top, bottom, outer, inner):
        vertical   = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(outer - inner)
        return vertical / (horizontal + 1e-6)

    def _head_yaw(self, lm):
        # nose x relative to midpoint between both eye outer corners (normalised 0-1)
        nose_x   = lm[NOSE_TIP].x
        center_x = (lm[LEFT_EYE_OUTER].x + lm[RIGHT_EYE_OUTER].x) / 2
        eye_width = abs(lm[RIGHT_EYE_OUTER].x - lm[LEFT_EYE_OUTER].x)
        offset    = (nose_x - center_x) / (eye_width + 1e-6)
        # on a mirrored frame: positive offset = user looking to their RIGHT
        return offset

    def _classify(self, left_ear, right_ear, lip_gap, tongue_y, lm, look_dir):
        if left_ear < WINK_THRESHOLD and right_ear > WINK_THRESHOLD:
            return self._confirm("wink_left")
        if right_ear < WINK_THRESHOLD and left_ear > WINK_THRESHOLD:
            return self._confirm("wink_right")
        if lip_gap > TONGUE_THRESHOLD and tongue_y > lm[LOWER_LIP].y:
            return self._confirm("tongue_out")
        self._counter.clear()
        return None

    def _confirm(self, name):
        self._counter[name] = self._counter.get(name, 0) + 1
        for k in list(self._counter):
            if k != name:
                self._counter[k] = 0
        return name if self._counter[name] >= HOLD_FRAMES else None
