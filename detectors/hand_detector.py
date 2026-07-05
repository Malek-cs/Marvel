import mediapipe as mp
import numpy as np

HOLD_FRAMES   = 8
NOSE_RADIUS   = 75
HAIR_SLACK_PX = 50

WRIST      = 0
THUMB_TIP  = 4
THUMB_IP   = 3
THUMB_MCP  = 2
INDEX_TIP  = 8
MIDDLE_TIP = 12
RING_TIP   = 16
PINKY_TIP  = 20
INDEX_MCP  = 5
MIDDLE_MCP = 9
RING_MCP   = 13
PINKY_MCP  = 17

_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._counter      = {}
        self.last_landmarks = None   # exposed for trainer

    def detect(self, rgb_frame, draw_frame, nose_center=None, forehead_center=None):
        result = self.hands.process(rgb_frame)

        if not result.multi_hand_landmarks:
            self._counter.clear()
            self.last_landmarks = None
            return None

        if len(result.multi_hand_landmarks) == 2:
            self.last_landmarks = result.multi_hand_landmarks[0]
            return self._confirm("two_hands")

        lm = result.multi_hand_landmarks[0].landmark
        self.last_landmarks = result.multi_hand_landmarks[0]
        h, w = draw_frame.shape[:2]

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        tips_px = [pt(t) for t in _TIPS]

        # Hand on hair: any fingertip above the forehead line
        if forehead_center is not None:
            hair_line_y = forehead_center[1] + HAIR_SLACK_PX
            if any(tp[1] < hair_line_y for tp in tips_px):
                return self._confirm("hand_on_head")

        # Hand on nose
        if nose_center is not None:
            if any(np.linalg.norm(tp - nose_center) < NOSE_RADIUS for tp in tips_px):
                return self._confirm("hand_on_nose")

        # Thumbs up (dedicated check — more accurate)
        if self._is_thumbs_up(lm):
            return self._confirm("thumbs_up")

        fingers = self._fingers_up(lm)
        return self._classify(fingers)

    def _fingers_up(self, lm):
        tips  = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        bases = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        return [lm[t].y < lm[b].y for t, b in zip(tips, bases)]

    def _is_thumbs_up(self, lm) -> bool:
        thumb_up      = lm[THUMB_TIP].y < lm[THUMB_MCP].y - 0.04
        fingers_down  = all(lm[t].y > lm[b].y
                            for t, b in zip(
                                [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP],
                                [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]))
        above_wrist   = lm[THUMB_TIP].y < lm[WRIST].y
        return thumb_up and fingers_down and above_wrist

    def _classify(self, fingers):
        index, middle, ring, pinky = fingers

        if not any(fingers):
            return self._confirm("fist")
        if all(fingers):
            return self._confirm("open_palm")
        if index and middle and not ring and not pinky:
            return self._confirm("peace")
        if index and not middle and not ring and pinky:
            return self._confirm("spiderman_web")

        self._counter.clear()
        return None

    def _confirm(self, name):
        self._counter[name] = self._counter.get(name, 0) + 1
        for k in list(self._counter):
            if k != name:
                self._counter[k] = 0
        return name if self._counter[name] >= HOLD_FRAMES else None
