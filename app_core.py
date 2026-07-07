"""
Headless detection loop — runs in a background thread.
No cv2.imshow, no cv2.waitKey. All controls come from AppState (set by the dashboard).
"""

import cv2

from app_state import AppState
from detectors.face_detector import FaceDetector
from detectors.hand_detector import HandDetector
from emotion_detector import EmotionDetector
from combo_engine import ComboEngine
from matcher import GestureMatcher
from media_player import MediaPlayer
from display import build_combined_frame
from recorder import ScreenRecorder
from trainer.gesture_trainer import GestureTrainer
from trainer.custom_detector import CustomGestureDetector


def run_loop(state: AppState):
    face     = FaceDetector()
    hand     = HandDetector()
    emotion  = EmotionDetector()
    combo    = ComboEngine()
    matcher  = GestureMatcher("assets/", state)
    recorder = ScreenRecorder("recordings/")
    trainer  = GestureTrainer()
    custom   = CustomGestureDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[core] ERROR: Cannot open webcam")
        state.cam_error = True
        return

    active_player: MediaPlayer | None = None
    combo_player:  MediaPlayer | None = None

    while cap.isOpened() and not state.stop_requested:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detection
        face_gesture  = face.detect(rgb, frame)
        hand_gesture  = hand.detect(rgb, frame,
                                    nose_center=face.nose_center,
                                    forehead_center=face.forehead_center)
        emotion_label = emotion.classify(face.get_ratios())
        custom_gesture = (custom.detect(hand.last_landmarks)
                          if hand.last_landmarks else None)

        active_gesture         = face_gesture or hand_gesture or custom_gesture
        state.current_gesture  = active_gesture or ""
        state.current_emotion  = emotion_label or ""

        # Training
        if state.training_active and hand.last_landmarks:
            if not trainer._capturing_name:
                trainer.start_capture(state.training_name)
            count = trainer.feed(hand.last_landmarks)
            state.training_count = count
            if trainer.is_complete():
                trainer.save_and_train()
                custom.reload()
                state.training_active = False
                state.training_count  = 0

        # Combos
        new_combo = combo.update(active_gesture)
        if new_combo:
            combo_player        = matcher.get_player(new_combo)
            state.current_combo = new_combo
        if not combo.active_combo():
            combo_player        = None
            state.current_combo = ""

        # Media selection
        if combo_player:
            display_player = combo_player
        elif active_gesture:
            display_player = matcher.get_player(active_gesture)
            if display_player is not None:
                active_player = display_player
            else:
                display_player = active_player
        else:
            if active_player:
                active_player.reset()
            active_player  = None
            display_player = None

        ref_frame = display_player.get_frame() if display_player else None

        # Build status lines
        status = []
        if state.recording:
            status.append("REC")
        if state.training_active:
            status.append(f"TRAIN {state.training_count}/60 — {state.training_name}")
        if state.current_combo:
            status.append(f"COMBO: {state.current_combo}")
        if emotion_label:
            status.append(f"EMOTION: {emotion_label}")

        combined = build_combined_frame(
            frame, ref_frame, active_gesture,
            overlay_mode=state.overlay_mode,
            status_lines=status,
        )
        state.combined_frame = combined

        # Recording
        if state.recording:
            if not recorder.is_recording:
                h, w = combined.shape[:2]
                recorder.start(w, h)
            recorder.write(combined)
        elif recorder.is_recording:
            recorder.stop()

    recorder.stop()
    matcher.release_all()
    cap.release()
