import threading
import cv2

from app_state import AppState
from detectors.face_detector import FaceDetector
from detectors.hand_detector import HandDetector
from emotion_detector import EmotionDetector
from combo_engine import ComboEngine
from matcher import GestureMatcher
from media_player import MediaPlayer
from display import build_combined_frame, show_dual_window
from recorder import ScreenRecorder
from trainer.gesture_trainer import GestureTrainer
from trainer.custom_detector import CustomGestureDetector
from gui.config_panel import ConfigPanel
from gui.dashboard_server import DashboardServer


def _ask_gesture_name(state: AppState):
    name = input("Enter gesture name to train: ").strip()
    if name:
        state.training_name   = name
        state.training_active = True


def main():
    state     = AppState()
    face      = FaceDetector()
    hand      = HandDetector()
    emotion   = EmotionDetector()
    combo     = ComboEngine()
    matcher   = GestureMatcher("assets/", state)
    recorder  = ScreenRecorder("recordings/")
    trainer   = GestureTrainer()
    custom    = CustomGestureDetector()
    panel     = ConfigPanel(state, "assets/")
    dashboard = DashboardServer(state, port=5000)
    dashboard.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    active_player: MediaPlayer | None = None
    combo_player:  MediaPlayer | None = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Detection ────────────────────────────────────────────────────────
        face_gesture = face.detect(rgb, frame)
        hand_gesture = hand.detect(rgb, frame,
                                   nose_center=face.nose_center,
                                   forehead_center=face.forehead_center)

        emotion_label = emotion.classify(face.get_ratios())

        custom_gesture = (custom.detect(hand.last_landmarks)
                          if hand.last_landmarks else None)

        active_gesture = face_gesture or hand_gesture or custom_gesture
        state.current_gesture = active_gesture or ""

        # ── Training mode ────────────────────────────────────────────────────
        if state.training_active and hand.last_landmarks:
            count = trainer.feed(hand.last_landmarks)
            state.training_count = count
            if trainer.is_complete():
                trainer.save_and_train()
                custom.reload()
                state.training_active = False
                state.training_count  = 0
                print("[main] Training complete!")

        # ── Combo engine ─────────────────────────────────────────────────────
        new_combo = combo.update(active_gesture)
        if new_combo:
            combo_player       = matcher.get_player(new_combo)
            state.current_combo = new_combo

        live_combo = combo.active_combo()
        if not live_combo:
            combo_player       = None
            state.current_combo = ""

        # ── Media selection ──────────────────────────────────────────────────
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

        # ── Build frame ──────────────────────────────────────────────────────
        status: list[str] = []
        if state.recording:
            status.append("REC")
        if state.training_active:
            status.append(f"TRAIN {state.training_count}/60 — {state.training_name}")
        if state.current_combo:
            status.append(f"COMBO: {state.current_combo}")
        if emotion_label:
            status.append(f"EMOTION: {emotion_label}")
            dashboard.set_emotion(emotion_label)
        else:
            dashboard.set_emotion(None)

        combined = build_combined_frame(
            frame, ref_frame, active_gesture,
            overlay_mode=state.overlay_mode,
            status_lines=status,
        )
        state.combined_frame = combined

        # ── Output ───────────────────────────────────────────────────────────
        show_dual_window(combined)
        if state.recording:
            recorder.write(combined)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("r"), ord("R")):
            if not state.recording:
                h, w = combined.shape[:2]
                recorder.start(w, h)
                state.recording = True
            else:
                recorder.stop()
                state.recording = False
        elif key in (ord("o"), ord("O")):
            state.overlay_mode = not state.overlay_mode
        elif key in (ord("g"), ord("G")):
            panel.launch()
        elif key in (ord("t"), ord("T")):
            if not state.training_active:
                threading.Thread(target=_ask_gesture_name,
                                 args=(state,), daemon=True).start()

    recorder.stop()
    matcher.release_all()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
