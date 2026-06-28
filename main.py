import cv2
from detectors.face_detector import FaceDetector
from detectors.hand_detector import HandDetector
from matcher import GestureMatcher
from display import show_dual_window

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    face    = FaceDetector()
    hand    = HandDetector()
    matcher = GestureMatcher("assets/")

    active_player = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_gesture = face.detect(rgb, frame)
        hand_gesture = hand.detect(rgb, frame, nose_center=face.nose_center, forehead_center=face.forehead_center)

        active_gesture = face_gesture or hand_gesture

        if active_gesture:
            active_player = matcher.get_player(active_gesture)
        else:
            if active_player is not None:
                active_player.reset()
            active_player = None

        ref_frame = active_player.get_frame() if active_player else None

        show_dual_window(frame, ref_frame, active_gesture)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    matcher.release_all()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
