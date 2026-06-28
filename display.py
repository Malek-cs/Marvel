import cv2
import numpy as np

CAM_W    = 640
CAM_H    = 480
REF_W    = 640   # wider panel so video isn't squeezed
REF_H    = 480

def _fit_frame(frame, target_w, target_h):
    """Scale frame to fill target size while keeping aspect ratio; pad with black."""
    fh, fw = frame.shape[:2]
    scale  = min(target_w / fw, target_h / fh)
    new_w  = int(fw * scale)
    new_h  = int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off  = (target_w - new_w) // 2
    y_off  = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

def show_dual_window(cam_frame, ref_frame, active_gesture=None):
    cam = cv2.resize(cam_frame, (CAM_W, CAM_H))

    if ref_frame is not None:
        ref = _fit_frame(ref_frame, REF_W, REF_H)
    else:
        ref = np.zeros((REF_H, REF_W, 3), dtype=np.uint8)
        cv2.putText(ref, "Show a gesture...", (REF_W // 2 - 130, REF_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

    divider  = np.full((CAM_H, 4, 3), 60, dtype=np.uint8)
    combined = np.hstack([cam, divider, ref])

    if active_gesture:
        label = f"  {active_gesture.replace('_', ' ').upper()}  "
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(combined, (0, 0), (tw + 12, th + 16), (0, 0, 0), -1)
        cv2.putText(combined, label, (6, th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)

    cv2.imshow("AR Pose Matcher", combined)
