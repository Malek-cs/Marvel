import cv2
import numpy as np

CAM_W  = 640
CAM_H  = 480
REF_W  = 640
REF_H  = 480

# PIP size in overlay mode (bottom-right corner)
PIP_H  = CAM_H // 3
PIP_PAD = 10


def build_combined_frame(
    cam_frame,
    ref_frame,
    active_gesture: str | None = None,
    overlay_mode: bool = False,
    status_lines: list | None = None,
) -> np.ndarray:

    cam = cv2.resize(cam_frame, (CAM_W, CAM_H))

    if overlay_mode:
        combined = _overlay(cam, ref_frame)
    else:
        combined = _side_by_side(cam, ref_frame)

    _draw_gesture_label(combined, active_gesture)
    _draw_status(combined, status_lines or [])

    return combined


def show_dual_window(combined: np.ndarray):
    cv2.imshow("AR Gesture Reactor", combined)


# ── Layout builders ──────────────────────────────────────────────────────────

def _side_by_side(cam, ref_frame) -> np.ndarray:
    if ref_frame is not None:
        ref = _fit(ref_frame, REF_W, REF_H)
    else:
        ref = _placeholder(REF_W, REF_H, "Show a gesture...")

    div      = np.full((CAM_H, 4, 3), 50, dtype=np.uint8)
    return np.hstack([cam, div, ref])


def _overlay(cam, ref_frame) -> np.ndarray:
    combined = cam.copy()

    if ref_frame is not None:
        ratio = PIP_H / ref_frame.shape[0]
        pw    = int(ref_frame.shape[1] * ratio)
        pip   = cv2.resize(ref_frame, (pw, PIP_H))

        x = CAM_W - pw  - PIP_PAD
        y = CAM_H - PIP_H - PIP_PAD

        # semi-transparent dark background
        roi = combined[y:y+PIP_H, x:x+pw]
        bg  = (roi * 0.4).astype(np.uint8)
        combined[y:y+PIP_H, x:x+pw] = bg
        combined[y:y+PIP_H, x:x+pw] = cv2.addWeighted(pip, 0.9, bg, 0.1, 0)

        cv2.rectangle(combined, (x-2, y-2), (x+pw+2, y+PIP_H+2), (0,255,120), 1)

    # extend canvas to match side-by-side width
    pad = np.zeros((CAM_H, REF_W + 4, 3), dtype=np.uint8)
    cv2.putText(pad, "OVERLAY MODE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    return np.hstack([combined, pad])


# ── HUD helpers ──────────────────────────────────────────────────────────────

def _draw_gesture_label(frame, gesture: str | None):
    if not gesture:
        return
    label = gesture.replace("_", " ").upper()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.rectangle(frame, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 120), 2)


def _draw_status(frame, lines: list):
    y = CAM_H - 10
    for line in reversed(lines):
        color = (0, 0, 220) if line.startswith("REC") else \
                (0, 215, 255) if line.startswith("COMBO") else \
                (0, 165, 255) if line.startswith("TRAIN") else \
                (80, 200, 120)
        if line.startswith("REC"):
            cv2.circle(frame, (frame.shape[1] - 18, 18), 8, (0, 0, 220), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 60, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
        else:
            cv2.putText(frame, line, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y -= 22


def _fit(frame, tw, th) -> np.ndarray:
    fh, fw = frame.shape[:2]
    scale  = min(tw / fw, th / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas  = np.zeros((th, tw, 3), dtype=np.uint8)
    x = (tw - nw) // 2
    y = (th - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas


def _placeholder(w, h, text) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(canvas, text, (w // 2 - 130, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 70, 70), 2)
    return canvas
