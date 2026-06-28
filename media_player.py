import cv2
import os

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

class MediaPlayer:
    def __init__(self, path):
        ext = os.path.splitext(path)[1].lower()
        self._is_video = ext in VIDEO_EXTS

        if self._is_video:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
        else:
            self._frame = cv2.imread(path)
            if self._frame is None:
                raise RuntimeError(f"Cannot load image: {path}")

    def get_frame(self):
        if self._is_video:
            ret, frame = self._cap.read()
            if not ret:                             # loop back to start
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            return frame if ret else None
        else:
            return self._frame.copy()

    def reset(self):
        if self._is_video:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self._is_video and hasattr(self, '_cap'):
            self._cap.release()
