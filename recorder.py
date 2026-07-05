import cv2
import os
from datetime import datetime

class ScreenRecorder:
    FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
    FPS    = 20.0

    def __init__(self, output_dir: str = "recordings"):
        self.output_dir    = output_dir
        self.is_recording  = False
        self.current_file  = None
        self._writer       = None

    def start(self, width: int, height: int):
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = os.path.join(self.output_dir, f"rec_{ts}.mp4")
        self._writer = cv2.VideoWriter(
            self.current_file, self.FOURCC, self.FPS, (width, height))
        self.is_recording = True
        print(f"[recorder] Started → {self.current_file}")

    def write(self, frame):
        if self.is_recording and self._writer:
            self._writer.write(frame)

    def stop(self):
        if self._writer:
            self._writer.release()
            self._writer = None
        self.is_recording = False
        print(f"[recorder] Saved → {self.current_file}")
