import threading
import numpy as np

class AppState:
    def __init__(self):
        self.overlay_mode      = False
        self.recording         = False
        self.training_active   = False
        self.training_name     = ""
        self.training_count    = 0

        self.current_gesture   = ""
        self.current_combo     = ""
        self.combined_frame    = None   # latest rendered frame (for dashboard)

        self._map_lock         = threading.Lock()
        self.gesture_map       = {}     # seeded by matcher on init

    def update_mapping(self, gesture: str, filename: str):
        with self._map_lock:
            self.gesture_map[gesture] = filename

    def get_mapping(self) -> dict:
        with self._map_lock:
            return dict(self.gesture_map)
