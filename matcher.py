import os
from media_player import MediaPlayer

GESTURE_VIDEO_MAP = {
    "tongue_out":   "Cat.jpg",
    "hand_on_nose": "catnose.jpg",
    "hand_on_head": "fight.mp4",
    "thumbs_up":    "fight.mp4",
    "two_hands":    "look_right.mp4",
}

class GestureMatcher:
    def __init__(self, assets_dir):
        self.assets_dir = assets_dir
        self._players: dict[str, MediaPlayer] = {}

    def get_player(self, gesture_name) -> MediaPlayer | None:
        filename = GESTURE_VIDEO_MAP.get(gesture_name)
        if not filename:
            return None

        if filename not in self._players:
            path = os.path.join(self.assets_dir, filename)
            if not os.path.exists(path):
                print(f"[matcher] Missing: {path}")
                return None
            self._players[filename] = MediaPlayer(path)

        return self._players[filename]

    def release_all(self):
        for player in self._players.values():
            player.release()
        self._players.clear()
