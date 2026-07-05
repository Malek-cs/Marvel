import os
from media_player import MediaPlayer

DEFAULT_MAP = {
    # Face expressions
    "tongue_out":    "Cat.jpg",
    "hand_on_nose":  "catnose.jpg",
    "smile":         "smile.jpg",
    "open_mouth":    "open_mouth.jpg",
    # Emotions
    "happy":         "happy.mp4",
    "surprised":     "surprised.mp4",
    "angry":         "angry.mp4",
    # Hand gestures
    "hand_on_head":  "fight.mp4",
    "thumbs_up":     "fight.mp4",
    "two_hands":     "look_right.mp4",
    # Combos
    "combo_victory": "combo_victory.mp4",
    "combo_cat":     "combo_cat.mp4",
    "combo_ko":      "combo_ko.mp4",
    "combo_wow":     "combo_wow.mp4",
}


class GestureMatcher:
    def __init__(self, assets_dir: str, state):
        self.assets_dir = assets_dir
        self._state     = state
        self._players: dict[str, MediaPlayer] = {}

        with state._map_lock:
            state.gesture_map = dict(DEFAULT_MAP)

    def get_player(self, gesture_name: str) -> MediaPlayer | None:
        mapping  = self._state.get_mapping()
        filename = mapping.get(gesture_name)
        if not filename:
            return None

        if filename not in self._players:
            path = os.path.join(self.assets_dir, filename)
            if not os.path.exists(path):
                return None          # silently skip missing assets
            self._players[filename] = MediaPlayer(path)

        return self._players[filename]

    def release_all(self):
        for p in self._players.values():
            p.release()
        self._players.clear()
