"""
Single entry point — no OpenCV window needed.
Detection runs in a background thread; open http://localhost:5000 to use the app.
"""

import threading
from app_state import AppState
from app_core import run_loop
from gui.dashboard_server import DashboardServer


def main():
    state = AppState()

    dashboard = DashboardServer(state, port=5000)

    loop_thread = threading.Thread(target=run_loop, args=(state,), daemon=True)
    loop_thread.start()

    dashboard.run_blocking()   # blocks — Flask owns the main thread


if __name__ == "__main__":
    main()
