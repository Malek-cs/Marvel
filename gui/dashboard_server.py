import threading
import time
import cv2
from flask import Flask, Response, jsonify, request

HTML = """<!DOCTYPE html>
<html>
<head>
  <title>AR Gesture Reactor</title>
  <style>
    body{margin:0;background:#111;color:#fff;font-family:monospace;display:flex;
         flex-direction:column;align-items:center;padding:20px}
    h1{color:#00ff78;margin-bottom:10px}
    img{border:2px solid #333;border-radius:8px;max-width:100%}
    #gesture{font-size:2em;margin:12px 0;color:#0af;min-height:1.5em}
    #combo{font-size:1.4em;color:#ffd700;min-height:1.4em}
    #emotion{font-size:1em;color:#f90;min-height:1.2em}
    #rec{color:#f33;font-size:0.9em}
    table{border-collapse:collapse;margin-top:20px;width:600px}
    th{background:#222;padding:8px;text-align:left;color:#00ff78}
    td{padding:6px 8px;border-bottom:1px solid #333}
    input{background:#222;color:#fff;border:1px solid #555;padding:4px;width:160px}
    button{background:#333;color:#00ff78;border:none;padding:5px 10px;
           cursor:pointer;border-radius:4px}
    button:hover{background:#555}
  </style>
</head>
<body>
  <h1>AR Gesture Reactor</h1>
  <div id="rec"></div>
  <img src="/video_feed" />
  <div id="gesture">—</div>
  <div id="combo"></div>
  <div id="emotion"></div>

  <h3 style="color:#555;margin-top:30px">Live Gesture Map</h3>
  <table id="mapTable"><tr><th>Gesture</th><th>File</th><th></th></tr></table>

  <script>
    function poll(){
      fetch('/api/status').then(r=>r.json()).then(d=>{
        document.getElementById('gesture').textContent = d.gesture || '—';
        document.getElementById('combo').textContent   = d.combo   ? '⚡ '+d.combo : '';
        document.getElementById('emotion').textContent = d.emotion ? '😶 '+d.emotion : '';
        document.getElementById('rec').textContent     = d.recording ? '● REC' : '';
      }).catch(()=>{});
    }
    function loadMap(){
      fetch('/api/mapping').then(r=>r.json()).then(m=>{
        const t = document.getElementById('mapTable');
        while(t.rows.length>1) t.deleteRow(1);
        Object.entries(m).forEach(([g,f])=>{
          const r = t.insertRow();
          r.insertCell().textContent = g;
          const inp = document.createElement('input');
          inp.value = f;
          r.insertCell().appendChild(inp);
          const btn = document.createElement('button');
          btn.textContent='Apply';
          btn.onclick=()=>fetch('/api/mapping',{method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({gesture:g,file:inp.value})});
          r.insertCell().appendChild(btn);
        });
      });
    }
    setInterval(poll, 200);
    loadMap();
  </script>
</body>
</html>"""


class DashboardServer:
    def __init__(self, state, port: int = 5000):
        self._state  = state
        self._port   = port
        self._app    = Flask(__name__)
        self._emotion = ""   # updated externally
        self._setup()

    def set_emotion(self, emotion: str | None):
        self._emotion = emotion or ""

    def _setup(self):
        app   = self._app
        state = self._state

        @app.route("/")
        def index():
            return HTML

        @app.route("/video_feed")
        def video_feed():
            return Response(self._stream(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/api/status")
        def status():
            return jsonify({
                "gesture":   state.current_gesture,
                "combo":     state.current_combo,
                "emotion":   self._emotion,
                "recording": state.recording,
            })

        @app.route("/api/mapping", methods=["GET"])
        def get_mapping():
            return jsonify(state.get_mapping())

        @app.route("/api/mapping", methods=["POST"])
        def set_mapping():
            data = request.get_json(silent=True) or {}
            gesture = data.get("gesture", "")
            file    = data.get("file", "")
            if gesture and file:
                state.update_mapping(gesture, file)
                return jsonify({"ok": True})
            return jsonify({"error": "missing fields"}), 400

    def _stream(self):
        while True:
            frame = self._state.combined_frame
            if frame is None:
                time.sleep(0.033)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            time.sleep(0.033)

    def start(self):
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)   # silence Flask request logs
        t = threading.Thread(
            target=lambda: self._app.run(
                host="0.0.0.0", port=self._port,
                threaded=True, use_reloader=False),
            daemon=True)
        t.start()
        print(f"[dashboard] http://localhost:{self._port}")
