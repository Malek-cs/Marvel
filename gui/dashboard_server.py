import threading
import time
import cv2
from flask import Flask, Response, jsonify, request

HTML = """<!DOCTYPE html>
<html>
<head>
  <title>AR Gesture Reactor</title>
  <style>
    *{box-sizing:border-box}
    body{margin:0;background:#111;color:#fff;font-family:monospace;
         display:flex;flex-direction:column;align-items:center;padding:20px;gap:12px}
    h1{color:#00ff78;margin:0}
    img{border:2px solid #333;border-radius:8px;max-width:100%;width:900px}
    .badges{display:flex;gap:16px;align-items:center;font-size:1.1em}
    #gesture{color:#0af;min-width:200px}
    #combo{color:#ffd700}
    #emotion{color:#f90}
    #rec{color:#f33;font-weight:bold}
    .controls{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
    button{background:#222;color:#00ff78;border:1px solid #444;padding:8px 18px;
           cursor:pointer;border-radius:6px;font-family:monospace;font-size:0.95em}
    button:hover{background:#333}
    button.active{background:#005500;border-color:#00ff78}
    button.danger{color:#f55;border-color:#f55}
    button.danger:hover{background:#330000}
    .train-row{display:flex;gap:8px;align-items:center}
    .train-row input{background:#222;color:#fff;border:1px solid #555;
                     padding:6px 10px;border-radius:4px;font-family:monospace;width:180px}
    table{border-collapse:collapse;width:min(700px,95vw)}
    th{background:#1a1a1a;padding:8px;text-align:left;color:#00ff78;
       border-bottom:1px solid #333}
    td{padding:6px 8px;border-bottom:1px solid #222}
    td input{background:#1a1a1a;color:#fff;border:1px solid #444;
             padding:4px 8px;border-radius:4px;width:180px}
    td button{padding:4px 10px;font-size:0.85em}
    #status{color:#555;font-size:0.8em}
  </style>
</head>
<body>
  <h1>AR Gesture Reactor</h1>

  <img src="/video_feed" />

  <div class="badges">
    <span id="gesture">—</span>
    <span id="combo"></span>
    <span id="emotion"></span>
    <span id="rec"></span>
  </div>

  <div class="controls">
    <button id="btnRec"     onclick="toggleRec()">⏺ Record</button>
    <button id="btnOverlay" onclick="toggleOverlay()">⧉ Overlay Mode</button>
  </div>

  <div class="train-row">
    <input id="trainName" placeholder="gesture name..." />
    <button onclick="startTrain()">🧠 Train Gesture</button>
    <span id="trainStatus" style="color:#0af;font-size:0.85em"></span>
  </div>

  <h3 style="color:#444;margin-bottom:4px">Gesture Map</h3>
  <table id="mapTable">
    <tr><th>Gesture</th><th>Media File</th><th></th></tr>
  </table>
  <div id="status"></div>

  <script>
    let overlayOn = false;
    let recOn     = false;

    function poll(){
      fetch('/api/status').then(r=>r.json()).then(d=>{
        document.getElementById('gesture').textContent  = d.gesture  || '—';
        document.getElementById('combo').textContent    = d.combo    ? '⚡ '+d.combo   : '';
        document.getElementById('emotion').textContent  = d.emotion  ? '😶 '+d.emotion : '';
        document.getElementById('rec').textContent      = d.recording ? '● REC' : '';
        if(d.training_active){
          document.getElementById('trainStatus').textContent =
            'Training: '+d.training_count+'/60';
        } else if(document.getElementById('trainStatus').textContent.startsWith('Training')){
          document.getElementById('trainStatus').textContent = '✅ Done!';
          setTimeout(()=>document.getElementById('trainStatus').textContent='', 2000);
        }
        recOn     = d.recording;
        overlayOn = d.overlay;
        document.getElementById('btnRec').classList.toggle('active', recOn);
        document.getElementById('btnOverlay').classList.toggle('active', overlayOn);
      }).catch(()=>{});
    }

    function toggleRec(){
      fetch('/api/record',{method:'POST'}).then(r=>r.json())
        .then(d=>document.getElementById('status').textContent=d.message||'');
    }

    function toggleOverlay(){
      fetch('/api/overlay',{method:'POST'}).then(r=>r.json())
        .then(d=>document.getElementById('status').textContent=d.message||'');
    }

    function startTrain(){
      const name = document.getElementById('trainName').value.trim();
      if(!name){ alert('Enter a gesture name first'); return; }
      fetch('/api/train',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({name})})
      .then(r=>r.json())
      .then(d=>{
        document.getElementById('trainStatus').textContent = d.message || '';
      });
    }

    function loadMap(){
      fetch('/api/mapping').then(r=>r.json()).then(m=>{
        const t = document.getElementById('mapTable');
        while(t.rows.length > 1) t.deleteRow(1);
        Object.entries(m).forEach(([g,f])=>{
          const r   = t.insertRow();
          r.insertCell().textContent = g;
          const inp = document.createElement('input');
          inp.value = f;
          r.insertCell().appendChild(inp);
          const btn = document.createElement('button');
          btn.textContent = 'Apply';
          btn.onclick = ()=>{
            fetch('/api/mapping',{method:'POST',
              headers:{'Content-Type':'application/json'},
              body:JSON.stringify({gesture:g, file:inp.value})})
            .then(()=>document.getElementById('status').textContent='Saved: '+g);
          };
          r.insertCell().appendChild(btn);
        });
      });
    }

    setInterval(poll, 250);
    loadMap();
  </script>
</body>
</html>"""


class DashboardServer:
    def __init__(self, state, port: int = 5000):
        self._state   = state
        self._port    = port
        self._app     = Flask(__name__)
        self._setup()

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
        def api_status():
            return jsonify({
                "gesture":        state.current_gesture,
                "combo":          state.current_combo,
                "emotion":        getattr(state, "current_emotion", ""),
                "recording":      state.recording,
                "overlay":        state.overlay_mode,
                "training_active":state.training_active,
                "training_count": state.training_count,
            })

        @app.route("/api/record", methods=["POST"])
        def api_record():
            state.recording = not state.recording
            msg = "Recording started" if state.recording else "Recording stopped"
            return jsonify({"recording": state.recording, "message": msg})

        @app.route("/api/overlay", methods=["POST"])
        def api_overlay():
            state.overlay_mode = not state.overlay_mode
            msg = "Overlay ON" if state.overlay_mode else "Overlay OFF"
            return jsonify({"overlay": state.overlay_mode, "message": msg})

        @app.route("/api/train", methods=["POST"])
        def api_train():
            if state.training_active:
                return jsonify({"message": "Already training..."}), 400
            data = request.get_json(silent=True) or {}
            name = data.get("name", "").strip()
            if not name:
                return jsonify({"error": "name required"}), 400
            state.training_name   = name
            state.training_active = True
            return jsonify({"message": f"Training '{name}' — hold pose for 60 frames"})

        @app.route("/api/mapping", methods=["GET"])
        def api_get_mapping():
            return jsonify(state.get_mapping())

        @app.route("/api/mapping", methods=["POST"])
        def api_set_mapping():
            data    = request.get_json(silent=True) or {}
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
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            time.sleep(0.033)

    def start(self):
        """Run Flask in a daemon thread (used by main.py)."""
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        t = threading.Thread(
            target=lambda: self._app.run(
                host="0.0.0.0", port=self._port,
                threaded=True, use_reloader=False),
            daemon=True)
        t.start()
        print(f"[dashboard] http://localhost:{self._port}")

    def run_blocking(self):
        """Run Flask in the main thread (used by run.py)."""
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        print(f"[dashboard] http://localhost:{self._port}")
        self._app.run(host="0.0.0.0", port=self._port,
                      threaded=True, use_reloader=False)
