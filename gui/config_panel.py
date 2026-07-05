import threading
import tkinter as tk
from tkinter import ttk, filedialog
import os

class ConfigPanel:
    def __init__(self, state, assets_dir: str):
        self._state      = state
        self._assets_dir = assets_dir
        self._thread     = None

    def launch(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        root = tk.Tk()
        root.title("AR Gesture Config")
        root.geometry("520x480")
        root.configure(bg="#1e1e1e")
        self._build(root)
        root.mainloop()

    def _build(self, root):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel",  background="#1e1e1e", foreground="white")
        style.configure("TButton", background="#333", foreground="white")
        style.configure("TCombobox", fieldbackground="#333", foreground="white")

        tk.Label(root, text="Gesture → Media Mapper",
                 bg="#1e1e1e", fg="#00ff78",
                 font=("Helvetica", 14, "bold")).pack(pady=10)

        frame = tk.Frame(root, bg="#1e1e1e")
        frame.pack(fill=tk.BOTH, expand=True, padx=10)

        canvas = tk.Canvas(frame, bg="#1e1e1e", highlightthickness=0)
        scroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner  = tk.Frame(canvas, bg="#1e1e1e")

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        assets = self._scan_assets()
        mapping = self._state.get_mapping()

        for gesture, current_file in mapping.items():
            row = tk.Frame(inner, bg="#1e1e1e")
            row.pack(fill=tk.X, pady=2)

            tk.Label(row, text=gesture, width=18, anchor="w",
                     bg="#1e1e1e", fg="white",
                     font=("Courier", 9)).pack(side=tk.LEFT)

            var = tk.StringVar(value=current_file)
            combo = ttk.Combobox(row, textvariable=var, values=assets, width=22)
            combo.pack(side=tk.LEFT, padx=4)

            g = gesture
            tk.Button(row, text="Apply", bg="#333", fg="#00ff78",
                      command=lambda v=var, ge=g: self._apply(ge, v.get())
                      ).pack(side=tk.LEFT)

        # Bottom controls
        bottom = tk.Frame(root, bg="#1e1e1e")
        bottom.pack(fill=tk.X, padx=10, pady=6)

        tk.Button(bottom, text="Add New Asset", bg="#222", fg="white",
                  command=lambda: self._browse_new(mapping, assets, inner)
                  ).pack(side=tk.LEFT, padx=4)

        tk.Button(bottom, text="Close", bg="#550000", fg="white",
                  command=root.destroy).pack(side=tk.RIGHT, padx=4)

    def _scan_assets(self):
        exts = {".mp4", ".avi", ".mov", ".webm", ".png", ".jpg", ".jpeg"}
        try:
            return [f for f in os.listdir(self._assets_dir)
                    if os.path.splitext(f)[1].lower() in exts]
        except FileNotFoundError:
            return []

    def _apply(self, gesture: str, filename: str):
        self._state.update_mapping(gesture, filename)
        print(f"[config] {gesture} → {filename}")

    def _browse_new(self, mapping, assets, inner):
        path = filedialog.askopenfilename(
            initialdir=self._assets_dir,
            filetypes=[("Media", "*.mp4 *.avi *.mov *.png *.jpg *.jpeg"),
                       ("All", "*.*")])
        if path:
            fname = os.path.basename(path)
            if fname not in assets:
                assets.append(fname)
            print(f"[config] Added asset: {fname}")
