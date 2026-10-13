"""
HYS Spectra → Structure  |  app.py
Run with: python app.py

Project layout:
  HYS_project/
  ├── app.py
  ├── setup_db.py        ← run once to build the database
  ├── model.py
  ├── evaluator.py
  ├── smiles_tokenizer.py
  ├── preprocessor.py
  ├── server.py          ← shared DB lookup + model inference
  ├── dataset_config.json
  ├── best_model.pt
  └── search_data/
      └── nmrshiftdb.parquet
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, os, sys, re, io

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDLogger.DisableLog("rdApp.*")
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# Database lookup and model inference are shared with the web app — see server.py.
# Edit tolerances (H1_TOL, C13_TOL) there.
import server
DB_PATH = server.DB_PATH

# ── Colours ───────────────────────────────────────────────────────────────────
BG        = "#F4F3F0"
SURFACE   = "#FFFFFF"
BORDER    = "#D8D6CE"
HDR_BG    = "#F0EEE9"
PURPLE    = "#534AB7"
PURPLE_LT = "#EEEDFE"
TEAL      = "#1D9E75"
TEAL_LT   = "#E1F5EE"
AMBER     = "#BA7517"
AMBER_LT  = "#FDF3DC"
RED       = "#A32D2D"
TEXT      = "#111111"
TEXT2     = "#666662"
TEXT3     = "#AAAAAA"

MAC       = sys.platform == "darwin"
FONT      = ("SF Pro Text",    13) if MAC else ("Segoe UI", 12)
FONT_SM   = ("SF Pro Text",    11) if MAC else ("Segoe UI", 10)
FONT_MONO = ("SF Mono",        11) if MAC else ("Consolas", 11)
FONT_LG   = ("SF Pro Display", 17, "bold") if MAC else ("Segoe UI", 15, "bold")
FONT_H2   = ("SF Pro Text",    12, "bold") if MAC else ("Segoe UI", 11, "bold")
FONT_TITLE= ("SF Pro Display", 14, "bold") if MAC else ("Segoe UI", 12, "bold")

PLACEHOLDER = "paste ppm shifts, or upload a file…"

# ── Demo data ─────────────────────────────────────────────────────────────────
DEMOS = {
    "Aspirin (3 spectra)": {
        "c13": "20.6, 122.3, 124.0, 126.0, 131.4, 134.2, 151.0, 169.6, 170.7",
        "h1":  "2.43, 7.10, 7.35, 7.62, 8.12, 11.89",
        "ir":  "760, 1010, 1183, 1368, 1456, 1605, 1692, 1750, 2900, 3300",
        "force_model": True,
    },
    "Caffeine (¹³C only)": {
        "c13": "27.9, 29.6, 33.6, 107.3, 141.5, 148.5, 151.6, 155.0",
        "h1":  "",
        "ir":  "",
        "force_model": False,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_peaks(text):
    if not text or text.strip() == PLACEHOLDER:
        return []
    result = []
    for t in re.split(r"[\s,;\n]+", text.strip()):
        try:
            result.append(float(t))
        except ValueError:
            pass
    return result

def run_pipeline(h1, c13, ir, force_model=False, beam_width=5):
    """h1, c13, ir: lists of plain shift / wavenumber floats."""
    candidates = []
    if not force_model:
        candidates = server.db_lookup(h1, c13)
    if not candidates:
        as_rich = lambda shifts: [{"shift": s, "nH": None, "multiplicity": None, "J": []}
                                  for s in shifts]
        candidates = server.model_predict(as_rich(h1), as_rich(c13), ir, beam_width)
    if not candidates:
        return None, []
    return candidates[0], candidates[1:]


# ══════════════════════════════════════════════════════════════════════════════
# RDKit helpers
# ══════════════════════════════════════════════════════════════════════════════

def render_mol_image(smiles, width=440, height=240):
    if not RDKIT_OK:
        return None
    try:
        from PIL import Image, ImageTk
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().padding = 0.12
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))
        return ImageTk.PhotoImage(img)
    except Exception:
        return None

def mol_properties(smiles):
    if not RDKIT_OK:
        return {}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "mw":      f"{Descriptors.MolWt(mol):.1f}",
            "atoms":   str(mol.GetNumHeavyAtoms()),
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# GUI helpers
# ══════════════════════════════════════════════════════════════════════════════

def titled_box(parent, title, bg=BG):
    """
    Returns (outer_frame, inner_frame, hdr_frame).
    Draws a visible bordered card with a title header bar.
    """
    # Outer border: 1px solid via a coloured frame acting as border
    border = tk.Frame(parent, bg=BORDER)
    # Title header
    hdr = tk.Frame(border, bg=HDR_BG)
    hdr.pack(fill="x", side="top", padx=1, pady=(1,0))
    tk.Label(hdr, text=title, font=FONT_TITLE, fg=TEXT,
             bg=HDR_BG, anchor="w").pack(side="left", padx=12, pady=8)
    tk.Frame(border, bg=BORDER, height=1).pack(fill="x", padx=1)
    # Content area
    inner = tk.Frame(border, bg=SURFACE)
    inner.pack(fill="both", expand=True, padx=1, pady=(0,1))
    return border, inner, hdr


# ══════════════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════════════

class HYSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HYS — Spectra → Structure")
        self.configure(bg=BG)
        self.geometry("1220x860")
        self.minsize(1020, 720)

        self._top  = None
        self._alts = []
        self._busy = False

        self._build_styles()
        self._build_ui()

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame", background=BG)

    # ── Top bar ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        bar = tk.Frame(self, bg=SURFACE, height=48)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=BORDER, height=1).pack(side="bottom", fill="x")
        tk.Label(bar, text="⬡", font=("Arial", 18), fg=PURPLE,
                 bg=SURFACE).pack(side="left", padx=(16, 6))
        tk.Label(bar, text="HYS  Spectra → Structure",
                 font=FONT_LG, fg=TEXT, bg=SURFACE).pack(side="left")
        self._status = tk.Label(bar, text="", font=FONT_SM, fg=TEXT2, bg=SURFACE)
        self._status.pack(side="right", padx=16)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=14)

        # Left column — fixed width
        left = tk.Frame(body, bg=BG, width=430)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        # Right column
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_right(right)

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN
    # ══════════════════════════════════════════════════════════════════════════
    def _build_left(self, parent):
        self._panels = {}
        specs = [
            ("h1",  "¹H Spectrum",  PURPLE),
            ("c13", "¹³C Spectrum", TEAL),
            ("ir",  "IR Spectrum",  AMBER),
        ]
        for fid, title, color in specs:
            self._panels[fid] = self._make_panel(parent, fid, title, color)

        # Buttons
        btn_row = tk.Frame(parent, bg=BG)
        btn_row.pack(fill="x", pady=(10, 0))

        self._pred_btn = tk.Button(
            btn_row, text="▶  Predict",
            font=FONT, bg=PURPLE, fg="white",
            relief="flat", bd=0, padx=16, pady=10,
            cursor="hand2",
            activebackground="#3C3489", activeforeground="white",
            command=self._run_predict,
        )
        self._pred_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        # Demo menubutton — styled to match Predict
        self._demo_var = tk.StringVar(value="Demo  ▾")
        demo_mb = tk.Menubutton(
            btn_row, textvariable=self._demo_var,
            font=FONT, bg=SURFACE, fg=TEXT,
            relief="flat", bd=0, padx=14, pady=10,
            cursor="hand2",
            highlightthickness=1, highlightbackground=BORDER,
        )
        demo_mb.pack(side="left")
        menu = tk.Menu(demo_mb, tearoff=0, font=FONT_SM,
                       bg=SURFACE, fg=TEXT, relief="flat",
                       activebackground=PURPLE_LT, activeforeground=PURPLE)
        for lbl in DEMOS:
            menu.add_command(label=lbl, command=lambda l=lbl: self._load_demo(l))
        demo_mb["menu"] = menu

    def _make_panel(self, parent, fid, title, color):
        """One spectrum panel: titled box + chart + text input."""
        outer, inner, hdr = titled_box(parent, title)
        outer.pack(fill="x", pady=(0, 10))

        # Upload button in header
        tk.Button(hdr, text="↑ upload", font=FONT_SM, fg=TEXT2,
                  bg=HDR_BG, relief="flat", bd=0, padx=8, pady=4,
                  cursor="hand2", activebackground=BORDER,
                  command=lambda f=fid: self._upload(f)
                  ).pack(side="right", padx=8, pady=4)

        # Coloured dot in header
        tk.Label(hdr, text="●", fg=color, bg=HDR_BG,
                 font=("Arial", 10)).pack(side="left", padx=(0,4), pady=8)

        # Chart — fixed height via frame
        chart_frame = tk.Frame(inner, bg=SURFACE, height=110)
        chart_frame.pack(fill="x", padx=0)
        chart_frame.pack_propagate(False)

        fig = Figure(figsize=(1, 1), dpi=100, facecolor="white")
        fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.18)
        ax = fig.add_subplot(111)
        self._style_ax(ax, color)
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        w = canvas.get_tk_widget()
        w.configure(bg=SURFACE, highlightthickness=0)
        w.pack(fill="both", expand=True)

        # Separator
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x")

        # Text input
        txt_bg = "#F8F7F4"
        txt = tk.Text(inner, height=2, font=FONT_MONO,
                      bg=txt_bg, fg=TEXT2, insertbackground=TEXT,
                      relief="flat", bd=0, padx=10, pady=8,
                      wrap="word", highlightthickness=0)
        txt.pack(fill="x")
        txt.insert("1.0", PLACEHOLDER)

        txt.bind("<FocusIn>",    lambda e, t=txt: self._clear_ph(t))
        txt.bind("<FocusOut>",   lambda e, t=txt: self._restore_ph(t))
        txt.bind("<KeyRelease>",
                 lambda e, t=txt, a=ax, f=fig, c=color, cv=canvas:
                     self._refresh_chart(t, a, f, c, cv))

        return {"txt": txt, "ax": ax, "fig": fig, "canvas": canvas, "color": color}

    def _style_ax(self, ax, color, peaks=None):
        ax.clear()
        ax.set_facecolor("white")
        ax.set_yticks([])
        ax.tick_params(colors=TEXT2, labelsize=7, length=2, pad=2)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
            sp.set_linewidth(0.6)
        if peaks:
            ax.vlines(peaks, 0, 1, color=color, linewidth=1.8, alpha=0.9)
            ax.set_ylim(0, 1.7)
            for p in sorted(peaks)[:10]:
                ax.text(p, 1.08, f"{p:.1f}", ha="center", va="bottom",
                        fontsize=6, color=color, rotation=70, clip_on=True)
        else:
            ax.text(0.5, 0.42, "no data", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT3, fontsize=9,
                    style="italic")
            ax.set_ylim(0, 1)

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN
    # ══════════════════════════════════════════════════════════════════════════
    def _build_right(self, parent):
        # ── Top Match box ────────────────────────────────────────────────────
        tm_outer, tm_inner, tm_hdr = titled_box(parent, "Top Match")
        tm_outer.pack(fill="both", expand=True, pady=(0, 10))

        # Badge + confidence in header
        self._src_badge = tk.Label(tm_hdr, text="", font=FONT_SM,
                                   fg=TEAL, bg=TEAL_LT, padx=8, pady=2)
        self._src_badge.pack(side="left", padx=6)
        self._conf_lbl = tk.Label(tm_hdr, text="", font=FONT_SM,
                                  fg=TEXT2, bg=HDR_BG)
        self._conf_lbl.pack(side="left")

        # Molecule image area
        self._mol_lbl = tk.Label(tm_inner, bg=SURFACE, text="")
        self._mol_lbl.pack(expand=True, fill="both")

        # SMILES strip
        smiles_strip = tk.Frame(tm_inner, bg=HDR_BG)
        smiles_strip.pack(fill="x", side="bottom")
        tk.Frame(smiles_strip, bg=BORDER, height=1).pack(fill="x", side="top")
        tk.Label(smiles_strip, text="SMILES", font=FONT_SM,
                 fg=TEXT2, bg=HDR_BG).pack(side="left", padx=(10,4), pady=6)
        self._smiles_lbl = tk.Label(smiles_strip, text="—",
                                    font=FONT_MONO, fg=PURPLE, bg=HDR_BG,
                                    wraplength=480, justify="left")
        self._smiles_lbl.pack(side="left", padx=4, pady=6)
        tk.Button(smiles_strip, text="copy", font=FONT_SM, fg=TEXT2,
                  bg=HDR_BG, relief="flat", bd=0, padx=8, pady=4,
                  cursor="hand2", command=self._copy_smiles
                  ).pack(side="right", padx=8)

        # ── Properties strip ──────────────────────────────────────────────────
        prop_row = tk.Frame(parent, bg=BG)
        prop_row.pack(fill="x", pady=(0, 10))
        self._prop = {}
        for key, lbl in [("formula","Formula"),("mw","MW (g/mol)"),("atoms","Heavy atoms")]:
            box_outer, box_inner, _ = titled_box(prop_row, lbl)
            box_outer.pack(side="left", fill="x", expand=True, padx=(0, 8))
            v = tk.Label(box_inner, text="—", font=FONT_H2,
                         fg=TEXT, bg=SURFACE)
            v.pack(anchor="w", padx=12, pady=10)
            self._prop[key] = v

        # ── Alt Options box ───────────────────────────────────────────────────
        alt_outer, alt_inner, _ = titled_box(parent, "Alt options")
        alt_outer.pack(fill="x")

        self._alt_rows = []
        for i in range(4):
            row = tk.Frame(alt_inner, bg=SURFACE)
            row.pack(fill="x")
            if i < 3:
                tk.Frame(alt_inner, bg=BORDER, height=1).pack(fill="x")
            tk.Label(row, text=f"{i+1}.", font=FONT_SM,
                     fg=TEXT3, bg=SURFACE, width=3,
                     anchor="e").pack(side="left", padx=(12,4), pady=9)
            btn = tk.Button(row, text="—", font=FONT_MONO,
                            fg=PURPLE, bg=SURFACE, relief="flat", bd=0,
                            anchor="w", cursor="hand2", pady=6,
                            activebackground=PURPLE_LT,
                            disabledforeground=TEXT3,
                            command=lambda idx=i: self._pick_alt(idx))
            btn.pack(side="left", fill="x", expand=True)
            sc = tk.Label(row, text="", font=FONT_SM,
                          fg=TEXT2, bg=SURFACE)
            sc.pack(side="right", padx=12)
            self._alt_rows.append((btn, sc))

        # ── Empty overlay ─────────────────────────────────────────────────────
        self._overlay = tk.Frame(tm_inner, bg=SURFACE)
        self._overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        tk.Label(self._overlay, text="⬡", font=("Arial", 46),
                 fg="#DDDDDD", bg=SURFACE).pack(expand=True, pady=(50,6))
        tk.Label(self._overlay,
                 text="Enter spectra and click  ▶ Predict",
                 font=FONT_SM, fg=TEXT2, bg=SURFACE).pack()
        tk.Label(self._overlay,
                 text="or choose a Demo from the dropdown",
                 font=FONT_SM, fg=TEXT3, bg=SURFACE).pack(pady=(3, 50))

    # ══════════════════════════════════════════════════════════════════════════
    # Interactions
    # ══════════════════════════════════════════════════════════════════════════
    def _clear_ph(self, txt):
        if txt.get("1.0","end").strip() == PLACEHOLDER:
            txt.delete("1.0","end")
            txt.config(fg=TEXT)

    def _restore_ph(self, txt):
        if not txt.get("1.0","end").strip():
            txt.config(fg=TEXT2)
            txt.insert("1.0", PLACEHOLDER)

    def _refresh_chart(self, txt, ax, fig, color, canvas):
        raw = txt.get("1.0","end").strip()
        peaks = parse_peaks(raw)
        self._style_ax(ax, color, peaks or None)
        canvas.draw()

    def _get_peaks(self, fid):
        return parse_peaks(self._panels[fid]["txt"].get("1.0","end").strip())

    def _set_field(self, fid, value):
        p = self._panels[fid]
        txt, ax, fig, canvas, color = (p["txt"], p["ax"],
                                       p["fig"], p["canvas"], p["color"])
        txt.delete("1.0","end")
        if value:
            txt.config(fg=TEXT)
            txt.insert("1.0", value)
            peaks = parse_peaks(value)
        else:
            txt.config(fg=TEXT2)
            txt.insert("1.0", PLACEHOLDER)
            peaks = []
        self._style_ax(ax, color, peaks or None)
        canvas.draw()

    def _upload(self, fid):
        path = filedialog.askopenfilename(
            filetypes=[("CSV / Text","*.csv *.txt"),
                       ("JCAMP-DX","*.jcamp *.dx"),
                       ("All files","*.*")])
        if not path:
            return
        try:
            with open(path,"r",errors="ignore") as f:
                raw = f.read()
            lines = [l for l in raw.splitlines()
                     if not l.startswith("##") and not l.startswith("$")]
            peaks = parse_peaks(" ".join(lines))
            self._set_field(fid, ", ".join(f"{p:.3f}" for p in peaks))
        except Exception as e:
            messagebox.showerror("Upload error", str(e))

    def _load_demo(self, label):
        demo = DEMOS[label]
        self._demo_var.set("Demo  ▾")
        self._set_field("h1",  demo.get("h1",  ""))
        self._set_field("c13", demo.get("c13", ""))
        self._set_field("ir",  demo.get("ir",  ""))
        self._run_predict(force_model=demo.get("force_model", False))

    def _run_predict(self, force_model=False):
        if self._busy:
            return
        h1  = self._get_peaks("h1")
        c13 = self._get_peaks("c13")
        ir  = self._get_peaks("ir")
        if not h1 and not c13 and not ir:
            messagebox.showwarning("No input",
                "Please enter at least one set of spectral peaks.")
            return
        self._busy = True
        self._pred_btn.config(text="⏳  Running…", state="disabled", bg="#9C96DC")
        self._status.config(text="Searching database…", fg=AMBER)

        def worker():
            top, alts = run_pipeline(h1, c13, ir,
                                     force_model=force_model, beam_width=5)
            self.after(0, lambda: self._show(top, alts))

        threading.Thread(target=worker, daemon=True).start()

    def _show(self, top, alts):
        self._busy = False
        self._pred_btn.config(text="▶  Predict", state="normal", bg=PURPLE)
        if top is None:
            messagebox.showerror("No result","No predictions returned.")
            self._status.config(text="No result", fg=RED)
            return

        warning = top.get("warning","")
        if warning:
            messagebox.showwarning("Model not loaded",
                f"Could not load model — showing demo output.\n\n"
                f"Error: {warning}\n\n"
                f"Check that best_model.pt and dataset_config.json are "
                f"in the project folder.")

        source = top.get("source","demo")
        badge  = {"database": ("DB match", TEAL,   TEAL_LT),
                  "model":    ("model",    PURPLE,  PURPLE_LT),
                  "demo":     ("demo",     AMBER,   AMBER_LT)}[source]
        self._src_badge.config(text=f"  {badge[0]}  ", fg=badge[1], bg=badge[2])
        self._conf_lbl.config(text=f"{int(top['score']*100)}% confidence")
        self._status.config(
            text=f"{'DB hit' if source=='database' else 'Model prediction'}"
                 f" · {len(alts)} alternatives",
            fg=TEAL if source=="database" else PURPLE)

        self._top  = top
        self._alts = alts
        self._render_top(top)
        self._render_alts(alts)
        self._overlay.place_forget()

    def _render_top(self, c):
        smiles = c["smiles"]
        self._smiles_lbl.config(text=smiles)
        for k, v in mol_properties(smiles).items():
            if k in self._prop:
                self._prop[k].config(text=v)
        photo = render_mol_image(smiles)
        if photo:
            self._mol_lbl.config(image=photo, text="")
            self._mol_lbl._img = photo
        else:
            self._mol_lbl.config(
                image="",
                text="pip install rdkit Pillow  to see structure\n\n" + smiles,
                font=FONT_MONO, fg=TEXT2)

    def _render_alts(self, alts):
        for i,(btn,sc_lbl) in enumerate(self._alt_rows):
            if i < len(alts):
                sm = alts[i]["smiles"]
                btn.config(text=(sm[:54]+"…" if len(sm)>54 else sm),
                           state="normal",
                           command=lambda idx=i: self._pick_alt(idx))
                sc_lbl.config(text=f"{int(alts[i]['score']*100)}%")
            else:
                btn.config(text="—", state="disabled")
                sc_lbl.config(text="")

    def _pick_alt(self, idx):
        if idx >= len(self._alts):
            return
        new_top  = self._alts[idx]
        new_alts = [self._top]+[a for j,a in enumerate(self._alts) if j!=idx]
        self._top  = new_top
        self._alts = new_alts
        self._conf_lbl.config(text=f"{int(new_top['score']*100)}% confidence")
        self._render_top(new_top)
        self._render_alts(new_alts)

    def _copy_smiles(self):
        if self._top:
            self.clipboard_clear()
            self.clipboard_append(self._top["smiles"])
            self._status.config(text="SMILES copied ✓", fg=TEAL)
            self.after(1800, lambda: self._status.config(text="", fg=TEXT2))


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"⚠  Database not found at:\n   {DB_PATH}")
        print("   Run  python setup_db.py  first to build it.")
    app = HYSApp()
    app.mainloop()