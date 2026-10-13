"""
HYS Spectra → Structure  |  server.py
Run with: python server.py
Then open: http://localhost:5000
"""

import os, sys, json, re
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Multiplicity vocabulary — must match preprocessor.py's MULTIPLICITY_MAP ║
# ╚══════════════════════════════════════════════════════════════════════════╝
KNOWN_MULTIPLICITIES = {"s", "d", "t", "q", "p", "m", "dd", "ddd", "dt", "td"}

# Common aliases / variants seen in copy-pasted NMR data → map onto the
# 10 classes the model was actually trained on.
MULTIPLICITY_ALIASES = {
    "br":     "m",   "brs":  "s",   "br s":  "s",   "bs":    "s",
    "singlet":"s",   "doublet":"d", "triplet":"t",  "quartet":"q",
    "quintet":"p",   "pentet":"p",  "multiplet":"m",
    "dq":     "m",   "qd":   "m",   "tt":   "m",    "ddt":   "m",
    "dtd":    "m",   "tdd":  "m",
}

DEFAULT_NH = 1   # best guess when integration isn't provided

def normalize_multiplicity(raw):
    """Map a raw multiplicity token onto one of the 10 trained classes."""
    if not raw:
        return None
    key = raw.strip().lower().replace(".", "")
    if key in KNOWN_MULTIPLICITIES:
        return key
    if key in MULTIPLICITY_ALIASES:
        return MULTIPLICITY_ALIASES[key]
    return None  # unrecognised — caller will fall back to default


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Rich peak parser                                                        ║
# ║  Accepts anything from a bare shift to a fully annotated peak:           ║
# ║                                                                          ║
# ║    7.1                                  → shift only                    ║
# ║    7.34 (d, 1H)                         → shift + multiplicity + nH     ║
# ║    7.34 (d, J=8.1, 1H)                  → + J coupling (parsed, unused) ║
# ║    7.34 (d, J = 8.1 Hz, 1H)             → spacing variants              ║
# ║    7.34(dd,J=8.1,1.2,2H)                → multi-coupling, no spaces     ║
# ║    11.89 (br s, 1H)                     → broad singlet alias           ║
# ║                                                                          ║
# ║  Returns a list of dicts: {"shift": float, "nH": int|None,              ║
# ║                             "multiplicity": str|None}                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Matches: <shift> [ ( <stuff inside parens> ) ]
PEAK_PATTERN = re.compile(
    r'(-?\d+\.?\d*)\s*(?:\(([^)]*)\))?'
)

def parse_rich_peaks(text):
    """
    Parse free-form pasted NMR peak text into structured peak dicts.
    Tolerant of missing info — only shift is required per peak.
    """
    if not text or not text.strip():
        return []

    peaks = []
    # Split on commas/semicolons/newlines that are NOT inside parentheses
    # by temporarily protecting paren contents, then splitting.
    # Simpler approach: find all "<num> (...)"" or "<num>" chunks via regex.
    for match in PEAK_PATTERN.finditer(text):
        shift_str, inner = match.group(1), match.group(2)
        if not shift_str:
            continue
        try:
            shift = float(shift_str)
        except ValueError:
            continue
        if shift == 0:
            continue

        peak = {"shift": shift, "nH": None, "multiplicity": None, "J": [], "count": None}

        if inner:
            # inner looks like: "d, J=8.1 Hz, 1H"  or  "dd, J = 8.1, 1.2, 2H"
            # or just: "1H"  or just: "d"
            tokens = [t.strip() for t in inner.split(",") if t.strip()]
            for tok in tokens:
                tok_low = tok.lower()

                # nH — look for trailing "H" with a leading number
                nh_match = re.match(r'^(\d+\.?\d*)\s*h$', tok_low)
                if nh_match:
                    peak["nH"] = int(round(float(nh_match.group(1))))
                    continue

                # 13C DEPT type — "C", "CH", "CH2", "CH3" → attached-H count
                dept_match = re.match(r'^c(h([0-3])?)?$', tok_low)
                if dept_match:
                    peak["nH"] = 0 if not dept_match.group(1) else int(dept_match.group(2) or 1)
                    continue

                # 13C equivalent-carbon count — "2C"
                count_match = re.match(r'^(\d+)\s*c$', tok_low)
                if count_match:
                    peak["count"] = int(count_match.group(1))
                    continue

                # J coupling — "J=8.1", "J = 8.1 Hz", "J=8.1Hz"
                j_match = re.match(r'^j\s*=\s*([\d.]+)\s*(hz)?$', tok_low)
                if j_match:
                    peak["J"].append(float(j_match.group(1)))
                    continue

                # Bare number with no unit, following a J= token — treat as
                # an additional J value (e.g. "dd, J=8.1, 1.2, 2H").
                # Only do this if we've already seen at least one J value
                # this peak, so it's not confused with a bare shift typo.
                bare_num = re.match(r'^([\d.]+)$', tok_low)
                if bare_num and peak["J"]:
                    peak["J"].append(float(bare_num.group(1)))
                    continue

                # multiplicity — remaining alphabetic tokens
                mult = normalize_multiplicity(tok_low)
                if mult:
                    peak["multiplicity"] = mult
                    continue
                # "br s" style — two words, try as combined alias
                mult2 = normalize_multiplicity(tok_low.replace(" ", ""))
                if mult2:
                    peak["multiplicity"] = mult2

        peaks.append(peak)

    return peaks


def shifts_only(rich_peaks):
    """Extract just the shift values — used for DB lookup matching."""
    return [p["shift"] for p in rich_peaks]

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_DIR, "dataset_config.json")
CKPT_PATH   = os.path.join(PROJECT_DIR, "best_model.pt")
DB_PATH     = os.path.join(PROJECT_DIR, "search_data", "nmrshiftdb.parquet")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TOLERANCES — edit to adjust database matching strictness                ║
# ║  H1_TOL  : max ppm difference for a ¹H  peak to count as a match        ║
# ║  C13_TOL : max ppm difference for a ¹³C peak to count as a match        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
H1_TOL  = 0.05
C13_TOL = 0.5

# ── Lazy globals ──────────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_config    = None
_db        = None

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from preprocessor import (encode_h_peaks, encode_c_peaks, make_peak_mask,
                          ir_from_peaks, encode_formula, parse_formula,
                          formula_counts_from_smiles, carbon_intensity,
                          C13_TYPICAL_INTENSITY)


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

def get_config():
    global _config
    if _config is None:
        with open(CONFIG_PATH) as f:
            _config = json.load(f)
    return _config


# ══════════════════════════════════════════════════════════════════════════════
# Database lookup
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    global _db
    if _db is None and os.path.exists(DB_PATH):
        import pandas as pd
        _db = pd.read_parquet(DB_PATH)
    return _db


def match_score(query, db_shifts, tol):
    if not query or not db_shifts:
        return 0.0
    return sum(1 for q in query
               if any(abs(q - d) <= tol for d in db_shifts)) / len(query)


def bidirectional_score(query_shifts, db_shifts, tol):
    """
    Jaccard-like score that penalises both missing query peaks AND
    extra DB peaks the query didn't account for.

    Score = matched / (len(query) + len(db) - matched)

    This prevents a 4-peak query from matching a 20-peak DB entry at
    100% just because those 4 shifts happen to appear somewhere in it.
    """
    if not query_shifts or not db_shifts:
        return 0.0
    matched = sum(1 for q in query_shifts
                  if any(abs(q - d) <= tol for d in db_shifts))
    return matched / (len(query_shifts) + len(db_shifts) - matched)


def db_lookup(h1_shifts, c13_shifts):
    """
    Stage 1: query NMRShiftDB for direct spectral matches.
    Uses bidirectional scoring to prevent small queries matching large molecules.
    Returns up to 5 candidates with score >= 0.5.
    """
    df = get_db()
    if df is None:
        return []

    c13_scores = {}
    h1_scores  = {}

    if c13_shifts:
        for _, row in df[df["nmr_type"] == "13C"].iterrows():
            s = bidirectional_score(c13_shifts, list(row["shifts"]), C13_TOL)
            if s > 0:
                c13_scores[row["smiles"]] = max(c13_scores.get(row["smiles"], 0.0), s)

    if h1_shifts:
        for _, row in df[df["nmr_type"] == "1H"].iterrows():
            s = bidirectional_score(h1_shifts, list(row["shifts"]), H1_TOL)
            if s > 0:
                h1_scores[row["smiles"]] = max(h1_scores.get(row["smiles"], 0.0), s)

    # Combine: average scores from each spectrum type provided
    all_smiles = set(c13_scores) | set(h1_scores)
    scores = {}
    for smiles in all_smiles:
        parts = []
        if smiles in c13_scores: parts.append(c13_scores[smiles])
        if smiles in h1_scores:  parts.append(h1_scores[smiles])
        scores[smiles] = sum(parts) / len(parts)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    results = [{"smiles": s, "score": round(sc, 4), "source": "database"}
            for s, sc in ranked if sc >= 0.5]
    print(f"[db_lookup] h1={len(h1_shifts)} peaks, c13={len(c13_shifts)} peaks → {len(results)} hits")
    for r in results:
        print(f"  {r['score']:.3f} {r['smiles'][:60]}")
    return results


def db_verify_candidate(smiles, h1_shifts, c13_shifts):
    """
    Stage 2: given a SMILES predicted by the model, look it up in
    NMRShiftDB and check how well the recorded experimental shifts
    agree with the user's input peaks.

    Returns a db_score in [0, 1] — 0 means not in DB or no agreement,
    1 means perfect bidirectional match.  None means not found in DB.
    """
    df = get_db()
    if df is None:
        return None

    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol)
    except Exception:
        return None

    rows = df[df["smiles"] == canonical]
    if rows.empty:
        return None   # not in DB — can't verify either way

    best = 0.0
    if h1_shifts:
        h1_rows = rows[rows["nmr_type"] == "1H"]
        for _, row in h1_rows.iterrows():
            s = bidirectional_score(h1_shifts, list(row["shifts"]), H1_TOL)
            best = max(best, s)

    if c13_shifts:
        c13_rows = rows[rows["nmr_type"] == "13C"]
        for _, row in c13_rows.iterrows():
            s = bidirectional_score(c13_shifts, list(row["shifts"]), C13_TOL)
            best = max(best, s)

    return round(best, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Model inference
# ══════════════════════════════════════════════════════════════════════════════

def get_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from model            import SpectraToSmilesModel
    from smiles_tokenizer import SmilesTokenizer

    config = get_config()

    tokenizer = SmilesTokenizer()
    tokenizer.load_vocab(config["token_to_id"])

    # Read architecture from checkpoint tensor shapes.
    # Handles both the old weights-only format and the new full checkpoint
    # format produced by the updated chunked_trainer.py.
    raw = torch.load(CKPT_PATH, map_location="cpu")
    if isinstance(raw, dict) and "model_state_dict" in raw:
        # New full checkpoint format
        state_dict = raw["model_state_dict"]
        input_version = raw.get("input_version", 1)
        saved_loss = raw.get("best_val_loss", "?")
        saved_epoch = raw.get("outer_epoch", "?")
        print(f"[model] Full checkpoint loaded (input_version={input_version}, "
              f"epoch={saved_epoch}, best_val_loss={saved_loss})")
    else:
        # Legacy weights-only format
        state_dict = raw
        input_version = 1
        print("[model] Legacy weights-only checkpoint loaded")

    d_model         = state_dict["token_embedding.weight"].shape[1]
    dim_feedforward = state_dict["decoder.layers.0.linear1.weight"].shape[0]
    num_enc         = sum(1 for k in state_dict
                         if re.match(r"memory_encoder\.layers\.\d+\.norm1\.weight", k))
    num_dec         = sum(1 for k in state_dict
                         if re.match(r"decoder\.layers\.\d+\.norm1\.weight", k))
    nhead           = 8  # not recoverable from weights; matches model.py default

    model = SpectraToSmilesModel(
        vocab_size         = config["vocab_size"],
        d_model            = d_model,
        nhead              = nhead,
        num_encoder_layers = num_enc,
        num_decoder_layers = num_dec,
        dim_feedforward    = dim_feedforward,
        max_smiles_length  = config["max_smiles_length"],
        input_version      = input_version,
    )
    model.load_state_dict(state_dict)
    model.eval()

    _model     = model
    _tokenizer = tokenizer
    return _model, _tokenizer


def infer_nh(peak):
    """
    Best-guess nH when the user didn't provide integration.
    Uses multiplicity to infer the most likely value:
      - singlet → 3H is most common (methyl) but 1H also common
        use 2 as a neutral midpoint when truly unknown
      - doublet/quartet → usually 3H (CH3 next to CH or CH2)
      - triplet → usually 2H (CH2 next to CH3) or 3H
      - multiplet → no good guess, use 1
      - no multiplicity at all → use 1 (safest fallback)
    This is still a guess — providing nH in the input is always better.
    """
    mult = peak.get("multiplicity")
    if mult in ("d", "q"):    return 3   # most common: CH3 doublet/quartet
    if mult == "t":           return 2   # most common: CH2 triplet
    if mult == "s":           return 2   # could be CH3 or ArH, split the diff
    if mult in ("dd","dt","td","ddd"): return 1  # usually single H
    return 1  # multiplet or unknown


def build_h_peaks(rich_peaks, guess_unknown=True):
    """
    Convert rich peak dicts (from parse_rich_peaks) → dicts expected by
    encode_h_peaks: {"centroid", "nH", "category"}.

    Values the user gave are used exactly. Missing ones:
      guess_unknown=True  (input_version 1 models, which never saw "unknown"):
          nH from infer_nh(), multiplicity "m".
      guess_unknown=False (input_version 2 models, trained with hidden values):
          left as None → encoded as the explicit "unknown" codes.
    """
    out = []
    for p in rich_peaks:
        nH, mult = p["nH"], p["multiplicity"]
        if guess_unknown:
            nH = nH if nH is not None else infer_nh(p)
            mult = mult or "m"
        out.append({"centroid": p["shift"], "nH": nH, "category": mult})
    return out


# Off-resonance 13C multiplicity → attached H (s = C, d = CH, t = CH2, q = CH3)
C13_MULT_TO_NH = {"s": 0, "d": 1, "t": 2, "q": 3}


def build_c_peaks(rich_peaks, guess_unknown=True):
    """
    Convert rich peak dicts → dicts expected by encode_c_peaks.

    The model reads 13C intensity as DEPT type × equivalent carbons (see
    preprocessor.carbon_intensity), so it's rebuilt from what the user gave:
    "(CH3)" / "(q)" sets the attached H, "(2C)" or a repeated shift sets the
    count. Without a DEPT type the intensity is unknown:
      guess_unknown=True  (input_version 1): training-set median intensity.
      guess_unknown=False (input_version 2): None → explicit "unknown" code.
    """
    merged = {}   # identical shifts listed twice = equivalent carbons
    for p in rich_peaks:
        key = round(p["shift"], 2)
        n_h = p["nH"] if p["nH"] is not None else C13_MULT_TO_NH.get(p["multiplicity"])
        count = p.get("count") or 1
        if key in merged:
            merged[key]["count"] += count
        else:
            merged[key] = {"shift": p["shift"], "n_h": n_h, "count": count}

    out = []
    for m in merged.values():
        if m["n_h"] is not None:
            intensity = carbon_intensity(m["n_h"], m["count"])
        else:
            intensity = C13_TYPICAL_INTENSITY if guess_unknown else None
        out.append({"delta (ppm)": m["shift"], "intensity": intensity})
    return out


def canonical(smiles):
    """Canonical SMILES, or None if RDKit can't parse it."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def apply_formula(candidates, formula_counts):
    """
    Tag each candidate with formula_match and put matches first (stable, so
    score order is kept within each group). Non-matching candidates are kept
    only if nothing matches, so the user still sees something.
    """
    if not formula_counts:
        return candidates
    for c in candidates:
        c["formula_match"] = formula_counts_from_smiles(c["smiles"]) == formula_counts
    matches = [c for c in candidates if c["formula_match"]]
    return matches if matches else candidates


def model_predict(h1_peaks, c13_peaks, ir_shifts, beam_width=5, formula_counts=None):
    """
    h1_peaks, c13_peaks: lists of rich peak dicts from parse_rich_peaks()
    ir_shifts: list of plain wavenumber floats
    formula_counts: element-count dict from parse_formula(), or None
    Returns unique, valid candidates (canonical SMILES), best first.
    """
    try:
        import torch
        from evaluator import generate_smiles_beam_search

        model, tokenizer = get_model()
        config           = get_config()
        v2               = model.input_version >= 2

        max_h   = config["max_h_peaks"]       # 28
        max_c   = config["max_c_peaks"]        # 74
        max_len = config["max_smiles_length"]  # 81

        # ── Encode peaks ──────────────────────────────────────────────────────
        h_peaks  = build_h_peaks(h1_peaks, guess_unknown=not v2)
        c_peaks  = build_c_peaks(c13_peaks, guess_unknown=not v2)

        # ── Encode IR ─────────────────────────────────────────────────────────
        # The model was trained on 1800-point continuous absorbance spectra,
        # so pasted wavenumbers are rebuilt into one (Lorentzian per peak).
        # No IR given → all zeros, which v2 models treat as "absent".
        ir_arr = ir_from_peaks(ir_shifts)

        # ── To tensors (unbatched — beam search adds the batch dim) ───────────
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        candidates = generate_smiles_beam_search(
            model,
            t(encode_h_peaks(h_peaks, max_h)), t(make_peak_mask(len(h_peaks), max_h)),
            t(encode_c_peaks(c_peaks, max_c)), t(make_peak_mask(len(c_peaks), max_c)),
            t(ir_arr),
            tokenizer  = tokenizer,
            device     = torch.device("cpu"),
            max_length = max_len,
            beam_width = beam_width,
            return_all = True,
            formula    = t(encode_formula(formula_counts)) if v2 else None,
        )

        # Different beams can spell the same molecule differently; keep the
        # best-scoring spelling of each, and drop chemically invalid SMILES.
        out, seen = [], set()
        for s, sc in candidates:
            can = canonical(s)
            if can is None or can in seen:
                continue
            seen.add(can)
            out.append({"smiles": can, "score": round(float(sc), 4), "source": "model"})
        return out

    except Exception as e:
        print(f"[model] Prediction failed, falling back to demo results: {e!r}")
        return [{"smiles": s, "score": sc, "source": "demo", "warning": str(e)}
                for s, sc in [
                    ("CC(=O)Oc1ccccc1C(=O)O", 0.91),
                    ("CC(=O)Oc1ccccc1C(=O)OC", 0.74),
                    ("O=C(O)c1ccccc1OC(C)=O",  0.68),
                    ("CC(=O)Oc1ccccc1",         0.41),
                    ("O=C(O)c1ccccc1O",         0.28),
                    ("CC(=O)c1ccccc1",          0.19),
                ]]


def mol_svg(smiles):
    """Return an SVG string for the molecule, or empty string on failure."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Draw import rdMolDraw2D
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[svg] Could not parse SMILES: {smiles[:40]}")
            return ""
        drawer = rdMolDraw2D.MolDraw2DSVG(460, 280)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().padding = 0.12
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return ""


def mol_props(smiles):
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdMolDescriptors
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[props] Could not parse SMILES: {smiles[:40]}")
            return {"formula": "?", "mw": "?", "atoms": "?"}
        return {
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "mw":      f"{Descriptors.MolWt(mol):.1f}",
            "atoms":   str(mol.GetNumHeavyAtoms()),
        }
    except Exception as e:
        print(f"[props] Error: {e}")
        return {"formula": "?", "mw": "?", "atoms": "?"}


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data        = request.json
    h1_text     = data.get("h1",  "")
    c13_text    = data.get("c13", "")
    ir_text     = data.get("ir",  "")
    force_model = data.get("force_model", False)
    formula_txt = (data.get("formula") or "").strip()

    # ── Parse + range-filter inputs ───────────────────────────────────────────
    def filter_range(peaks, mn, mx):
        return [p for p in peaks if mn <= p["shift"] <= mx]

    h1_peaks  = filter_range(parse_rich_peaks(h1_text),  -2,   14)
    c13_peaks = filter_range(parse_rich_peaks(c13_text), -5,  220)
    ir_peaks  = filter_range(parse_rich_peaks(ir_text),  400, 4000)

    h1_shifts  = shifts_only(h1_peaks)
    c13_shifts = shifts_only(c13_peaks)
    ir_shifts  = shifts_only(ir_peaks)

    if not h1_peaks and not c13_peaks and not ir_peaks:
        return jsonify({"error": "No input provided"}), 400

    formula_counts = None
    if formula_txt:
        try:
            formula_counts = parse_formula(formula_txt)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE: DB lookup → model fills remaining slots → formula filter
    # ══════════════════════════════════════════════════════════════════════════
    #
    # DB hits (real measured spectra) come first; the model fills the rest
    # and can propose structures the DB doesn't contain.
    # force_model=True (demos) bypasses DB and goes straight to model.
    # If a molecular formula is given, it's the strongest constraint we have:
    # DB hits that don't match are dropped, and the model searches wider
    # (bigger beam) so enough formula-matching candidates survive.

    DB_VERIFY_THRESHOLD = 0.25   # minimum bidirectional score to count as a hit
    SLOTS = 5                    # total candidates to return

    db_hits    = []
    model_hits = []

    # ── Stage 1: DB lookup ────────────────────────────────────────────────────
    if not force_model and (h1_shifts or c13_shifts):
        db_hits = db_lookup(h1_shifts, c13_shifts)
        # db_lookup already filters to score >= 0.5 bidirectional
        for c in db_hits:
            c["source"]   = "database"
            c["verified"] = True
        if formula_counts:
            db_hits = [c for c in apply_formula(db_hits, formula_counts)
                       if c["formula_match"]]

    # ── Stage 2: Model fills remaining slots ──────────────────────────────────
    # Always run the model — it may find structures the DB doesn't have.
    # model_predict already returns unique canonical SMILES.
    slots_needed = SLOTS - len(db_hits)
    model_candidates = model_predict(
        h1_peaks, c13_peaks, ir_shifts,
        beam_width=20 if formula_counts else 10,
        formula_counts=formula_counts,
    )
    model_candidates = apply_formula(model_candidates, formula_counts)

    db_smiles_set = {canonical(c["smiles"]) for c in db_hits}
    for c in model_candidates:
        if len(model_hits) >= slots_needed:
            break
        if c["smiles"] in db_smiles_set:   # same structure as a DB hit
            continue

        # Verify each model candidate against DB to get a confidence boost
        db_s = db_verify_candidate(c["smiles"], h1_shifts, c13_shifts)
        if db_s is not None and db_s >= DB_VERIFY_THRESHOLD:
            c["source"]   = "model+db"
            c["verified"] = True
            c["score"]    = round(c["score"] * 0.6 + db_s * 0.4, 4)
        else:
            c["verified"] = False
            if db_s is not None:
                c["score"] = round(c["score"] * 0.7, 4)  # weak DB match — slight penalty
        model_hits.append(c)

    # ── Stage 3: Combine ──────────────────────────────────────────────────────
    # DB hits first (the reliable source), then model candidates; within each
    # group, formula matches first, then by score.
    rank = lambda c: (c.get("formula_match", True), c["score"])
    db_hits.sort(key=rank, reverse=True)
    model_hits.sort(key=rank, reverse=True)
    candidates = (db_hits + model_hits)[:SLOTS]

    if not candidates:
        return jsonify({"error": "No predictions returned"}), 500

    # ── Build response ────────────────────────────────────────────────────────
    top  = candidates[0]
    alts = candidates[1:]

    top["svg"]   = mol_svg(top["smiles"])
    top["props"] = mol_props(top["smiles"])

    for a in alts:
        a["svg"] = mol_svg(a["smiles"])

    return jsonify({"top": top, "alts": alts})


if __name__ == "__main__":
    print("Starting HYS server...")
    print(f"Project dir : {PROJECT_DIR}")
    print(f"Config      : {'✓' if os.path.exists(CONFIG_PATH) else '✗ NOT FOUND'}")
    print(f"Checkpoint  : {'✓' if os.path.exists(CKPT_PATH) else '✗ NOT FOUND'}")
    print(f"Database    : {'✓' if os.path.exists(DB_PATH) else '✗ NOT FOUND (run setup_db.py)'}")
    print("\nOpen http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000)