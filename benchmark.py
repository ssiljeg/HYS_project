# benchmark.py
# Responsibility: measure accuracy the way the web app is actually used —
# pasted peak lists (optionally without integrations/multiplicities, with
# experimental-style shift error, with or without a molecular formula) —
# on held-out molecules the model never trained on.
#
# Runs through server.model_predict + server.apply_formula, i.e. the exact
# web-app model path (the NMRShiftDB lookup is skipped: it would only find
# molecules that happen to be in that database).
#
# Usage:  python benchmark.py            (default 50 molecules, ~10-15 min on CPU)
#         python benchmark.py --n 200    (more molecules = more reliable numbers)

import argparse
import random
import time

import numpy as np

import server
from data_loader import get_parquet_files, load_specific_files
from evaluator import tanimoto_similarity
from preprocessor import carbon_intensity
from chunked_trainer import FOLDER, NUM_VAL_FILES

# (name, 1H input, 13C input, use formula)
#   1H:  "ann" = shift (mult, nH)        "shifts" = bare shifts      None = not given
#   13C: "full" = shift (CH2, 2C)        "dept" = shift (CH2)        "shifts" = bare
SCENARIOS = [
    ("1H + 13C DEPT+counts",            "ann",    "full",   False),
    ("1H + 13C DEPT",                   "ann",    "dept",   False),
    ("1H + 13C shifts",                 "ann",    "shifts", False),
    ("1H + 13C shifts + formula",       "ann",    "shifts", True),
    ("1H shifts + 13C shifts",          "shifts", "shifts", False),
    ("13C DEPT+counts only",            None,     "full",   False),
    ("13C shifts only + formula",       None,     "shifts", True),
]

# 13C intensity -> (attached H, equivalent carbons); see preprocessor.carbon_intensity
_C13_LEVELS = {carbon_intensity(h, k): (h, k) for h in range(4) for k in range(1, 4)}


def to_rich_h(peaks, annotated, rng, noise):
    out = []
    for p in peaks:
        out.append({
            "shift": p["centroid"] + (rng.normal(0, noise[0]) if noise else 0.0),
            "nH": int(p["nH"]) if annotated else None,
            "multiplicity": server.normalize_multiplicity(p["category"]) if annotated else None,
            "J": [],
        })
    return out


def to_rich_c(peaks, mode, rng, noise):
    out = []
    for p in peaks:
        n_h, count = _C13_LEVELS[min(_C13_LEVELS, key=lambda lvl: abs(lvl - p["intensity"]))]
        out.append({
            "shift": p["delta (ppm)"] + (rng.normal(0, noise[1]) if noise else 0.0),
            "nH": n_h if mode in ("full", "dept") else None,
            "count": count if mode == "full" else None,
            "multiplicity": None, "J": [],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="molecules to test")
    ap.add_argument("--no-noise", action="store_true",
                    help="use exact simulated shifts (default adds ±0.05 ppm 1H / ±1 ppm 13C)")
    args = ap.parse_args()
    noise = None if args.no_noise else (0.05, 1.0)

    model, _ = server.get_model()
    print(f"Model input_version={model.input_version} | n={args.n} | "
          f"shift noise={'off' if noise is None else noise} | "
          f"rerank weight={server.RERANK_C_WEIGHT}\n")

    files = get_parquet_files(FOLDER)
    random.seed(42)
    random.shuffle(files)                      # same split as chunked_trainer.py
    df = load_specific_files(files[:NUM_VAL_FILES][:1]).sample(args.n, random_state=0)

    results = {name: {"top1": 0, "top5": 0, "sim": 0.0} for name, *_ in SCENARIOS}
    t0 = time.time()
    for i, row in enumerate(df.itertuples()):
        truth = server.canonical(row.smiles)
        formula = server.formula_counts_from_smiles(row.smiles)
        for name, h_mode, c_mode, use_formula in SCENARIOS:
            rng = np.random.default_rng(i)     # same noise for every model version
            h = to_rich_h(row.h_nmr_peaks, h_mode == "ann", rng, noise) if h_mode else []
            c = to_rich_c(row.c_nmr_peaks, c_mode, rng, noise) if c_mode else []
            fc = formula if use_formula else None
            cands = server.model_predict(h, c, [], beam_width=20,
                                         formula_counts=fc)
            cands = server.apply_formula(cands, fc)
            smiles = [c["smiles"] for c in cands]
            r = results[name]
            r["top1"] += bool(smiles) and smiles[0] == truth
            r["top5"] += truth in smiles[:5]
            r["sim"]  += tanimoto_similarity(truth, smiles[0]) if smiles else 0.0
        done = i + 1
        print(f"\r  {done}/{args.n} molecules  ({(time.time() - t0) / done:.1f}s each)",
              end="", flush=True)

    n = args.n
    # ± = 95% confidence margin (normal approximation) — differences smaller
    # than this between two runs are likely noise.
    pm = lambda k: f"{k/n:4.0%} ±{1.96 * ((k/n) * (1 - k/n) / n) ** 0.5:3.0%}"
    print(f"\n\n{'scenario':32s} {'top-1':>10s} {'top-5':>10s} {'similarity':>11s}")
    for name, r in results.items():
        print(f"{name:32s} {pm(r['top1']):>10s} {pm(r['top5']):>10s} {r['sim']/n:11.2f}")
    print("\n± = 95% confidence margin; differences smaller than this are likely noise."
          "\ntop-1/top-5 = exact structure (ignoring stereochemistry) is the first / among the five shown;"
          "\nsimilarity = Tanimoto of the first answer to the truth (1.0 = identical).")


if __name__ == "__main__":
    main()
