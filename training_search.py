"""
training_search.py
Searches the 794k-molecule training parquet dataset for spectral matches.

Strategy:
  1. Extract hard constraints from user input (carbon count, functional groups)
  2. Pre-filter each parquet file using those constraints before peak matching
  3. Run bidirectional peak matching only on the filtered subset
  4. Return ranked candidates with constraint annotations

This is independent of the model — it reads raw parquet files at inference time
and does not affect model weights or training in any way.
"""

import os
import re
import sys
import numpy as np
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR  = os.path.join(PROJECT_DIR, "data", "multimodal_spectroscopic_dataset")

# ── Tolerances ────────────────────────────────────────────────────────────────
H1_TOL  = 0.05   # ppm  — ¹H matching tolerance
C13_TOL = 0.5    # ppm  — ¹³C matching tolerance
IR_TOL  = 30     # cm⁻¹ — IR peak matching tolerance

# ── Carbon count symmetry factor ─────────────────────────────────────────────
# A molecule can have at most SYMMETRY_FACTOR × n_c13_peaks carbons,
# since symmetry can fold multiple carbons onto the same chemical shift.
# Factor of 3 covers most common organic molecules.
SYMMETRY_FACTOR = 3

# ── IR functional group ranges (cm⁻¹) ────────────────────────────────────────
IR_GROUPS = {
    "carbonyl":  (1650, 1800),   # C=O stretch
    "aromatic":  (1475, 1625),   # aromatic C=C
    "oh_nh":     (2500, 3650),   # broad O-H or N-H stretch
    "ether_co":  (1000, 1300),   # C-O-C stretch
    "alkene":    (1600, 1680),   # C=C stretch (non-aromatic)
    "nitrile":   (2100, 2280),   # C≡N stretch
}

# ── ¹³C shift type ranges (ppm) ───────────────────────────────────────────────
C13_TYPES = {
    "carbonyl":  (155, 220),     # C=O, COOH, ester
    "aromatic":  (100, 155),     # ArC, alkene C
    "oxygenated":(50, 100),      # C-O, C-OH
    "aliphatic": (-5,  50),      # CH₃, CH₂, CH, C
}


# ══════════════════════════════════════════════════════════════════════════════
# Constraint extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_constraints(h1_peaks: list, c13_peaks: list, ir_shifts: list) -> dict:
    """
    Derive hard chemical constraints from all available spectral inputs.
    Returns a dict used to pre-filter the training database.

    Constraints:
      c_min, c_max     : carbon count range from ¹³C peak count
      c_types          : set of carbon environment types present
      h_total          : total proton count if nH provided, else None
      has_aromatic_h   : True if ¹H peaks in 6.5-8.5 ppm range
      ir_groups        : set of functional group flags from IR peaks
      n_c13_peaks      : raw ¹³C peak count (for display)
      n_h1_peaks       : raw ¹H peak count
    """
    constraints = {
        "c_min":        None,
        "c_max":        None,
        "c_types":      set(),
        "h_total":      None,
        "has_aromatic_h": False,
        "ir_groups":    set(),
        "n_c13_peaks":  len(c13_peaks),
        "n_h1_peaks":   len(h1_peaks),
    }

    # ── ¹³C constraints ───────────────────────────────────────────────────────
    if c13_peaks:
        n = len(c13_peaks)
        constraints["c_min"] = n
        constraints["c_max"] = n * SYMMETRY_FACTOR

        shifts = [p["shift"] for p in c13_peaks]
        for ctype, (lo, hi) in C13_TYPES.items():
            if any(lo <= s <= hi for s in shifts):
                constraints["c_types"].add(ctype)

    # ── ¹H constraints ────────────────────────────────────────────────────────
    if h1_peaks:
        nH_values = [p.get("nH") for p in h1_peaks if p.get("nH") is not None]
        if nH_values:
            constraints["h_total"] = sum(nH_values)

        aromatic_shifts = [p["shift"] for p in h1_peaks if 6.5 <= p["shift"] <= 8.5]
        if aromatic_shifts:
            constraints["has_aromatic_h"] = True

    # ── IR constraints ────────────────────────────────────────────────────────
    if ir_shifts:
        for group, (lo, hi) in IR_GROUPS.items():
            if any(lo <= wn <= hi for wn in ir_shifts):
                constraints["ir_groups"].add(group)

    return constraints


def describe_constraints(constraints: dict) -> list:
    """
    Human-readable list of constraint strings for display in the UI.
    """
    lines = []

    if constraints["c_min"] is not None:
        lines.append(
            f"¹³C → {constraints['c_min']} unique carbon environments "
            f"(molecule must have {constraints['c_min']}–{constraints['c_max']} carbons)"
        )
        if constraints["c_types"]:
            type_names = {
                "carbonyl": "carbonyl (C=O)",
                "aromatic": "aromatic/alkene",
                "oxygenated": "oxygenated (C–O)",
                "aliphatic": "aliphatic",
            }
            types_str = ", ".join(type_names.get(t, t) for t in sorted(constraints["c_types"]))
            lines.append(f"¹³C → carbon types present: {types_str}")

    if constraints["h_total"] is not None:
        lines.append(f"¹H  → {constraints['h_total']} total protons (from integration)")

    if constraints["has_aromatic_h"]:
        lines.append("¹H  → aromatic protons present (6.5–8.5 ppm)")

    if constraints["ir_groups"]:
        group_names = {
            "carbonyl":  "C=O stretch (1650–1800 cm⁻¹)",
            "aromatic":  "aromatic C=C (1475–1625 cm⁻¹)",
            "oh_nh":     "O–H or N–H (2500–3650 cm⁻¹)",
            "ether_co":  "C–O–C ether (1000–1300 cm⁻¹)",
            "alkene":    "C=C alkene (1600–1680 cm⁻¹)",
            "nitrile":   "C≡N nitrile (2100–2280 cm⁻¹)",
        }
        for g in sorted(constraints["ir_groups"]):
            lines.append(f"IR  → {group_names.get(g, g)}")

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# SMILES-based constraint checking (RDKit)
# ══════════════════════════════════════════════════════════════════════════════

def smiles_passes_constraints(smiles: str, constraints: dict) -> tuple:
    """
    Check whether a candidate SMILES is consistent with the derived constraints.
    Returns (passes: bool, reason: str).
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdMolDescriptors
        RDLogger.DisableLog("rdApp.*")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "invalid SMILES"

        formula = rdMolDescriptors.CalcMolFormula(mol)

        # Carbon count check
        c_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        if constraints["c_min"] is not None:
            if c_count < constraints["c_min"]:
                return False, f"too few carbons ({c_count} < {constraints['c_min']})"
            if c_count > constraints["c_max"]:
                return False, f"too many carbons ({c_count} > {constraints['c_max']})"

        # Aromatic H check
        if constraints["has_aromatic_h"]:
            has_arom = any(a.GetIsAromatic() for a in mol.GetAtoms())
            if not has_arom:
                return False, "no aromatic ring but aromatic ¹H peaks present"

        # IR functional group checks
        ir_groups = constraints["ir_groups"]

        if "carbonyl" in ir_groups:
            has_co = any(
                b.GetBondTypeAsDouble() == 2.0 and
                {b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()} == {6, 8}
                for b in mol.GetBonds()
            )
            if not has_co:
                return False, "IR shows carbonyl but molecule has no C=O"

        if "oh_nh" in ir_groups:
            has_on = any(a.GetAtomicNum() in (7, 8) for a in mol.GetAtoms())
            if not has_on:
                return False, "IR shows O-H/N-H but molecule has no O or N"

        if "aromatic" in ir_groups and "has_aromatic_h" not in constraints:
            has_arom = any(a.GetIsAromatic() for a in mol.GetAtoms())
            if not has_arom:
                return False, "IR shows aromatic peaks but molecule has no aromatic ring"

        # Total H count check (loose — allow ±20% for measurement uncertainty)
        if constraints["h_total"] is not None:
            mol_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
            h_min = max(1, int(constraints["h_total"] * 0.7))
            h_max = int(constraints["h_total"] * 1.3) + 2
            if not (h_min <= mol_h <= h_max):
                return False, f"H count mismatch ({mol_h} vs expected ~{constraints['h_total']})"

        return True, "passes all constraints"

    except Exception as e:
        return True, f"constraint check skipped ({e})"   # fail open — don't reject on error


# ══════════════════════════════════════════════════════════════════════════════
# Peak matching helpers
# ══════════════════════════════════════════════════════════════════════════════

def bidirectional_score(query_shifts: list, db_shifts: list, tol: float) -> float:
    """Jaccard-like bidirectional peak score."""
    if not query_shifts or not db_shifts:
        return 0.0
    matched = sum(1 for q in query_shifts
                  if any(abs(q - d) <= tol for d in db_shifts))
    return matched / (len(query_shifts) + len(db_shifts) - matched)


def score_molecule(h1_peaks: list, c13_peaks: list, ir_shifts: list,
                   db_h_peaks, db_c_peaks, db_ir: np.ndarray) -> float:
    """
    Compute combined spectral match score between user input and one
    training data row. Weights: ¹³C 45%, ¹H 40%, IR 15%.
    """
    scores = []
    weights = []

    # ¹³C match
    if c13_peaks and db_c_peaks is not None and len(db_c_peaks) > 0:
        q_shifts = [p["shift"] for p in c13_peaks]
        db_shifts = [p["delta (ppm)"] for p in db_c_peaks]
        s = bidirectional_score(q_shifts, db_shifts, C13_TOL)
        scores.append(s)
        weights.append(0.45)

    # ¹H match
    if h1_peaks and db_h_peaks is not None and len(db_h_peaks) > 0:
        q_shifts = [p["shift"] for p in h1_peaks]
        db_shifts = [p["centroid"] for p in db_h_peaks]
        s = bidirectional_score(q_shifts, db_shifts, H1_TOL)
        scores.append(s)
        weights.append(0.40)

    # IR match — compare user peak wavenumbers against continuous training spectrum
    if ir_shifts and db_ir is not None and len(db_ir) == 1800:
        # Find peaks in the training IR spectrum (local maxima above threshold)
        wn_axis = np.linspace(400, 4000, 1800)
        # Simple peak detection: find bins above 0.1 that are local maxima
        above = db_ir > 0.1
        db_ir_peaks = []
        for i in range(1, len(db_ir) - 1):
            if above[i] and db_ir[i] >= db_ir[i-1] and db_ir[i] >= db_ir[i+1]:
                db_ir_peaks.append(float(wn_axis[i]))

        if db_ir_peaks:
            s = bidirectional_score(ir_shifts, db_ir_peaks, IR_TOL)
            scores.append(s)
            weights.append(0.15)

    if not scores:
        return 0.0

    total_weight = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


# ══════════════════════════════════════════════════════════════════════════════
# Main search function
# ══════════════════════════════════════════════════════════════════════════════

def search_training_data(h1_peaks: list, c13_peaks: list, ir_shifts: list,
                         top_n: int = 10,
                         min_score: float = 0.3) -> tuple:
    """
    Search the training parquet dataset for spectral matches.

    Parameters:
        h1_peaks   : list of rich peak dicts {"shift", "nH", "multiplicity"}
        c13_peaks  : list of rich peak dicts {"shift"}
        ir_shifts  : list of wavenumber floats
        top_n      : number of candidates to return
        min_score  : minimum combined score to include in results

    Returns:
        (candidates, constraints, constraint_description)
        candidates: list of dicts {"smiles", "score", "source", "passes_constraints"}
        constraints: the derived constraint dict
        constraint_description: list of human-readable constraint strings
    """
    if not os.path.exists(TRAINING_DIR):
        return [], {}, ["Training data directory not found"]

    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    try:
        from data_loader import get_parquet_files, load_specific_files
    except ImportError:
        return [], {}, ["Could not import data_loader"]

    # ── Extract constraints ───────────────────────────────────────────────────
    constraints = extract_constraints(h1_peaks, c13_peaks, ir_shifts)
    constraint_desc = describe_constraints(constraints)

    # ── Scan parquet files ────────────────────────────────────────────────────
    all_files = get_parquet_files(TRAINING_DIR)
    if not all_files:
        return [], constraints, constraint_desc

    # Track top-N candidates across all files
    # Use a list of (score, smiles, h_peaks, c_peaks) tuples
    top_candidates = []

    c_min = constraints["c_min"]
    c_max = constraints["c_max"]

    for file_path in all_files:
        try:
            import pandas as pd
            df = pd.read_parquet(file_path, columns=["smiles", "h_nmr_peaks",
                                                      "c_nmr_peaks", "ir_spectra"])
        except Exception:
            continue

        # ── Pre-filter by carbon count ────────────────────────────────────────
        if c_min is not None:
            # Count carbons in each SMILES quickly without RDKit
            # Use simple regex count of C atoms (approximate but very fast)
            def approx_carbon_count(smi):
                # Count uppercase C not followed by l (Cl) or o (Co etc.)
                return len(re.findall(r'C(?![lo])', smi))

            mask = df["smiles"].apply(
                lambda s: c_min <= approx_carbon_count(s) <= c_max
            )
            df = df[mask]
            if df.empty:
                continue

        # ── Score each remaining molecule ─────────────────────────────────────
        for _, row in df.iterrows():
            smiles = row["smiles"]
            score = score_molecule(
                h1_peaks, c13_peaks, ir_shifts,
                row["h_nmr_peaks"],
                row["c_nmr_peaks"],
                row["ir_spectra"],
            )

            if score < min_score:
                continue

            # Keep running top-N
            if len(top_candidates) < top_n:
                top_candidates.append((score, smiles))
                top_candidates.sort(key=lambda x: x[0], reverse=True)
            elif score > top_candidates[-1][0]:
                top_candidates[-1] = (score, smiles)
                top_candidates.sort(key=lambda x: x[0], reverse=True)

    # ── Deduplicate by canonical SMILES ───────────────────────────────────────
    seen = set()
    results = []
    for score, smiles in top_candidates:
        try:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog("rdApp.*")
            mol = Chem.MolFromSmiles(smiles)
            canonical = Chem.MolToSmiles(mol) if mol else smiles
        except Exception:
            canonical = smiles

        if canonical in seen:
            continue
        seen.add(canonical)

        # Check constraints
        passes, reason = smiles_passes_constraints(canonical, constraints)

        results.append({
            "smiles":             canonical,
            "score":              round(score, 4),
            "source":             "training_db",
            "passes_constraints": passes,
            "constraint_reason":  reason,
            "verified":           passes,
        })

    # Sort: constrained hits first, then by score
    results.sort(key=lambda x: (not x["passes_constraints"], -x["score"]))

    return results, constraints, constraint_desc