"""
setup_db.py — run once to convert nmrshiftdb2withsignals.sd → nmrshiftdb.parquet
Place in HYS_project/ and run: python setup_db.py
Output: search_data/nmrshiftdb.parquet
"""

import os, re, sys, time
import pandas as pd

SDF_PATH = os.path.join("search_data", "nmrshiftdb2withsignals.sd")
OUT_PATH = os.path.join("search_data", "nmrshiftdb.parquet")

def parse_shifts(raw):
    shifts = []
    for entry in raw.strip().split("|"):
        entry = entry.strip()
        if entry:
            try:
                shifts.append(float(entry.split(";")[0]))
            except ValueError:
                pass
    return sorted(shifts)

def fix_molblock(molblock):
    """
    NMRShiftDB molblocks have only 2 header lines (name + blank).
    V2000 format requires exactly 3 (name + program + comment).
    Insert a missing blank line so RDKit can parse it.
    """
    lines = molblock.split("\n")
    # lines[0] = "CDKD" (name), lines[1] = "" (blank), lines[2] = counts line
    # We need: name, program, comment, counts
    if len(lines) >= 3 and lines[1] == "" and lines[2].strip().endswith("V2000"):
        lines.insert(2, "")  # insert extra blank as comment line
    return "\n".join(lines)

def main():
    if not os.path.exists(SDF_PATH):
        print(f"ERROR: {SDF_PATH} not found.")
        sys.exit(1)

    os.makedirs("search_data", exist_ok=True)

    print(f"Reading {SDF_PATH} ...")
    with open(SDF_PATH, "r", errors="ignore") as f:
        content = f.read()

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    records = content.split("$$$$")
    print(f"Found {len(records):,} records. Parsing...")

    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        rdkit_ok = True
    except ImportError:
        print("WARNING: rdkit not installed.")
        rdkit_ok = False

    rows = []
    t0   = time.time()
    ok   = 0
    fail = 0

    for i, rec in enumerate(records):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  {i:,}/{len(records):,} ({i/len(records)*100:.0f}%) "
                  f"— {elapsed:.0f}s — {len(rows):,} entries, "
                  f"{ok} mol ok / {fail} fail", flush=True)

        rec = rec.strip()
        if not rec:
            continue

        # ── SMILES from molblock ──────────────────────────────────────────────
        smiles = ""
        if rdkit_ok:
            end_idx = rec.find("M  END")
            if end_idx != -1:
                molblock = rec[:end_idx + 6] + "\n"
                molblock = fix_molblock(molblock)
                try:
                    mol = Chem.MolFromMolBlock(molblock, sanitize=True,
                                               removeHs=True)
                    if mol:
                        smiles = Chem.MolToSmiles(mol)
                        ok += 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1

        if not smiles:
            continue

        # ── InChI key ─────────────────────────────────────────────────────────
        inchi_m   = re.search(r'> <INChI key>\n(.+)', rec)
        inchi_key = inchi_m.group(1).strip() if inchi_m else ""

        sol_m   = re.search(r'> <Solvent>\n(.+)', rec)
        solvent = sol_m.group(1).strip() if sol_m else ""

        # ── 13C spectra ───────────────────────────────────────────────────────
        for raw in re.findall(r'> <Spectrum 13C \d+>\n(.+)', rec):
            shifts = parse_shifts(raw)
            if shifts:
                rows.append({
                    "smiles":    smiles,
                    "inchi_key": inchi_key,
                    "nmr_type":  "13C",
                    "shifts":    shifts,
                    "solvent":   solvent,
                })

        # ── 1H spectra ────────────────────────────────────────────────────────
        for raw in re.findall(r'> <Spectrum 1H \d+>\n(.+)', rec):
            shifts = parse_shifts(raw)
            if shifts:
                rows.append({
                    "smiles":    smiles,
                    "inchi_key": inchi_key,
                    "nmr_type":  "1H",
                    "shifts":    shifts,
                    "solvent":   solvent,
                })

    print(f"\nParsed {len(rows):,} spectra from {len(records):,} records.")
    print(f"RDKit: {ok} molecules ok, {fail} failed")

    if not rows:
        print("ERROR: no rows parsed.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Saved → {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"\nBreakdown:")
    print(df["nmr_type"].value_counts().to_string())
    print(f"Unique molecules: {df['smiles'].nunique():,}")
    print(f"Done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()