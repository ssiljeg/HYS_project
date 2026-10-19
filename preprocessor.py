# preprocessor.py
# Responsibility: convert raw peaks/spectra + SMILES into PyTorch tensors ready for training

import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from smiles_tokenizer import SmilesTokenizer

# ── Config ──────────────────────────────────────────────────────────────────────
H_FEATURES_PER_PEAK = 3   # [shift, nH, multiplicity_code]
C_FEATURES_PER_PEAK = 2   # [shift, intensity]

MULTIPLICITY_MAP = {
    "s": 0, "d": 1, "t": 2, "q": 3, "p": 4,
    "m": 5, "dd": 6, "ddd": 7, "dt": 8, "td": 9,
}
DEFAULT_MULTIPLICITY = 5

# "Not provided by the user" codes (input_version 2 models only — v1 models
# never saw these and need guessed values instead; see server.build_h_peaks).
MULTIPLICITY_UNKNOWN = 10
NH_UNKNOWN = -1           # not 0: the dataset has genuine 0H peaks

# 13C peak intensity in the (simulated) training data is a DEPT-like signal:
#     intensity = n_equivalent_carbons × (0.0503 + 0.0352 × n_attached_H)
# e.g. C 0.0503, CH 0.0855, CH2 0.1207, CH3 0.1559, 2×CH (phenyl) 0.171.
# The model relies on it heavily, so users who know DEPT type / equivalent
# carbon counts should give them (see carbon_intensity).
C13_BASE_INTENSITY = 0.0503
C13_PER_H_INTENSITY = 0.0352
C13_INTENSITY_UNKNOWN = -1.0   # v2 models only; outside the real range (> 0)
# Median 13C intensity in the training data — the least-bad stand-in for an
# unknown intensity with input_version 1 models, which never saw "unknown".
C13_TYPICAL_INTENSITY = 0.0512

# IR spectra are 1800 points spanning 400-4000 cm-1
IR_LEN = 1800
IR_WN_AXIS = np.linspace(400, 4000, IR_LEN)

# Molecular formula -> fixed-length count vector (+ a "formula known" flag)
FORMULA_ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
FORMULA_FEATURES = len(FORMULA_ELEMENTS) + 2   # + "other elements" + known flag


def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Scale a spectrum to [0, 1] by dividing by its own max."""
    max_val = spectrum.max()
    if max_val > 0:
        return spectrum / max_val
    return spectrum


def encode_h_peaks(peaks, max_h_peaks) -> np.ndarray:
    """Converts a list of 1H peak dicts into a fixed-size [max_h_peaks, 3] array.
    nH=None / category=None mean "unknown" and get the *_UNKNOWN codes."""
    features = np.zeros((max_h_peaks, H_FEATURES_PER_PEAK), dtype=np.float32)
    for i, peak in enumerate(peaks):
        if i >= max_h_peaks:
            break
        shift = peak["centroid"]
        nH = NH_UNKNOWN if peak["nH"] is None else peak["nH"]
        if peak["category"] is None:
            multiplicity_code = MULTIPLICITY_UNKNOWN
        else:
            multiplicity_code = MULTIPLICITY_MAP.get(peak["category"], DEFAULT_MULTIPLICITY)
        features[i] = [shift, nH, multiplicity_code]
    return features


def encode_c_peaks(peaks, max_c_peaks) -> np.ndarray:
    """Converts a list of 13C peak dicts into a fixed-size [max_c_peaks, 2] array.
    intensity=None (unknown) gets C13_INTENSITY_UNKNOWN."""
    features = np.zeros((max_c_peaks, C_FEATURES_PER_PEAK), dtype=np.float32)
    for i, peak in enumerate(peaks):
        if i >= max_c_peaks:
            break
        shift = peak["delta (ppm)"]
        intensity = peak.get("intensity")
        if intensity is None:
            intensity = C13_INTENSITY_UNKNOWN
        features[i] = [shift, intensity]
    return features


def carbon_intensity(n_h, n_equivalent=1):
    """Training-data intensity for a carbon with n_h attached H (DEPT type)."""
    return n_equivalent * (C13_BASE_INTENSITY + C13_PER_H_INTENSITY * n_h)


_C13_LEVELS = {carbon_intensity(h, k): h for h in range(4) for k in range(1, 4)}

def dept_only_intensity(intensity):
    """Keep a peak's DEPT type (C/CH/CH2/CH3) but forget how many equivalent
    carbons it stands for — what a user with DEPT data but no symmetry
    assignment would give."""
    n_h = _C13_LEVELS[min(_C13_LEVELS, key=lambda lvl: abs(lvl - intensity))]
    return carbon_intensity(n_h)


# ── Target SMILES ──────────────────────────────────────────────────────────────
def strip_stereo(smiles):
    """Canonical SMILES without stereochemistry (isotopes kept), or None if
    unparseable. 1D NMR/IR can't distinguish stereoisomers, so the model is
    trained, and scored, on stereo-free structures."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


# ── Molecular formula ──────────────────────────────────────────────────────────
def formula_counts_from_smiles(smiles):
    """Element counts (incl. implicit H) for a SMILES, or None if it doesn't parse."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    counts = {}
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
        if atom.GetTotalNumHs():
            counts["H"] = counts.get("H", 0) + atom.GetTotalNumHs()
    return counts


def parse_formula(text):
    """'C9H8O4' -> {'C': 9, 'H': 8, 'O': 4}. Raises ValueError if unparseable."""
    text = text.strip().replace(" ", "")
    if not text or not re.fullmatch(r"([A-Z][a-z]?\d*)+", text):
        raise ValueError(f"Can't read molecular formula '{text}'")
    counts = {}
    for elem, n in re.findall(r"([A-Z][a-z]?)(\d*)", text):
        counts[elem] = counts.get(elem, 0) + (int(n) if n else 1)
    return counts


def encode_formula(counts) -> np.ndarray:
    """Element counts dict (or None = unknown) -> [FORMULA_FEATURES] array."""
    features = np.zeros(FORMULA_FEATURES, dtype=np.float32)
    if counts is None:
        return features
    for i, elem in enumerate(FORMULA_ELEMENTS):
        features[i] = counts.get(elem, 0) / 10.0
    features[-2] = sum(n for e, n in counts.items() if e not in FORMULA_ELEMENTS) / 10.0
    features[-1] = 1.0
    return features


# ── IR helpers ─────────────────────────────────────────────────────────────────
def ir_from_peaks(wavenumbers, width=20.0) -> np.ndarray:
    """Rebuild a continuous 1800-point IR spectrum from a list of peak
    wavenumbers (Lorentzian per peak, normalized to [0, 1]). This is what the
    web app does with pasted peaks, and training mimics it via augmentation."""
    ir = np.zeros(IR_LEN, dtype=np.float32)
    for wn in wavenumbers:
        ir += (width ** 2 / ((IR_WN_AXIS - wn) ** 2 + width ** 2)).astype(np.float32)
    return normalize_spectrum(ir)


def pick_ir_peaks(spectrum, max_peaks=12, min_height=0.15, min_separation=25.0):
    """Wavenumbers of the strongest local maxima in a normalized IR spectrum,
    roughly what a person would read off and type in. Maxima closer than
    min_separation cm-1 to a stronger one count as the same band."""
    s = np.asarray(spectrum)
    is_peak = (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] >= min_height)
    idx = np.where(is_peak)[0] + 1
    picked = []
    for wn in IR_WN_AXIS[idx[np.argsort(s[idx])[::-1]]]:
        if all(abs(wn - p) >= min_separation for p in picked):
            picked.append(wn)
            if len(picked) == max_peaks:
                break
    return picked


# ── Augmentation ───────────────────────────────────────────────────────────────
# Training data is clean simulated spectra; real users paste partial, noisy,
# experimental peak lists. Each sample is randomly degraded so the model learns
# to cope with what it'll actually be given. All probabilities are per molecule.
AUGMENT = {
    "drop_h":          0.15,  # no 1H spectrum at all
    "drop_c":          0.15,  # no 13C spectrum at all (never both)
    "hide_nh":         0.40,  # 1H integrations unknown
    "hide_mult":       0.40,  # 1H multiplicities unknown
    "c_int_unknown":   0.45,  # 13C intensities unknown (bare shift lists)
    "c_int_dept_only": 0.25,  # 13C DEPT type known, equivalent-carbon count not
                              # remainder (0.30): full intensity (DEPT + counts)
    "shift_noise":     0.70,  # add experimental-vs-simulated shift error
    "h_noise_sd":      0.05,  # ppm
    "c_noise_sd":      1.0,   # ppm
    "peak_drop":       0.05,  # per-peak chance of being missed (when noise applied)
    "ir_none":         0.45,  # no IR
    "ir_from_peaks":   0.35,  # IR rebuilt from picked peaks (as the web app does)
                              # remainder (0.20): full simulated IR spectrum
    "formula_known":   0.50,
}


def augment_sample(h_peaks, c_peaks, ir, rng, cfg=AUGMENT):
    """Returns degraded (h_peaks, c_peaks, ir, formula_known)."""
    h_peaks = [dict(p) for p in h_peaks]
    c_peaks = [dict(p) for p in c_peaks]

    r = rng.random()
    if r < cfg["drop_h"]:
        h_peaks = []
    elif r < cfg["drop_h"] + cfg["drop_c"]:
        c_peaks = []

    if rng.random() < cfg["hide_nh"]:
        for p in h_peaks: p["nH"] = None
    if rng.random() < cfg["hide_mult"]:
        for p in h_peaks: p["category"] = None
    r = rng.random()
    if r < cfg["c_int_unknown"]:
        for p in c_peaks: p["intensity"] = None
    elif r < cfg["c_int_unknown"] + cfg["c_int_dept_only"]:
        for p in c_peaks: p["intensity"] = dept_only_intensity(p["intensity"])

    if rng.random() < cfg["shift_noise"]:
        for p in h_peaks: p["centroid"] += rng.normal(0, cfg["h_noise_sd"])
        for p in c_peaks: p["delta (ppm)"] += rng.normal(0, cfg["c_noise_sd"])
        if len(h_peaks) > 1:
            h_peaks = [p for p in h_peaks if rng.random() >= cfg["peak_drop"]] or h_peaks[:1]
        if len(c_peaks) > 1:
            c_peaks = [p for p in c_peaks if rng.random() >= cfg["peak_drop"]] or c_peaks[:1]

    r = rng.random()
    if r < cfg["ir_none"]:
        ir = np.zeros_like(ir)
    elif r < cfg["ir_none"] + cfg["ir_from_peaks"]:
        ir = ir_from_peaks(pick_ir_peaks(ir))

    return h_peaks, c_peaks, ir, rng.random() < cfg["formula_known"]


def make_peak_mask(num_real_peaks: int, max_peaks: int) -> np.ndarray:
    """Returns a [max_peaks] array of 1s (real peak) and 0s (padding)."""
    mask = np.zeros(max_peaks, dtype=np.float32)
    real_count = min(num_real_peaks, max_peaks)
    mask[:real_count] = 1.0
    return mask


class SpectraToSmilesDataset(Dataset):
    """
    PyTorch Dataset for the encoder-decoder pipeline.
    Each item returns:
        h_peaks   [max_h_peaks, 3]
        h_mask    [max_h_peaks]
        c_peaks   [max_c_peaks, 2]
        c_mask    [max_c_peaks]
        ir        [1800]
        formula   [FORMULA_FEATURES]
        target_ids [max_smiles_length]

    augment: None  -> clean data, formula always given
             "train" -> random degradation (see AUGMENT), different every epoch
             "val"   -> same degradation, but seeded per index so val loss is
                        comparable between evaluations
    """

    def __init__(self, df, tokenizer: SmilesTokenizer, max_smiles_length: int,
                max_h_peaks: int, max_c_peaks: int, augment=None):
        self.h_nmr_peaks = df['h_nmr_peaks'].values
        self.c_nmr_peaks = df['c_nmr_peaks'].values
        self.ir = df['ir_spectra'].values
        self.smiles = df['smiles'].values
        # Training targets have stereochemistry removed (see strip_stereo)
        self.targets = [strip_stereo(s) or s for s in self.smiles]

        self.tokenizer = tokenizer
        self.max_smiles_length = max_smiles_length
        self.max_h_peaks = max_h_peaks
        self.max_c_peaks = max_c_peaks
        self.augment = augment
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        h_peaks = self.h_nmr_peaks[idx]
        c_peaks = self.c_nmr_peaks[idx]
        ir_array = normalize_spectrum(np.array(self.ir[idx], dtype=np.float32))
        formula_known = True

        if self.augment:
            rng = np.random.default_rng(idx) if self.augment == "val" else self._rng
            h_peaks, c_peaks, ir_array, formula_known = augment_sample(
                h_peaks, c_peaks, ir_array, rng)

        h_features = encode_h_peaks(h_peaks, self.max_h_peaks)
        c_features = encode_c_peaks(c_peaks, self.max_c_peaks)

        h_mask = make_peak_mask(len(h_peaks), self.max_h_peaks)
        c_mask = make_peak_mask(len(c_peaks), self.max_c_peaks)

        formula = encode_formula(
            formula_counts_from_smiles(self.smiles[idx]) if formula_known else None)
        target_ids = self.tokenizer.encode(self.targets[idx], self.max_smiles_length)

        return (
            torch.tensor(h_features, dtype=torch.float32),
            torch.tensor(h_mask, dtype=torch.float32),
            torch.tensor(c_features, dtype=torch.float32),
            torch.tensor(c_mask, dtype=torch.float32),
            torch.tensor(ir_array, dtype=torch.float32),
            torch.tensor(formula, dtype=torch.float32),
            torch.tensor(target_ids, dtype=torch.long),
        )


def split_dataset(df, val_frac=0.15, test_frac=0.15, seed=42):
    """Splits a DataFrame into train/val/test DataFrames."""
    n = len(df)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df = df_shuffled.iloc[:n_train]
    val_df = df_shuffled.iloc[n_train:n_train + n_val]
    test_df = df_shuffled.iloc[n_train + n_val:]

    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def make_dataloaders(df, tokenizer, max_smiles_length, max_h_peaks, max_c_peaks,
                     batch_size=64, val_frac=0.15, test_frac=0.15, seed=42):
    """Full pipeline: split DataFrame, wrap in Datasets, return DataLoaders."""
    train_df, val_df, test_df = split_dataset(df, val_frac, test_frac, seed)

    train_ds = SpectraToSmilesDataset(train_df, tokenizer, max_smiles_length, max_h_peaks, max_c_peaks)
    val_ds = SpectraToSmilesDataset(val_df, tokenizer, max_smiles_length, max_h_peaks, max_c_peaks)
    test_ds = SpectraToSmilesDataset(test_df, tokenizer, max_smiles_length, max_h_peaks, max_c_peaks)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ── Sanity check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_all_parquets

    FOLDER = "data/multimodal_spectroscopic_dataset"

    df = load_all_parquets(FOLDER, limit=5)

    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(df['smiles'].tolist())

    MAX_SMILES_LENGTH = 81
    MAX_H_PEAKS = 28
    MAX_C_PEAKS = 74

    train_loader, val_loader, test_loader = make_dataloaders(
        df, tokenizer, MAX_SMILES_LENGTH, MAX_H_PEAKS, MAX_C_PEAKS, batch_size=32
    )

    h_peaks, h_mask, c_peaks, c_mask, ir, formula, target_ids = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"  H peaks    : {h_peaks.shape}")
    print(f"  H mask     : {h_mask.shape}")
    print(f"  C peaks    : {c_peaks.shape}")
    print(f"  C mask     : {c_mask.shape}")
    print(f"  IR         : {ir.shape}")
    print(f"  Formula    : {formula.shape}")
    print(f"  Target IDs : {target_ids.shape}")