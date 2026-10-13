# preprocessor.py
# Responsibility: convert raw peaks/spectra + SMILES into PyTorch tensors ready for training

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


def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Scale a spectrum to [0, 1] by dividing by its own max."""
    max_val = spectrum.max()
    if max_val > 0:
        return spectrum / max_val
    return spectrum


def encode_h_peaks(peaks, max_h_peaks) -> np.ndarray:
    """Converts a list of 1H peak dicts into a fixed-size [max_h_peaks, 3] array."""
    features = np.zeros((max_h_peaks, H_FEATURES_PER_PEAK), dtype=np.float32)
    for i, peak in enumerate(peaks):
        if i >= max_h_peaks:
            break
        shift = peak["centroid"]
        nH = peak["nH"]
        multiplicity_code = MULTIPLICITY_MAP.get(peak["category"], DEFAULT_MULTIPLICITY)
        features[i] = [shift, nH, multiplicity_code]
    return features


def encode_c_peaks(peaks, max_c_peaks) -> np.ndarray:
    """Converts a list of 13C peak dicts into a fixed-size [max_c_peaks, 2] array."""
    features = np.zeros((max_c_peaks, C_FEATURES_PER_PEAK), dtype=np.float32)
    for i, peak in enumerate(peaks):
        if i >= max_c_peaks:
            break
        shift = peak["delta (ppm)"]
        intensity = peak["intensity"]
        features[i] = [shift, intensity]
    return features


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
        target_ids [max_smiles_length]
    """

    def __init__(self, df, tokenizer: SmilesTokenizer, max_smiles_length: int,
                max_h_peaks: int, max_c_peaks: int):
        self.h_nmr_peaks = df['h_nmr_peaks'].values
        self.c_nmr_peaks = df['c_nmr_peaks'].values
        self.ir = df['ir_spectra'].values
        self.smiles = df['smiles'].values

        self.tokenizer = tokenizer
        self.max_smiles_length = max_smiles_length
        self.max_h_peaks = max_h_peaks
        self.max_c_peaks = max_c_peaks

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        h_features = encode_h_peaks(self.h_nmr_peaks[idx], self.max_h_peaks)
        c_features = encode_c_peaks(self.c_nmr_peaks[idx], self.max_c_peaks)

        h_mask = make_peak_mask(len(self.h_nmr_peaks[idx]), self.max_h_peaks)
        c_mask = make_peak_mask(len(self.c_nmr_peaks[idx]), self.max_c_peaks)

        ir_array = normalize_spectrum(np.array(self.ir[idx], dtype=np.float32))
        target_ids = self.tokenizer.encode(self.smiles[idx], self.max_smiles_length)

        return (
            torch.tensor(h_features, dtype=torch.float32),
            torch.tensor(h_mask, dtype=torch.float32),
            torch.tensor(c_features, dtype=torch.float32),
            torch.tensor(c_mask, dtype=torch.float32),
            torch.tensor(ir_array, dtype=torch.float32),
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

    h_peaks, h_mask, c_peaks, c_mask, ir, target_ids = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"  H peaks    : {h_peaks.shape}")
    print(f"  H mask     : {h_mask.shape}")
    print(f"  C peaks    : {c_peaks.shape}")
    print(f"  C mask     : {c_mask.shape}")
    print(f"  IR         : {ir.shape}")
    print(f"  Target IDs : {target_ids.shape}")