# scan_dataset.py
# Responsibility: scan the FULL dataset once to lock in vocab + size limits
# before real training begins, so later chunks never get silently truncated

import json
import pandas as pd
from rdkit import RDLogger

from data_loader import get_parquet_files
from smiles_tokenizer import tokenize_smiles, SPECIAL_TOKENS

RDLogger.DisableLog('rdApp.*')  # silence RDKit's noisy C++ logging

CONFIG_PATH = "dataset_config.json"


def scan_all_files(folder_path: str):
    files = get_parquet_files(folder_path)  # no limit -- ALL parquet files
    print(f"Scanning {len(files)} parquet files...")

    unique_tokens = set()
    max_h_peaks = 0
    max_c_peaks = 0
    max_smiles_tokens = 0
    total_molecules = 0

    for i, path in enumerate(files):
        # Only read the 3 columns we need for scanning -- keeps memory low
        df = pd.read_parquet(path, columns=['smiles', 'h_nmr_peaks', 'c_nmr_peaks'])

        for smiles, h_peaks, c_peaks in zip(df['smiles'], df['h_nmr_peaks'], df['c_nmr_peaks']):
            tokens = tokenize_smiles(smiles)
            unique_tokens.update(tokens)

            max_smiles_tokens = max(max_smiles_tokens, len(tokens))
            max_h_peaks = max(max_h_peaks, len(h_peaks))
            max_c_peaks = max(max_c_peaks, len(c_peaks))

        total_molecules += len(df)
        del df  # free this file's memory before moving to the next one

        if (i + 1) % 20 == 0:
            print(f"  Scanned {i+1}/{len(files)} files... "
                  f"(so far: {total_molecules} molecules, {len(unique_tokens)} tokens, "
                  f"max H={max_h_peaks}, max C={max_c_peaks}, max SMILES len={max_smiles_tokens})")

    # Build the final vocabulary: special tokens first, then everything found
    all_tokens = SPECIAL_TOKENS + sorted(unique_tokens)
    token_to_id = {tok: idx for idx, tok in enumerate(all_tokens)}

    config = {
        "total_molecules": total_molecules,
        "vocab_size": len(token_to_id),
        "token_to_id": token_to_id,
        "max_h_peaks": max_h_peaks,
        "max_c_peaks": max_c_peaks,
        "max_smiles_length": max_smiles_tokens + 2 + 5,  # +2 for START/END, +5 safety margin
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nScan complete. Saved config to {CONFIG_PATH}")
    print(f"  Total molecules   : {total_molecules}")
    print(f"  Vocab size        : {config['vocab_size']}")
    print(f"  Max H peaks       : {max_h_peaks}")
    print(f"  Max C peaks       : {max_c_peaks}")
    print(f"  Max SMILES length : {config['max_smiles_length']}")

    return config


if __name__ == "__main__":
    FOLDER = "data/multimodal_spectroscopic_dataset"  # <-- change this
    scan_all_files(FOLDER)