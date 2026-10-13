# Responsibility: Find paqruet files, load them, basic cleaning

import os
import pandas as pd 

COLUMNS_TO_KEEP = ['smiles', 'h_nmr_peaks', 'c_nmr_peaks', 'ir_spectra']

def get_parquet_files(folder_path: str, limit: int = None) -> list:
    """Get a sorted list of parquet file paths. Optionally limit how many."""
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".parquet")
    ]
    files = sorted(files)
    if limit is not None:
        files = files[:limit]
    return files

def load_parquet(file_path: str) -> pd.DataFrame:
    """Load a single parquet file, keeping only the columns we need"""
    df = pd.read_parquet(file_path, columns=COLUMNS_TO_KEEP)
    return df 

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that can be used"""
    before = len(df)

    df = df.dropna()
    df = df.drop_duplicates("smiles")

    after = len(df)
    dropped = before - after 
    if dropped > 0:
        print(f" Cleaned: dropped {dropped} rows ({before} -> {after}")
    
    return df

def load_all_parquets(folder_path: str, limit: int = None) -> pd.DataFrame:
    """ Load and clean every parquet in a folder
        returns a combinded one 
    """
    files = get_parquet_files(folder_path, limit=limit)
    print(f"Found {len(files)} parquet files")

    all_dfs = []
    for i, path in enumerate(files):
        df = load_parquet(path)
        df = clean_dataframe(df)
        all_dfs.append(df)

        if (i + 1) % 20 == 0:
            print(f"  Loaded {i + 1}/{len(files)} files...")

    combined = pd.concat(all_dfs, ignore_index=True)

    # One final dedup pass across files (same molecule can appear in multiple files)
    before = len(combined)
    combined = combined.drop_duplicates('smiles').reset_index(drop=True)
    print(f"\nFinal dataset: {len(combined)} molecules ({before - len(combined)} cross-file duplicates removed)")

    return combined

def load_specific_files(file_paths: list) -> pd.DataFrame:
    """Load and clean a specific list of parquet files (used for one chunk at a time)."""
    all_dfs = []
    for path in file_paths:
        df = load_parquet(path)
        df = clean_dataframe(df)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates('smiles').reset_index(drop=True)
    return combined



# --------- Sanity Check -----------

if __name__ == "__main__":
    FOLDER = "data/multimodal_spectroscopic_dataset" 

    df = load_all_parquets(FOLDER, limit=5)

    print(f"\nShape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print(f"Null values  : {df.isnull().sum().sum()}")
    print(f"Sample SMILES: {df['smiles'].iloc[0]}")