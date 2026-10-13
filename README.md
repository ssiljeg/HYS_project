# HYS — Spectra → Structure

Predicts a molecular structure (SMILES + 2D drawing) from ¹H NMR, ¹³C NMR and IR peaks.

Each prediction runs in two stages:
1. **Database lookup.** Peaks are matched against NMRShiftDB (`search_data/nmrshiftdb.parquet`).
2. **Model.** A Transformer encoder–decoder (`best_model.pt`) generates candidates with beam search to fill the remaining slots.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Files not included in the repo

These are too large for GitHub and are excluded by `.gitignore`:

| File | Needed for | Where it goes |
|---|---|---|
| `best_model.pt` (~400 MB) | Real model predictions. Without it the apps show demo output | Project root. Download it from this repo's Releases page |
| Training data (~20 GB of parquet chunks) | Training and `evaluator.py` only | `data/multimodal_spectroscopic_dataset/` |
| `nmrshiftdb2withsignals.sd` | Only to rebuild the search DB (the built `nmrshiftdb.parquet` is included) | `search_data/` |

## Running

**Web app (recommended):**
```bash
python server.py        # then open http://localhost:5000
```
The web app accepts annotated ¹H peaks such as `7.34 (d, J=8.1 Hz, 1H)`.

**Desktop app (Tkinter):**
```bash
python app.py
```
The desktop app reuses `server.py`'s database lookup and model code.

If `search_data/nmrshiftdb.parquet` is missing, rebuild it once from the SDF:
```bash
python setup_db.py
```

## Project layout

| File | Purpose |
|---|---|
| `server.py` | Flask web app. Peak parsing, DB lookup, model inference |
| `templates/index.html` | Web UI |
| `app.py` | Desktop UI (Tkinter) |
| `model.py` | `SpectraToSmilesModel`: peak/IR encoder + Transformer decoder |
| `evaluator.py` | Beam search generation and validity/accuracy metrics |
| `preprocessor.py` | Peaks → tensors, SMILES → token ids, `SpectraToSmilesDataset` |
| `smiles_tokenizer.py` | SMILES tokenizer |
| `data_loader.py` | Loads/cleans the training parquet chunks |
| `scan_dataset.py` | Scans the dataset and writes `dataset_config.json` (vocab, max lengths) |
| `trainer.py` | Per-epoch train/eval helpers |
| `chunked_trainer.py` | Main training script (streams parquet chunks, writes `best_model.pt` / `checkpoint.pt`) |
| `setup_db.py` | Converts `nmrshiftdb2withsignals.sd` → `nmrshiftdb.parquet` |
| `training_search.py` | Experimental search over the training set. Not wired into the apps yet |

## Architecture

```
H peaks [28,3]  ─→ MLP projection ─→ [28, d_model] ─┐
C peaks [74,2]  ─→ MLP projection ─→ [74, d_model] ─┼─→ Transformer encoder ─→ memory
IR [1800]       ─→ CNN            ─→ [1,  d_model] ─┘                            │
                                                                     (cross-attention)
SMILES tokens ─→ embedding + positions ─→ Transformer decoder ─→ next token
```

Current checkpoint: d_model=512, 8 heads, 4 encoder / 6 decoder layers, FFN 1536 (~28M params).

## Training

```bash
python scan_dataset.py      # only if the dataset changes
python chunked_trainer.py   # resumes from checkpoint.pt if present
python evaluator.py         # beam-search metrics on the held-out files
```

## Demo data: Aspirin (CDCl₃)

- **¹³C (ppm):** 20.6, 122.3, 124.0, 126.0, 131.4, 134.2, 151.0, 169.6, 170.7
- **¹H (ppm):** 2.43, 7.10, 7.35, 7.62, 8.12, 11.89
- **IR (cm⁻¹):** 760, 1010, 1183, 1368, 1456, 1605, 1692, 1750, 2900, 3300
