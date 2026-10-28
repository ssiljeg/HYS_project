# HYS — Spectra → Structure

Predicts a molecular structure (SMILES + 2D drawing) from ¹H NMR, ¹³C NMR and IR peaks.

Each prediction runs in two stages:
1. **Database lookup.** Peaks are matched against NMRShiftDB (`search_data/nmrshiftdb.parquet`).
2. **Model.** A Transformer encoder–decoder (`best_model.pt`) generates 20 candidates with beam search. If a molecular formula was given, candidates with the wrong formula drop to the bottom. These fill the remaining slots. (`server.py` also has an optional carbon-count reranking, `RERANK_C_WEIGHT`. It's off because it didn't improve benchmark accuracy.)

Stereochemistry is ignored throughout: 1D NMR and IR can't distinguish
stereoisomers, so the model is trained on stereo-free SMILES and predictions
are scored without stereo.

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
| `best_model.pt` (~400 MB) | Real model predictions. Without it the app shows demo output | Project root. Download it from this repo's Releases page |
| Training data (~20 GB of parquet chunks) | Training and `evaluator.py` only | `data/multimodal_spectroscopic_dataset/` |
| `nmrshiftdb2withsignals.sd` | Only to rebuild the search DB (the built `nmrshiftdb.parquet` is included) | `search_data/` |

## Running

```bash
python server.py        # then open http://localhost:5000
```

### Entering data (what matters for accuracy)

| Field | Format | Notes |
|---|---|---|
| ¹H | `7.34 (d, J=8.1 Hz, 2H), 2.36 (s, 3H)` | Multiplicity and integration help a lot; bare shifts also work |
| ¹³C | `170.7 (C), 128.5 (CH, 2C), 20.6 (CH3)` | **DEPT type and equivalent-carbon count matter most.** `s/d/t/q` off-resonance notation also works; a repeated shift counts as 2C |
| IR | `1750, 1692, 3300` | Peak wavenumbers in cm⁻¹ |
| Formula | `C9H8O4` | Optional. Candidates with the wrong formula are filtered out |

### Expected accuracy

Measured with `benchmark.py` on 200 held-out molecules the model never trained
on (simulated spectra with small shift errors added; stereochemistry ignored):

| Input | Top answer correct | Correct answer in top 5 |
|---|---|---|
| ¹H (with multiplicity and integration) + ¹³C + formula | **73% ± 6** | **86% ± 5** |
| ¹H + ¹³C with DEPT type and counts | 62% ± 7 | 79% ± 6 |
| ¹H + ¹³C with DEPT type | 61% ± 7 | 79% ± 6 |
| ¹H + ¹³C bare shifts | 56% ± 7 | 70% ± 6 |
| ¹H bare shifts + ¹³C bare shifts | 25% ± 6 | 44% ± 7 |
| ¹³C only, with DEPT type and counts | 23% ± 6 | 40% ± 7 |
| ¹³C bare shifts only + formula | 27% ± 6 | 38% ± 7 |

Unless marked "bare shifts", ¹H includes multiplicity and integration. What you
enter matters more than anything else: give ¹H multiplicity and integration,
¹³C DEPT type, and a molecular formula when you have them. The model was
trained on simulated spectra, so expect somewhat lower accuracy on real
experimental data.

If `search_data/nmrshiftdb.parquet` is missing, rebuild it once from the SDF:
```bash
python setup_db.py
```

## Project layout

| File | Purpose |
|---|---|
| `server.py` | Flask web app. Peak parsing, DB lookup, model inference |
| `templates/index.html` | Web UI |
| `model.py` | `SpectraToSmilesModel`: peak/IR encoder + Transformer decoder |
| `evaluator.py` | Beam search generation and validity/accuracy metrics |
| `preprocessor.py` | Peaks → tensors, SMILES → token ids, `SpectraToSmilesDataset` |
| `smiles_tokenizer.py` | SMILES tokenizer |
| `data_loader.py` | Loads/cleans the training parquet chunks |
| `scan_dataset.py` | Scans the dataset and writes `dataset_config.json` (vocab, max lengths) |
| `trainer.py` | Per-epoch train/eval helpers |
| `chunked_trainer.py` | Main training script (streams parquet chunks, writes `best_model.pt` / `checkpoint.pt`) |
| `benchmark.py` | Accuracy on held-out molecules for realistic web-app input (see below) |
| `setup_db.py` | Converts `nmrshiftdb2withsignals.sd` → `nmrshiftdb.parquet` |
| `training_search.py` | Experimental search over the training set. Not wired into the apps yet |

## Architecture

```
H peaks [28,3]  ─→ MLP projection ─→ [28, d_model] ─┐
C peaks [74,2]  ─→ MLP projection ─→ [74, d_model] ─┤
IR [1800]       ─→ CNN            ─→ [L,  d_model] ─┼─→ Transformer encoder ─→ memory
Formula [12]    ─→ MLP projection ─→ [1,  d_model] ─┘   (v2 only)            │
                                                                     (cross-attention)
SMILES tokens ─→ embedding + positions ─→ Transformer decoder ─→ next token
```

d_model=512, 8 heads, 4 encoder / 6 decoder layers, FFN 1536 (~28M params).

Checkpoints record an `input_version`, and `server.py` adapts to it:

| Version | Inputs |
|---|---|
| 1 (original `best_model.pt`) | IR is ignored (masking bug), no formula input, never saw missing values, so the server guesses them |
| 2 (after retraining) | IR used when given, formula token, explicit "unknown" codes for missing ¹H integrations/multiplicities and ¹³C intensities |

## Training

```bash
python scan_dataset.py      # only if the dataset changes
python chunked_trainer.py
python evaluator.py         # beam-search metrics on clean held-out data
python benchmark.py         # accuracy for realistic web-app input
```

By default (`WEIGHTS_ONLY_MODE = False`) `chunked_trainer.py` resumes exactly
from `checkpoint.pt` and trains until `NUM_OUTER_EPOCHS`; raise that number to
train longer. Set `WEIGHTS_ONLY_MODE = True` to start a fresh run from
`best_model.pt`'s weights instead, which is required after changing the
architecture or input version. When upgrading from a version 1 model, that run:
- backs up the old model to `best_model_v1.pt`
- starts the new formula layer from scratch, keeping all other weights
- resets the best validation loss, since validation now uses degraded input

Training samples are randomly degraded (`AUGMENT` in `preprocessor.py`) to look
like real input: missing spectra, hidden integrations/multiplicities/DEPT info,
shift noise, missed peaks, IR rebuilt from picked peaks, and a formula given
half the time. Validation uses the same degradation with a fixed seed, so expect
higher validation loss than with version 1. The final report prints exact-match
and Tanimoto for both **realistic** (degraded) and **clean** input. Only the
clean numbers are comparable with version 1 runs. Use `benchmark.py` to compare
models on realistic web-app input.

## Benchmark

`benchmark.py` runs held-out molecules through the web app's model path (no
database lookup), with added shift noise, for different input scenarios. It
reports how often the exact structure (ignoring stereochemistry) is the top
answer (top-1) or among the five shown (top-5), plus the Tanimoto similarity of
the top answer to the truth. Run it before and after retraining to compare.
