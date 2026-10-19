# chunked_trainer.py
# Responsibility: train on the FULL dataset without loading it all into memory at once,
# by looping over small groups of parquet files ("chunks"), training the same model
# continuously across all of them.
#
# Improvements over v1:
#   - Full checkpoint saves model + optimizer + epoch + best_val_loss
#     so training resumes correctly with Adam momentum intact
#   - All output logged to training.log in addition to console
#   - ReduceLROnPlateau scheduler — LR drops when val loss stops improving
#   - Val loss checked every N chunks (not just once per outer epoch)
#   - Beam-search exact match evaluated at end of each outer epoch

import json
import random
import gc
import os
import shutil
import sys
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import get_parquet_files, load_specific_files
from smiles_tokenizer import SmilesTokenizer
from preprocessor import SpectraToSmilesDataset
from model import SpectraToSmilesModel
from trainer import get_device, train_one_epoch, evaluate, get_memory_status

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = "dataset_config.json"
SAVE_PATH     = "best_model.pt"       # best val-loss checkpoint (full)
RESUME_PATH   = "checkpoint.pt"       # latest checkpoint for resuming (full)
LOG_PATH      = "training.log"
FOLDER        = "data/multimodal_spectroscopic_dataset"

CHUNK_SIZE        = 10    # parquet files per chunk (reduced from 25 — the new,
                          # ~9x larger model made it worth shrinking how much
                          # parquet data is held in memory per chunk; this is
                          # separate from BATCH_SIZE, which controls the model's
                          # own per-batch forward/backward memory footprint)
NUM_VAL_FILES     = 5     # files held out for validation, never trained on
EPOCHS_PER_CHUNK  = 1     # passes over each chunk before moving on
NUM_OUTER_EPOCHS  = 30    # how many times to loop through ALL chunks (total, incl. resumed ones)
BATCH_SIZE        = 32
LEARNING_RATE     = 1e-4
VAL_EVERY_N_CHUNKS = 24   # scaled up from 10 to match the new, higher chunk count
                          # (24 chunks now instead of 10, given 240 files / 10 per
                          # chunk) — keeps validation happening roughly once per
                          # epoch instead of ~2.4x more often than before
FINAL_EVAL_BATCHES = 20   # batches of beam search used for the final post-training report
                          # (bigger than the quick per-epoch check, but capped so it
                          # doesn't take forever — raise this if you want a more thorough
                          # final number, at the cost of a slower wrap-up)

# Scheduled sampling: ramps linearly from 0 -> MAX_SAMPLING_RATE across outer epochs.
# DISABLED for now: the current implementation reruns full self-attention over the
# whole growing sequence at every one of ~80 timesteps, under gradient tracking,
# which appears to push this machine into swap (system RAM pinned at 90%+, chunk
# times went from ~190s to ~6 hours). Needs a KV-cache rewrite before it's safe to
# re-enable at this batch size / MAX_SAMPLING_RATE.
SCHEDULED_SAMPLING    = False
MAX_SAMPLING_RATE     = 0.3

# LR scheduler: reduce LR by FACTOR if val loss doesn't improve for PATIENCE evals
# Raised from patience=3, min_lr=1e-6 after the last full run showed the LR hitting
# the floor by epoch ~21 and then sitting there frozen for 30 more epochs with
# essentially zero net progress. A higher floor keeps some real learning signal
# alive instead of decaying all the way down to near-zero.
LR_PATIENCE = 5
LR_FACTOR   = 0.5
LR_MIN      = 1e-5

# Learning rate to switch to when resuming from checkpoint.pt (None = keep the
# saved one). Set when val loss has flattened but the scheduler hasn't cut
# the LR yet — in the first v1 run, the LR drop to 5e-5 gave the biggest
# late gains.
RESUME_LR   = 3e-5


# ── Logging ───────────────────────────────────────────────────────────────────
_log_file = None

def log(msg):
    """Print to console and append to training.log.
    Uses tqdm.write() instead of print() so log lines don't break/duplicate
    the live progress bars mid-render.
    """
    global _log_file
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    tqdm.write(line)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


# ── Config loading ────────────────────────────────────────────────────────────
def load_config():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    log(f"Config: {config['total_molecules']:,} molecules, "
        f"vocab={config['vocab_size']}, max_h={config['max_h_peaks']}, "
        f"max_c={config['max_c_peaks']}, max_len={config['max_smiles_length']}")
    return config


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler,
                    outer_epoch, chunk_idx, best_val_loss, val_loss):
    """Save a full checkpoint including optimizer and scheduler state."""
    torch.save({
        "input_version":        model.input_version,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "outer_epoch":          outer_epoch,
        "chunk_idx":            chunk_idx,
        "best_val_loss":        best_val_loss,
        "val_loss":             val_loss,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load a full checkpoint. Returns (outer_epoch, chunk_idx, best_val_loss)."""
    ckpt = torch.load(path, map_location=device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            f"\n\nCan't resume from '{path}': its saved weights don't match the "
            f"current model.py architecture (different d_model/layers/etc). Full "
            f"resume needs the architecture to be identical, since optimizer and "
            f"scheduler state are tied to it too — there's no way to partially "
            f"load here.\n\nIf you've intentionally changed the architecture, use "
            f"weights_only=True instead (which falls back to a fresh init on "
            f"mismatch), or delete/rename '{path}' to start clean.\n\n"
            f"Underlying error: {e}"
        ) from e
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    outer_epoch   = ckpt.get("outer_epoch",   1)
    chunk_idx     = ckpt.get("chunk_idx",     0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    log(f"Resumed from {path} "
        f"(outer_epoch={outer_epoch}, chunk={chunk_idx}, "
        f"best_val_loss={best_val_loss:.4f})")
    return outer_epoch, chunk_idx, best_val_loss


def warm_start(model, path, device):
    """Load whatever weights from `path` fit the current model, leaving any new
    or reshaped layers at their fresh init. Returns the loaded checkpoint dict
    (or {} for a legacy weights-only file)."""
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    own = model.state_dict()
    usable = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(usable, strict=False)
    fresh = sorted(k for k in own if k not in usable)
    log(f"Warm start from {path}: loaded {len(usable)}/{len(own)} tensors")
    if fresh:
        log(f"  Fresh init: {', '.join(fresh)}")
    return ckpt if isinstance(ckpt, dict) and "model_state_dict" in ckpt else {}


# ── Data helpers ──────────────────────────────────────────────────────────────
def build_val_loaders(val_files, tokenizer, config):
    """Returns (realistic, clean) loaders over the same held-out molecules:
    realistic = degraded like real user input (used for val loss / checkpoints),
    clean     = complete simulated data (comparable with input_version 1 runs)."""
    log(f"Loading {len(val_files)} held-out validation files...")
    val_df = load_specific_files(val_files)
    loaders = []
    for augment in ("val", None):
        val_ds = SpectraToSmilesDataset(
            val_df, tokenizer,
            max_smiles_length=config["max_smiles_length"],
            max_h_peaks=config["max_h_peaks"],
            max_c_peaks=config["max_c_peaks"],
            augment=augment,
        )
        loaders.append(DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False))
    return tuple(loaders)


def train_on_chunk(model, chunk_files, tokenizer, config,
                   optimizer, loss_fn, device, desc="Chunk", sampling_rate=0.0):
    df = load_specific_files(chunk_files)
    train_ds = SpectraToSmilesDataset(
        df, tokenizer,
        max_smiles_length=config["max_smiles_length"],
        max_h_peaks=config["max_h_peaks"],
        max_c_peaks=config["max_c_peaks"],
        augment="train",
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    loss = 0.0
    for _ in range(EPOCHS_PER_CHUNK):
        loss = train_one_epoch(model, loader, optimizer, loss_fn, device,
                               sampling_rate=sampling_rate, desc=desc)
    del df, train_ds, loader
    gc.collect()
    return loss


# ── Optional beam-search eval ─────────────────────────────────────────────────
# NOTE: no longer called during training — evaluation now only happens once,
# after all epochs finish (see the "Final report" section at the end of
# run_chunked_training). Kept here in case it's useful for ad-hoc mid-training checks.
def run_beam_eval(model, val_loader, tokenizer, device, config, n_batches=5):
    """
    Quick beam-search exact-match check on the first n_batches of val set.
    Returns exact_match_rate in [0, 1].
    """
    try:
        from evaluator import evaluate_model_beam_search
        results = evaluate_model_beam_search(
            model, val_loader, tokenizer, device,
            max_length=config["max_smiles_length"],
            beam_width=5,
            max_batches=n_batches,
        )
        # FIX: evaluate_model_beam_search returns "exact_match_rate", not "exact_match".
        # The old key here never matched, so this silently returned 0.0 every call.
        return results.get("exact_match_rate", 0.0)
    except Exception as e:
        log(f"  Beam eval skipped: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def run_chunked_training(folder_path=FOLDER, resume=True, weights_only=False):
    """
    resume=True, weights_only=False (default): resume from checkpoint.pt if present —
        restores model + optimizer + scheduler + outer_epoch/chunk counters exactly.
    resume=True, weights_only=True: ignore checkpoint.pt entirely. Load ONLY the model
        weights from best_model.pt, and start a brand-new optimizer, scheduler, and
        outer_epoch/chunk counter (outer_epoch starts back at 1). This is the old
        pre-checkpoint behavior — use this if checkpoint.pt's stored epoch count is
        stale relative to NUM_OUTER_EPOCHS, or if you just want a clean continuation.
    """
    global _log_file
    _log_file = open(LOG_PATH, "a")

    log("=" * 60)
    log("HYS chunked training starting")
    log("=" * 60)

    config    = load_config()
    tokenizer = SmilesTokenizer()
    tokenizer.load_vocab(config["token_to_id"])

    all_files = get_parquet_files(folder_path)
    random.seed(42)
    random.shuffle(all_files)

    val_files   = all_files[:NUM_VAL_FILES]
    train_files = all_files[NUM_VAL_FILES:]
    chunks      = [train_files[i:i + CHUNK_SIZE]
                   for i in range(0, len(train_files), CHUNK_SIZE)]

    log(f"Train files: {len(train_files)} → {len(chunks)} chunks of ~{CHUNK_SIZE}")
    log(f"Val files  : {len(val_files)}")

    device = get_device()
    log(f"Device: {device}")

    # ── Build model ────────────────────────────────────────────────────────────
    model = SpectraToSmilesModel(
        vocab_size=config["vocab_size"],
        max_smiles_length=config["max_smiles_length"],
    ).to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    pad_id    = tokenizer.token_to_id["<PAD>"]
    loss_fn   = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_outer = 1
    start_chunk = 0
    best_val_loss = float("inf")

    if resume and weights_only:
        # Force the old behavior: ignore checkpoint.pt's saved epoch/optimizer state
        # entirely, and just load the model weights from best_model.pt (or checkpoint.pt
        # if that's all that exists). Fresh optimizer, fresh scheduler, outer_epoch=1.
        load_path = SAVE_PATH if os.path.exists(SAVE_PATH) else RESUME_PATH
        if os.path.exists(load_path):
            log(f"[weights_only mode] Loading model weights only from {load_path}")
            ckpt = warm_start(model, load_path, device)
            ckpt_version = ckpt.get("input_version", 1)
            if ckpt_version == model.input_version:
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
            else:
                # Inputs/val set changed meaning, so the old best val loss isn't
                # comparable — keeping it would mean best_model.pt never gets saved.
                log(f"[weights_only mode] Upgrading input_version {ckpt_version} -> "
                    f"{model.input_version}: best val loss reset")
                best_val_loss = float("inf")
                if ckpt_version < 2 <= model.input_version:
                    # The IR encoder never trained in v1 (its tokens were masked)
                    # and the formula encoder is new: zero their output layers so
                    # they start silent and fade in, instead of feeding random
                    # noise into the pretrained encoder/decoder.
                    for layer in (model.ir_encoder.fc, model.formula_encoder.projection[2]):
                        nn.init.zeros_(layer.weight)
                        nn.init.zeros_(layer.bias)
                    log("[weights_only mode] Zero-initialized IR/formula output layers")
                backup = SAVE_PATH.replace(".pt", f"_v{ckpt_version}.pt")
                if os.path.exists(SAVE_PATH) and not os.path.exists(backup):
                    shutil.copy2(SAVE_PATH, backup)
                    log(f"[weights_only mode] Backed up {SAVE_PATH} -> {backup}")
            log("[weights_only mode] Optimizer, scheduler, and outer_epoch reset to fresh start")
        else:
            log("[weights_only mode] No checkpoint found — starting from scratch")
    elif resume and os.path.exists(RESUME_PATH):
        start_outer, start_chunk, best_val_loss = load_checkpoint(
            RESUME_PATH, model, optimizer, scheduler, device
        )
        if RESUME_LR is not None:
            for group in optimizer.param_groups:
                group["lr"] = RESUME_LR
            log(f"LR set to RESUME_LR={RESUME_LR:.1e}")
    elif resume and os.path.exists(SAVE_PATH):
        # Legacy: load weights-only checkpoint (old format)
        log(f"Loading weights-only checkpoint from {SAVE_PATH}")
        ckpt = torch.load(SAVE_PATH, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            log("Checkpoint architecture doesn't match the current model.py "
                "(this is expected after changing d_model/layers/etc). "
                "Starting from a fresh random initialization instead.")
            log(f"Underlying error: {e}")
    else:
        log("Starting from scratch")

    val_loader, clean_val_loader = build_val_loaders(val_files, tokenizer, config)

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline = evaluate(model, val_loader, loss_fn, device)
    log(f"Baseline val loss: {baseline:.4f}")
    if best_val_loss == float("inf"):
        best_val_loss = baseline

    mem_str, _ = get_memory_status(device)
    log(f"Baseline memory: {mem_str}")

    # ══════════════════════════════════════════════════════════════════════════
    for outer_epoch in range(start_outer, NUM_OUTER_EPOCHS + 1):
        # Linearly ramp scheduled sampling from 0 -> MAX_SAMPLING_RATE across epochs,
        # same formula train_model() uses. Early epochs stay near-pure teacher forcing
        # (the model's own predictions are still too noisy to be useful training
        # signal); later epochs mix in more of the model's own generated tokens so
        # it practices recovering from its own mistakes, closing the train/inference gap.
        if SCHEDULED_SAMPLING:
            sampling_rate = min(MAX_SAMPLING_RATE, (outer_epoch / NUM_OUTER_EPOCHS) * MAX_SAMPLING_RATE)
        else:
            sampling_rate = 0.0

        log(f"\n{'='*60}")
        log(f"Outer epoch {outer_epoch}/{NUM_OUTER_EPOCHS}  "
            f"(LR={optimizer.param_groups[0]['lr']:.2e}, sampling_rate={sampling_rate:.2f})")
        log(f"{'='*60}")

        epoch_chunks = chunks[start_chunk:] if outer_epoch == start_outer else chunks
        start_chunk  = 0   # only skip chunks on first resumed epoch

        for chunk_idx, chunk_files in enumerate(epoch_chunks):
            actual_idx = (chunk_idx + (start_chunk if outer_epoch == start_outer else 0))
            t0         = time.time()

            chunk_loss = train_on_chunk(
                model, chunk_files, tokenizer, config,
                optimizer, loss_fn, device,
                desc=f"Chunk {actual_idx+1}/{len(chunks)}",
                sampling_rate=sampling_rate,
            )
            elapsed = time.time() - t0

            mem_str, mem_is_high = get_memory_status(device)
            log(f"  Chunk {actual_idx+1:>3}/{len(chunks)} | "
                f"files={len(chunk_files)} | "
                f"train_loss={chunk_loss:.4f} | "
                f"{elapsed:.0f}s | {mem_str}")
            if mem_is_high:
                log(f"  ⚠ WARNING: system RAM usage is high after this chunk")

            # Periodic mid-epoch validation
            if (actual_idx + 1) % VAL_EVERY_N_CHUNKS == 0:
                val_loss = evaluate(model, val_loader, loss_fn, device)
                scheduler.step(val_loss)
                cur_lr = optimizer.param_groups[0]["lr"]
                log(f"  [mid-epoch val] loss={val_loss:.4f}  LR={cur_lr:.2e}")

                # Save latest checkpoint so we can always resume
                save_checkpoint(RESUME_PATH, model, optimizer, scheduler,
                                outer_epoch, actual_idx + 1,
                                best_val_loss, val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(SAVE_PATH, model, optimizer, scheduler,
                                    outer_epoch, actual_idx + 1,
                                    best_val_loss, val_loss)
                    log(f"  *** New best model saved (val_loss={val_loss:.4f}) ***")

        # End-of-epoch full validation
        val_loss = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        log(f"\nEpoch {outer_epoch} end | val_loss={val_loss:.4f} | "
            f"best={best_val_loss:.4f} | LR={cur_lr:.2e}")

        # Save end-of-epoch checkpoint
        save_checkpoint(RESUME_PATH, model, optimizer, scheduler,
                        outer_epoch + 1, 0, best_val_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(SAVE_PATH, model, optimizer, scheduler,
                            outer_epoch, len(chunks),
                            best_val_loss, val_loss)
            log(f"*** New best model saved (val_loss={val_loss:.4f}) ***")

    # ── Final report ──────────────────────────────────────────────────────────
    # Real, full metrics from the evaluator — not the quick per-epoch check,
    # which only samples a handful of batches for speed.
    log(f"\n{'='*60}")
    log("Final evaluation on held-out validation set")
    log(f"{'='*60}")
    # Score the saved best model (what the web app will use), not whatever
    # the last epoch happened to end on.
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device)["model_state_dict"])
        log(f"Evaluating {SAVE_PATH} (best val loss {best_val_loss:.4f})")

    # "realistic" is the degraded val set (what val loss uses); "clean" is
    # complete data, the number to compare against input_version 1 runs.
    from evaluator import evaluate_model_beam_search
    for label, loader in (("realistic input", val_loader), ("clean input", clean_val_loader)):
        final_metrics = evaluate_model_beam_search(
            model, loader, tokenizer, device,
            max_length=config["max_smiles_length"],
            beam_width=5,
            max_batches=FINAL_EVAL_BATCHES,
        )
        log(f"[{label}] validity rate    : {final_metrics['validity_rate']*100:.1f}%")
        log(f"[{label}] exact match rate : {final_metrics['exact_match_rate']*100:.1f}%")
        log(f"[{label}] avg Tanimoto sim : {final_metrics['avg_similarity']:.3f}")
    log(f"Best val loss achieved : {best_val_loss:.4f}")

    log(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    _log_file.close()
    return model


if __name__ == "__main__":
    # False (default): resume exactly from checkpoint.pt — weights, optimizer,
    #   scheduler and epoch counter — and train until NUM_OUTER_EPOCHS.
    # True: ignore checkpoint.pt and start a fresh run (outer_epoch 1) from
    #   best_model.pt's weights. Needed when upgrading to a new input_version
    #   or model.py architecture, since checkpoint.pt can't be fully resumed then.
    WEIGHTS_ONLY_MODE = False

    run_chunked_training(folder_path=FOLDER, resume=True, weights_only=WEIGHTS_ONLY_MODE)