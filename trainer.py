# trainer.py
# Responsibility: train the model, track loss, save checkpoints

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# Fraction of batches that use the slow, token-by-token scheduled-sampling path
# (rather than fast teacher forcing) whenever sampling_rate > 0. Was previously
# hardcoded to 0.05 inside train_one_epoch, contradicting the intended/documented
# 25% — meaning scheduled sampling almost never actually fired.
SCHEDULED_SAMPLING_BATCH_FRACTION = 0.25

# Warn if system-wide RAM usage crosses this percentage. MPS shares memory with
# the whole machine (no dedicated VRAM), so a scheduled-sampling batch spiking
# memory can starve the rest of the system, not just "run out of GPU memory"
# the way a CUDA OOM would.
MEMORY_WARN_THRESHOLD_PCT = 90


def get_memory_status(device):
    """
    Best-effort memory report. Returns a short string suitable for logging,
    plus a bool indicating whether usage looks dangerously high.
    """
    parts = []
    is_high = False

    try:
        if device.type == "mps" and hasattr(torch, "mps"):
            allocated = torch.mps.current_allocated_memory() / 1e9
            driver = torch.mps.driver_allocated_memory() / 1e9
            parts.append(f"torch_mps_allocated={allocated:.2f}GB")
            parts.append(f"torch_mps_driver={driver:.2f}GB")
        elif device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            parts.append(f"torch_cuda_allocated={allocated:.2f}GB")
            parts.append(f"torch_cuda_reserved={reserved:.2f}GB")
    except Exception as e:
        parts.append(f"torch_mem_error={e}")

    if _PSUTIL_AVAILABLE:
        try:
            vm = psutil.virtual_memory()
            parts.append(f"system_ram={vm.percent:.0f}% ({vm.used/1e9:.1f}/{vm.total/1e9:.1f}GB)")
            if vm.percent >= MEMORY_WARN_THRESHOLD_PCT:
                is_high = True
        except Exception as e:
            parts.append(f"system_ram_error={e}")
    else:
        parts.append("system_ram=psutil not installed (pip install psutil)")

    return " | ".join(parts), is_high


def get_device():
    """Pick the best available device: MPS (Mac GPU) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, sampling_rate=0.0, desc="batches"):
    """
    sampling_rate=0.0 → pure teacher forcing (original behaviour)
    sampling_rate>0.0 → scheduled sampling on SCHEDULED_SAMPLING_BATCH_FRACTION of
    batches, teacher forcing on the rest. Only a fraction of batches use the slow
    token-by-token path — enough to close the teacher forcing gap without making
    training dramatically slower.

    desc: label shown on the batch progress bar (e.g. "Chunk 3/10").
    """
    model.train()
    total_loss = 0.0

    # Batch-level progress bar. leave=False so it clears once the chunk finishes,
    # keeping the outer chunk-level bar (in chunked_trainer.py) as the persistent one.
    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=False, position=0,
                mininterval=1.0, dynamic_ncols=False, ncols=100)

    for batch_idx, batch in enumerate(pbar):
        h_peaks, h_mask, c_peaks, c_mask, ir, formula, target_ids = batch

        h_peaks = h_peaks.to(device)
        h_mask = h_mask.to(device)
        c_peaks = c_peaks.to(device)
        c_mask = c_mask.to(device)
        ir = ir.to(device)
        formula = formula.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        # Only run the slow scheduled sampling path on a fraction of batches.
        # FIX: this was hardcoded to 0.05 (5%), which silently contradicted the
        # docstring's stated 25% and meant scheduled sampling barely ever fired,
        # undermining its whole purpose of closing the teacher-forcing gap.
        use_sampling = (sampling_rate > 0.0) and (torch.rand(1).item() < SCHEDULED_SAMPLING_BATCH_FRACTION)

        if not use_sampling:
            logits = model(h_peaks, h_mask, c_peaks, c_mask, ir, target_ids, formula=formula)
            targets = target_ids[:, 1:]
        else:
            memory, padding_mask = model.encode(h_peaks, h_mask, c_peaks, c_mask, ir, formula)

            seq_len = target_ids.size(1)
            decoder_input = target_ids[:, :1]
            all_logits = []

            for t in range(1, seq_len):
                step_logits = model.decode(decoder_input, memory, padding_mask)
                next_logits = step_logits[:, -1:, :]
                all_logits.append(next_logits)

                if torch.rand(1).item() < sampling_rate:
                    next_token = next_logits.squeeze(1).argmax(dim=-1, keepdim=True)
                else:
                    next_token = target_ids[:, t:t+1]

                decoder_input = torch.cat([decoder_input, next_token], dim=1)

            logits = torch.cat(all_logits, dim=1)
            targets = target_ids[:, 1:]

        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Check memory periodically (fixed interval, NOT tied to use_sampling —
        # that was firing hundreds of times per chunk and flooding/corrupting
        # the terminal output with rapid tqdm.write() calls).
        check_memory_now = (batch_idx % 50 == 0)
        if check_memory_now:
            mem_str, mem_is_high = get_memory_status(device)
            postfix = f"loss={total_loss / (batch_idx + 1):.4f} | {mem_str}"
            if mem_is_high:
                tqdm.write(f"  ⚠ HIGH MEMORY USAGE: {mem_str}")
            pbar.set_postfix_str(postfix)
        else:
            pbar.set_postfix_str(f"loss={total_loss / (batch_idx + 1):.4f}")

    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    """Runs the model on validation data WITHOUT updating weights. Just measures loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            h_peaks, h_mask, c_peaks, c_mask, ir, formula, target_ids = batch

            h_peaks = h_peaks.to(device)
            h_mask = h_mask.to(device)
            c_peaks = c_peaks.to(device)
            c_mask = c_mask.to(device)
            ir = ir.to(device)
            formula = formula.to(device)
            target_ids = target_ids.to(device)

            logits = model(h_peaks, h_mask, c_peaks, c_mask, ir, target_ids, formula=formula)
            targets = target_ids[:, 1:]

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, tokenizer,
                num_epochs=10, learning_rate=1e-4, save_path="best_model.pt",
                scheduled_sampling=True, max_sampling_rate=0.3):
    device = get_device()
    print(f"Training on device: {device}")
    model = model.to(device)

    pad_id = tokenizer.token_to_id["<PAD>"]
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # Linearly ramp from 0 → max_sampling_rate over the epochs
        if scheduled_sampling:
            sampling_rate = min(max_sampling_rate, (epoch / num_epochs) * max_sampling_rate)
        else:
            sampling_rate = 0.0

        print(f"Epoch {epoch}/{num_epochs} | sampling_rate: {sampling_rate:.2f}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, sampling_rate)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved (val loss: {val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model

# ── Sanity check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_all_parquets
    from smiles_tokenizer import SmilesTokenizer
    from preprocessor import make_dataloaders
    from model import SpectraToSmilesModel

    FOLDER = "data/multimodal_spectroscopic_dataset"  

    df = load_all_parquets(FOLDER, limit=5)

    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(df['smiles'].tolist())

    MAX_SMILES_LENGTH = 73

    train_loader, val_loader, test_loader = make_dataloaders(
        df, tokenizer, MAX_SMILES_LENGTH, batch_size=32
    )

    model = SpectraToSmilesModel(
        vocab_size=tokenizer.vocab_size,
        max_smiles_length=MAX_SMILES_LENGTH,
    )

    # Just run a few epochs to confirm loss actually goes down
    train_model(model, train_loader, val_loader, tokenizer, num_epochs=3)