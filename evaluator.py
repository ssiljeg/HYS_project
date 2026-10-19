# evaluator.py
# Responsibility: generate SMILES for unseen spectra, check chemistry validity and accuracy

import math
import torch
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def generate_smiles_batch(model, h_peaks, h_mask, c_peaks, c_mask, ir,
                          tokenizer, device, max_length, formula=None):
    """
    Autoregressive generation: starts with <START>, predicts one token at a time,
    feeds each prediction back in as input for the next step.
    This is what real inference looks like (no teacher forcing / no cheating).

    Returns: list of generated SMILES strings, one per molecule in the batch.
    """
    model.eval()
    batch_size = h_peaks.size(0)

    start_id = tokenizer.token_to_id["<START>"]
    end_id = tokenizer.token_to_id["<END>"]

    with torch.no_grad():
        # Encode the spectra once -- this doesn't change as we generate tokens
        memory, padding_mask = model.encode(h_peaks, h_mask, c_peaks, c_mask, ir, formula)

        # Start every sequence in the batch with just the <START> token
        generated = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)

        # Track which sequences have already produced <END>, so we stop updating them
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            logits = model.decode(generated, memory, padding_mask)
            # logits shape: [batch, current_seq_len, vocab_size]

            # We only care about the prediction for the NEXT token,
            # which is the last position in the sequence so far
            next_token_logits = logits[:, -1, :]              # [batch, vocab_size]
            next_token = next_token_logits.argmax(dim=-1)     # [batch] -- greedy: pick highest probability

            # For sequences that already finished, just keep appending <END>
            # so we don't corrupt them with further predictions
            next_token = torch.where(finished, torch.full_like(next_token, end_id), next_token)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            # Update which sequences are now finished
            finished = finished | (next_token == end_id)

            if finished.all():
                break  # every molecule in the batch is done, no need to keep generating

    # Convert token ID sequences back into SMILES strings
    smiles_list = []
    for i in range(batch_size):
        ids = generated[i].tolist()
        smiles = tokenizer.decode(ids)
        smiles_list.append(smiles)

    return smiles_list


def is_valid_smiles(smiles: str) -> bool:
    """Checks if RDKit can parse this SMILES into a real molecule."""
    if smiles == "":
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize(smiles: str):
    """
    Converts a SMILES string into its canonical form.
    Two different SMILES strings can represent the SAME molecule
    (e.g. "CCO" and "OCC" are both ethanol) -- canonicalizing lets us
    compare fairly instead of just doing exact string matching.
    Stereochemistry is removed: 1D NMR/IR can't distinguish stereoisomers, so
    a prediction that differs only in @/@@ or /\ counts as correct.
    Returns None if invalid.
    """
    from preprocessor import strip_stereo
    return strip_stereo(smiles)


def tanimoto_similarity(smiles_a: str, smiles_b: str):
    """
    Measures how chemically similar two molecules are, from 0 (nothing alike)
    to 1 (identical). Useful for 'near miss' predictions that aren't an exact
    match but are structurally close to the right answer.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    if mol_a is None or mol_b is None:
        return 0.0

    fp_generator = GetMorganGenerator(radius=2, fpSize=2048)
    fp_a = fp_generator.GetFingerprint(mol_a)
    fp_b = fp_generator.GetFingerprint(mol_b)

    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def evaluate_model(model, dataloader, tokenizer, device, max_length, max_batches=None):
    """
    Runs generation across a dataset and reports:
      - validity rate: % of generated SMILES that are chemically valid at all
      - exact match rate: % that exactly match the true molecule (after canonicalizing)
      - average Tanimoto similarity: how close the predictions are, even when wrong
    """
    model = model.to(device)
    model.eval()

    total = 0
    valid_count = 0
    exact_match_count = 0
    similarity_sum = 0.0

    examples = []  # store a few examples to print/inspect later

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break  # let us test on a small slice instead of the whole test set, for speed

        h_peaks, h_mask, c_peaks, c_mask, ir, formula, target_ids = batch

        h_peaks = h_peaks.to(device)
        h_mask = h_mask.to(device)
        c_peaks = c_peaks.to(device)
        c_mask = c_mask.to(device)
        ir = ir.to(device)
        formula = formula.to(device)

        # Get the TRUE SMILES strings for this batch (decode the ground truth tokens)
        true_smiles_list = [tokenizer.decode(ids.tolist()) for ids in target_ids]

        # Generate predicted SMILES
        predicted_smiles_list = generate_smiles_batch(
            model, h_peaks, h_mask, c_peaks, c_mask, ir, tokenizer, device, max_length,
            formula=formula,
        )

        for true_smiles, predicted_smiles in zip(true_smiles_list, predicted_smiles_list):
            total += 1

            if is_valid_smiles(predicted_smiles):
                valid_count += 1

                true_canonical = canonicalize(true_smiles)
                pred_canonical = canonicalize(predicted_smiles)

                if true_canonical is not None and pred_canonical is not None:
                    if true_canonical == pred_canonical:
                        exact_match_count += 1

                    similarity = tanimoto_similarity(true_smiles, predicted_smiles)
                    similarity_sum += similarity

            if len(examples) < 10:
                examples.append((true_smiles, predicted_smiles))

    validity_rate = valid_count / total if total > 0 else 0
    exact_match_rate = exact_match_count / total if total > 0 else 0
    avg_similarity = similarity_sum / total if total > 0 else 0

    print(f"\n=== Evaluation Results ({total} molecules) ===")
    print(f"  Validity rate     : {validity_rate:.1%}")
    print(f"  Exact match rate  : {exact_match_rate:.1%}")
    print(f"  Avg Tanimoto sim  : {avg_similarity:.3f}")

    print(f"\n=== Example predictions ===")
    for true_smiles, predicted_smiles in examples:
        match = "✓" if canonicalize(true_smiles) == canonicalize(predicted_smiles) else "✗"
        print(f"  {match} True: {true_smiles}")
        print(f"    Pred: {predicted_smiles}")

    return {
        "validity_rate": validity_rate,
        "exact_match_rate": exact_match_rate,
        "avg_similarity": avg_similarity,
    }

# Add these two functions to evaluator.py

def generate_smiles_beam_search(model, h_peaks, h_mask, c_peaks, c_mask, ir,
                                tokenizer, device, max_length, beam_width=5,
                                return_all=False, formula=None):
    """
    Generates ONE SMILES string for ONE molecule using beam search.
    Keeps the top `beam_width` candidate sequences alive at every step,
    instead of only ever keeping the single best guess (greedy).
    Inputs are unbatched (no leading batch dim). With return_all=True,
    returns every final beam as a ranked list of (smiles, confidence).
    formula: optional [FORMULA_FEATURES] tensor (input_version 2 models).
    """
    model.eval()
    start_id = tokenizer.token_to_id["<START>"]
    end_id = tokenizer.token_to_id["<END>"]

    with torch.no_grad():
        memory, padding_mask = model.encode(
            h_peaks.unsqueeze(0), h_mask.unsqueeze(0),
            c_peaks.unsqueeze(0), c_mask.unsqueeze(0), ir.unsqueeze(0),
            formula.unsqueeze(0) if formula is not None else None,
        )

        # Each beam: (list of token ids, cumulative log-prob, is_finished)
        beams = [([start_id], 0.0, False)]

        for _ in range(max_length - 1):
            active = [b for b in beams if not b[2]]
            finished = [b for b in beams if b[2]]

            if not active:
                break  # every beam already produced <END>

            # Batch all active beams together for ONE decode() call
            seqs = torch.tensor([b[0] for b in active], dtype=torch.long, device=device)
            batch_memory = memory.repeat(len(active), 1, 1)
            batch_mask = padding_mask.repeat(len(active), 1)

            logits = model.decode(seqs, batch_memory, batch_mask)
            next_logits = logits[:, -1, :]
            log_probs = torch.log_softmax(next_logits, dim=-1)

            new_candidates = []
            for i, (tokens, score, _) in enumerate(active):
                topk_log_probs, topk_ids = log_probs[i].topk(beam_width)
                for k in range(beam_width):
                    next_id = topk_ids[k].item()
                    new_score = score + topk_log_probs[k].item()
                    new_tokens = tokens + [next_id]
                    is_finished = (next_id == end_id)
                    new_candidates.append((new_tokens, new_score, is_finished))

            # Rank by AVERAGE log-prob per token, not raw total --
            # otherwise the search unfairly favors shorter sequences
            # (since every additional token adds a negative log-prob)
            all_candidates = finished + new_candidates
            all_candidates.sort(key=lambda b: b[1] / len(b[0]), reverse=True)

            beams = all_candidates[:beam_width]

            if all(b[2] for b in beams):
                break

    beams.sort(key=lambda b: b[1] / len(b[0]), reverse=True)
    if return_all:
        # Every beam as (smiles, confidence), confidence = exp(mean token log-prob) in (0, 1]
        return [(tokenizer.decode(tokens), math.exp(score / len(tokens)))
                for tokens, score, _ in beams]
    best_tokens = beams[0][0]
    return tokenizer.decode(best_tokens)


def evaluate_model_beam_search(model, dataloader, tokenizer, device, max_length,
                               beam_width=5, max_batches=None):
    """Same metrics as evaluate_model, but using beam search instead of greedy decoding."""
    model = model.to(device)
    model.eval()

    total = 0
    valid_count = 0
    exact_match_count = 0
    similarity_sum = 0.0
    examples = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        h_peaks, h_mask, c_peaks, c_mask, ir, formula, target_ids = batch
        h_peaks, h_mask = h_peaks.to(device), h_mask.to(device)
        c_peaks, c_mask = c_peaks.to(device), c_mask.to(device)
        ir, formula = ir.to(device), formula.to(device)

        for i in range(h_peaks.size(0)):
            true_smiles = tokenizer.decode(target_ids[i].tolist())

            predicted_smiles = generate_smiles_beam_search(
                model, h_peaks[i], h_mask[i], c_peaks[i], c_mask[i], ir[i],
                tokenizer, device, max_length, beam_width=beam_width,
                formula=formula[i],
            )

            total += 1
            if is_valid_smiles(predicted_smiles):
                valid_count += 1
                true_canonical = canonicalize(true_smiles)
                pred_canonical = canonicalize(predicted_smiles)
                if true_canonical is not None and pred_canonical is not None:
                    if true_canonical == pred_canonical:
                        exact_match_count += 1
                    similarity_sum += tanimoto_similarity(true_smiles, predicted_smiles)

            if len(examples) < 10:
                examples.append((true_smiles, predicted_smiles))

        print(f"  Processed batch {batch_idx + 1} ({total} molecules so far)...")

    validity_rate = valid_count / total if total > 0 else 0
    exact_match_rate = exact_match_count / total if total > 0 else 0
    avg_similarity = similarity_sum / total if total > 0 else 0

    print(f"\n=== Beam Search Results ({total} molecules, beam_width={beam_width}) ===")
    print(f"  Validity rate     : {validity_rate:.1%}")
    print(f"  Exact match rate  : {exact_match_rate:.1%}")
    print(f"  Avg Tanimoto sim  : {avg_similarity:.3f}")

    print(f"\n=== Example predictions ===")
    for true_smiles, predicted_smiles in examples:
        match = "✓" if canonicalize(true_smiles) == canonicalize(predicted_smiles) else "✗"
        print(f"  {match} True: {true_smiles}")
        print(f"    Pred: {predicted_smiles}")

    return {"validity_rate": validity_rate, "exact_match_rate": exact_match_rate, "avg_similarity": avg_similarity}


# ── Sanity check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import random
    from torch.utils.data import DataLoader

    from data_loader import get_parquet_files, load_specific_files
    from smiles_tokenizer import SmilesTokenizer
    from preprocessor import SpectraToSmilesDataset
    from model import SpectraToSmilesModel
    from trainer import get_device

    FOLDER = "data/multimodal_spectroscopic_dataset" 
    NUM_VAL_FILES = 5  # must match what chunked_trainer.py used

    # Load the SAME config the model was trained with
    with open("dataset_config.json") as f:
        config = json.load(f)

    tokenizer = SmilesTokenizer()
    tokenizer.load_vocab(config["token_to_id"])

    # Rebuild the EXACT same held-out validation files (same seed = same shuffle result)
    all_files = get_parquet_files(FOLDER)
    random.seed(42)
    random.shuffle(all_files)
    val_files = all_files[:NUM_VAL_FILES]

    print(f"Loading {len(val_files)} held-out validation files (never trained on)...")
    val_df = load_specific_files(val_files)

    val_ds = SpectraToSmilesDataset(
        val_df, tokenizer,
        max_smiles_length=config["max_smiles_length"],
        max_h_peaks=config["max_h_peaks"],
        max_c_peaks=config["max_c_peaks"],
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = get_device()
    checkpoint = torch.load("best_model.pt", map_location=device)

    # Build the model with the SAME shapes and input version it was trained with
    model = SpectraToSmilesModel(
        vocab_size=config["vocab_size"],
        max_smiles_length=config["max_smiles_length"],
        input_version=checkpoint.get("input_version", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    evaluate_model_beam_search(model, val_loader, tokenizer, device,
                               max_length=config["max_smiles_length"],
                               beam_width=5, max_batches=20)
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('outer_epoch')}, "
      f"chunk {checkpoint.get('chunk_idx')}, best_val_loss={checkpoint.get('best_val_loss')}")