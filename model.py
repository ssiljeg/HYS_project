# model.py
# Responsibility: define the encoder-decoder Transformer architecture

import math
import torch
import torch.nn as nn


# ── Positional Encoding ─────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Transformers have no built-in sense of order (unlike CNNs/RNNs).
    This adds a unique numeric 'fingerprint' to each position in a sequence,
    so the model can tell token 1 apart from token 2, etc.
    Uses the standard sin/cos pattern from the original Transformer paper.
    """

    def __init__(self, d_model, max_len=200):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices: cos

        self.register_buffer("pe", pe)  # not a trainable parameter, just fixed math

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:seq_len]  # add positional info to the token embeddings


# ── Peak Encoders ────────────────────────────────────────────────────────────────
class PeakEncoder(nn.Module):
    """
    Projects each individual peak (a small set of numbers) into a
    d_model-sized vector, so it can be treated as one 'token' in the
    Transformer's memory sequence.
    """

    def __init__(self, num_features, d_model):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, peaks):
        # peaks shape: [batch, num_peaks, num_features]
        return self.projection(peaks)  # [batch, num_peaks, d_model]


# ── IR Encoder (CNN, reused concept from before) ─────────────────────────────────
class IrEncoder(nn.Module):
    """
    Compresses the 1800-point IR spectrum into a short sequence of
    d_model-sized tokens (rather than a single summary vector), so the
    decoder can attend to specific regions of the spectrum.
    """

    def __init__(self, d_model):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
            # no final pool — keep the sequence dimension
        )
        self.fc = nn.Linear(64, d_model)

    def forward(self, ir):
        x = ir.unsqueeze(1) if ir.dim() == 2 else ir
        x = self.conv_layers(x)            # [batch, 64, L]
        x = x.transpose(1, 2)              # [batch, L, 64]
        x = self.fc(x)                     # [batch, L, d_model]
        return x                           # multiple IR tokens, not just 1


# ── Full Encoder-Decoder Model ────────────────────────────────────────────────────
FORMULA_FEATURES = 12   # must match preprocessor.FORMULA_FEATURES

# Bump when the model's inputs change meaning, so inference code can tell
# checkpoints apart:
#   1 = original: IR tokens were (accidentally) always masked out, no formula.
#   2 = IR tokens visible (masked only when no IR given), molecular-formula
#       token, trained with augmentation and "unknown" nH/multiplicity codes.
CURRENT_INPUT_VERSION = 2


class SpectraToSmilesModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=6,
                 dim_feedforward=1536, max_smiles_length=80,
                 input_version=CURRENT_INPUT_VERSION):
        # MEDIUM SCALE-UP: was d_model=256, nhead=4, enc=2, dec=3, ffn=512 (~3.6M params).
        # This config targets ~28M params, aimed at testing whether the previous
        # val_loss plateau (0.1456, unmoved across 10 epochs even with LR room to
        # explore) was a capacity ceiling rather than an LR/training-process issue.
        super().__init__()

        self.d_model = d_model
        self.input_version = input_version

        # --- Encoder side: turn peaks/IR into "memory" ---
        self.h_peak_encoder = PeakEncoder(num_features=3, d_model=d_model)
        self.c_peak_encoder = PeakEncoder(num_features=2, d_model=d_model)
        self.ir_encoder = IrEncoder(d_model=d_model)
        if input_version >= 2:
            # Whole formula -> one memory token (zeros + known-flag 0 = unknown)
            self.formula_encoder = PeakEncoder(num_features=FORMULA_FEATURES, d_model=d_model)

        # A few self-attention layers let the memory tokens "talk to each other"
        # before the decoder reads them (e.g. an H peak and a C peak that
        # correspond to the same carbon can reinforce each other)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
        encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False
        )

        # --- Decoder side: generate SMILES tokens ---
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_smiles_length)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_projection = nn.Linear(d_model, vocab_size)

    def encode(self, h_peaks, h_mask, c_peaks, c_mask, ir, formula=None):
        """Builds the 'memory' sequence the decoder will attend to.
        formula: [batch, FORMULA_FEATURES] or None (unknown); ignored by v1 models."""
        h_tokens = self.h_peak_encoder(h_peaks)   # [batch, 25, d_model]
        c_tokens = self.c_peak_encoder(c_peaks)   # [batch, 75, d_model]
        ir_tokens = self.ir_encoder(ir)           # [batch, L, d_model]  <- renamed, L tokens now

        batch_size = ir.size(0)
        ir_len = ir_tokens.size(1)                                   # <- L, not hardcoded 1

        if self.input_version < 2:
            # v1 behaviour, kept so old checkpoints still load and predict the
            # same: mask value 0 means "padding", so IR was never attended to.
            ir_mask = torch.zeros(batch_size, ir_len, device=ir.device)
            memory = torch.cat([h_tokens, c_tokens, ir_tokens], dim=1)
            combined_mask = torch.cat([h_mask, c_mask, ir_mask], dim=1)
        else:
            # IR tokens are real unless the spectrum is all zeros (= not provided)
            ir_present = (ir.abs().amax(dim=-1) > 0).float()               # [batch]
            ir_mask = ir_present.unsqueeze(1).expand(batch_size, ir_len)
            if formula is None:
                formula = torch.zeros(batch_size, FORMULA_FEATURES, device=ir.device)
            formula_token = self.formula_encoder(formula).unsqueeze(1)     # [batch, 1, d_model]
            # The formula token is always attendable (it encodes "unknown" itself),
            # which also guarantees memory is never fully masked.
            formula_mask = torch.ones(batch_size, 1, device=ir.device)
            memory = torch.cat([h_tokens, c_tokens, ir_tokens, formula_token], dim=1)
            combined_mask = torch.cat([h_mask, c_mask, ir_mask, formula_mask], dim=1)

        padding_mask = combined_mask == 0
        memory = self.memory_encoder(memory, src_key_padding_mask=padding_mask)

        return memory, padding_mask

    def decode(self, target_ids, memory, memory_padding_mask):
        """Generates token predictions, given memory and the (shifted) target sequence."""
        seq_len = target_ids.size(1)

        # Embed tokens + add positional info
        x = self.token_embedding(target_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        # Causal mask: position i can only attend to positions <= i
        # (prevents the decoder from "cheating" by looking at future tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(target_ids.device)

        decoded = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        logits = self.output_projection(decoded)  # [batch, seq_len, vocab_size]
        return logits

    def forward(self, h_peaks, h_mask, c_peaks, c_mask, ir, target_ids, formula=None):
        """
        Full forward pass for training (teacher forcing).
        target_ids should be the FULL target sequence; internally we
        feed in all-but-last as decoder input, and predict all-but-first as the answer.
        """
        decoder_input = target_ids[:, :-1]  # everything except the last token

        memory, padding_mask = self.encode(h_peaks, h_mask, c_peaks, c_mask, ir, formula)
        logits = self.decode(decoder_input, memory, padding_mask)

        return logits  # [batch, seq_len - 1, vocab_size]


# ── Sanity check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    VOCAB_SIZE = 46  # from our tokenizer sanity check
    MAX_SMILES_LENGTH = 73

    model = SpectraToSmilesModel(vocab_size=VOCAB_SIZE, max_smiles_length=MAX_SMILES_LENGTH)

    batch_size = 8
    fake_h_peaks = torch.randn(batch_size, 25, 3)
    fake_h_mask = torch.ones(batch_size, 25)
    fake_c_peaks = torch.randn(batch_size, 75, 2)
    fake_c_mask = torch.ones(batch_size, 75)
    fake_ir = torch.randn(batch_size, 1800)
    fake_target_ids = torch.randint(0, VOCAB_SIZE, (batch_size, MAX_SMILES_LENGTH))
    fake_formula = torch.rand(batch_size, FORMULA_FEATURES)

    logits = model(fake_h_peaks, fake_h_mask, fake_c_peaks, fake_c_mask, fake_ir,
                   fake_target_ids, formula=fake_formula)

    print(f"Output logits shape: {logits.shape}")  # expect [8, 72, 46]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")