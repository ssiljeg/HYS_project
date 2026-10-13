# Responsibility: convert SMILES strings into sequence of integer tokens and back 

import re 

# re = regular expression
# allows you to search, split, replace and parse complex text patterns 

# Special tokens every sequence needs
PAD_TOKEN = "<PAD>"                        # padding to make sequences equal length in a batch
START_TOKEN = "<START>"                    # tells the decoder "begin generating"
END_TOKEN = "<END>"                        # tells the decoder "stop generating"
UNK_TOKEN = "<UNK>"                        # unknown token, for characters we've never seen

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

# This regex pattern splits SMILES into chemically sensible chunks
# instead of single characters. For example "Cl" stays together as one token,
# "c1" stays together, multi-char elements like "Br" stay together, etc.
SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|@@|@|=|#|\(|\)|\.|\+|-|/|\\\\|%\d{2}|\d|[A-Za-z])"
)

def tokenize_smiles(smiles: str) -> list:
    """
    Splits a SMILES string into a list of chemically meaningful tokens.
    Example: "CCO" -> ["C", "C", "O"]
    Example: "Clc1ccccc1" -> ["Cl", "c", "1", "c", "c", "c", "c", "c", "1"]
    """
    tokens = SMILES_REGEX.findall(smiles)
    return tokens


class SmilesTokenizer:
    """
    Builds a vocabulary from a list of SMILES strings, and converts
    between SMILES strings and integer token sequences.
    """

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self, smiles_list: list):
        """
        Scans all SMILES strings, collects every unique token,
        and assigns each one an integer ID.
        """
        unique_tokens = set()

        for smiles in smiles_list:
            tokens = tokenize_smiles(smiles)
            unique_tokens.update(tokens)

        # Special tokens always get the first IDs (0, 1, 2, 3)
        all_tokens = SPECIAL_TOKENS + sorted(unique_tokens)

        self.token_to_id = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

        print(f"Vocabulary built: {len(self.token_to_id)} unique tokens")

    def encode(self, smiles: str, max_length: int) -> list:
        """
        Converts a SMILES string into a fixed-length list of integer IDs.
        Adds <START> at the beginning and <END> at the end.
        Pads with <PAD> if shorter than max_length, truncates if longer.
        """
        tokens = tokenize_smiles(smiles)
        tokens = [START_TOKEN] + tokens + [END_TOKEN]

        # Convert tokens to IDs (use UNK_TOKEN's id if we've never seen this token)
        unk_id = self.token_to_id[UNK_TOKEN]
        ids = [self.token_to_id.get(tok, unk_id) for tok in tokens]

        # Pad or truncate to exactly max_length
        if len(ids) < max_length:
            pad_id = self.token_to_id[PAD_TOKEN]
            ids = ids + [pad_id] * (max_length - len(ids))
        else:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: list) -> str:
        """
        Converts a list of integer IDs back into a SMILES string.
        Stops at <END>, skips <PAD> and <START>.
        """
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, UNK_TOKEN)

            if token == END_TOKEN:
                break  # stop reading once we hit the end marker
            if token in (START_TOKEN, PAD_TOKEN):
                continue  # skip these, they're not part of the actual SMILES

            tokens.append(token)

        return "".join(tokens)

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def load_vocab(self, token_to_id: dict):
        """Load a pre-built vocabulary (e.g. from a saved scan) instead of building a new one."""
        self.token_to_id = token_to_id
        self.id_to_token = {idx: tok for tok, idx in token_to_id.items()}


# ── Sanity check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_all_parquets

    FOLDER = "data/multimodal_spectroscopic_dataset"  # <-- change this

    df = load_all_parquets(FOLDER, limit=5)

    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(df['smiles'].tolist())

    # Test round-trip: SMILES -> tokens -> ids -> back to SMILES
    test_smiles = df['smiles'].iloc[0]
    print(f"\nOriginal SMILES : {test_smiles}")

    tokens = tokenize_smiles(test_smiles)
    print(f"Tokens          : {tokens}")
    print(f"Token count     : {len(tokens)}")

    # Check max length needed across the whole dataset
    max_token_len = max(len(tokenize_smiles(s)) for s in df['smiles'])
    print(f"\nLongest SMILES (in tokens): {max_token_len}")
    suggested_max_length = max_token_len + 2  # +2 for START and END tokens
    print(f"Suggested max_length (with START/END): {suggested_max_length}")

    encoded = tokenizer.encode(test_smiles, max_length=suggested_max_length)
    print(f"\nEncoded IDs     : {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded SMILES  : {decoded}")
    print(f"Round-trip match: {decoded == test_smiles}")