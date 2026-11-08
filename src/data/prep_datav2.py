
import torch
from pathlib import Path
from tokenizers import Tokenizer

# ==========================================
# STEP 1: LOCATE PROJECT FILES
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "dataset"

tokenizer_path = BASE_DIR / "nova_bpe_tokenizer.json"

# ==========================================
# STEP 2: LOAD BPE TOKENIZER
# ==========================================

tokenizer = Tokenizer.from_file(str(tokenizer_path))

vocab_size = tokenizer.get_vocab_size()

print("BPE tokenizer loaded!")
print("Vocabulary size:", vocab_size)

# ==========================================
# STEP 3: LOAD ENGLISH DATASET
# ==========================================

train_text = (
    DATASET_DIR / "english_train.txt"
).read_text(encoding="utf-8")

val_text = (
    DATASET_DIR / "english_valid.txt"
).read_text(encoding="utf-8")

test_text = (
    DATASET_DIR / "english_test.txt"
).read_text(encoding="utf-8")

print("\n========== DATASET INFORMATION ==========")

print("Training characters:", len(train_text))
print("Validation characters:", len(val_text))
print("Testing characters:", len(test_text))

# ==========================================
# STEP 4: ENCODE TEXT INTO TOKEN IDs
# ==========================================

train_ids = tokenizer.encode(train_text).ids
val_ids = tokenizer.encode(val_text).ids
test_ids = tokenizer.encode(test_text).ids

print("\n========== TOKEN INFORMATION ==========")

print("Training tokens:", len(train_ids))
print("Validation tokens:", len(val_ids))
print("Testing tokens:", len(test_ids))

# ==========================================
# STEP 5: CREATE PYTORCH TENSORS
# ==========================================

train_data = torch.tensor(train_ids, dtype=torch.long)
val_data = torch.tensor(val_ids, dtype=torch.long)
test_data = torch.tensor(test_ids, dtype=torch.long)

print("\n========== TENSOR INFORMATION ==========")

print("Training tensor:", train_data.shape)
print("Validation tensor:", val_data.shape)
print("Testing tensor:", test_data.shape)

# ==========================================
# STEP 6: CONFIGURE TRAINING
# ==========================================

batch_size = 16
block_size = 64

# ==========================================
# STEP 7: CREATE BATCH GENERATOR
# ==========================================

def get_batch(split):

    if split == "train":
        data = train_data

    elif split == "val":
        data = val_data

    else:
        raise ValueError("Split must be train or val")

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y

# ==========================================
# STEP 8: TEST BATCH GENERATION
# ==========================================

xb, yb = get_batch("train")

print("\n========== BATCH INFORMATION ==========")

print("Input shape:", xb.shape)
print("Target shape:", yb.shape)

print("Input sample:", xb[0][:10].tolist())
print("Target sample:", yb[0][:10].tolist())

# ==========================================
# STEP 9: TEST DECODE
# ==========================================

sample_ids = train_ids[:20]

decoded_text = tokenizer.decode(sample_ids)

print("\n========== DECODING TEST ==========")

print("Token IDs:", sample_ids)
print("Decoded text:", decoded_text)

print("\nNOVA V2 DATA PREPARATION SUCCESSFUL!")

print("\n========== TOKENIZER ROUND-TRIP TEST ==========")

sample = "Nova is learning to build language models."

encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded.ids)

print("Original:", sample)
print("Token IDs:", encoded.ids)
print("Tokens:", encoded.tokens)
print("Decoded:", decoded)

assert decoded == sample

print("Round-trip test successful!")