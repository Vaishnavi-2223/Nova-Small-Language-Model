import torch
from pathlib import Path
#torch handles the sensor and path helps us to locate the dataset files.


# STEP 1: LOAD WIKITEXT-2 DATASET


# Find the dataset folder relative to this Python file
DATASET_DIR = Path(__file__).resolve().parent / "dataset"

# Load all three text files
train_text = (DATASET_DIR / "wiki.train.raw").read_text(encoding="utf-8")

val_text = (DATASET_DIR / "wiki.valid.raw").read_text(encoding="utf-8")

test_text = (DATASET_DIR / "wiki.test.raw").read_text(encoding="utf-8")

print("========== DATASET INFORMATION ==========")

print("Training characters:", len(train_text))
print("Validation characters:", len(val_text))
print("Testing characters:", len(test_text))


# =============================================
# STEP 2: CREATE VOCABULARY
# =============================================

chars = sorted(set(train_text))

num_chars = len(chars)
unk_id = num_chars
vocab_size = num_chars + 1

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print("Unique characters:", num_chars)
print("Model vocabulary size:", vocab_size)

print("\n========== VOCABULARY DEBUG ==========")
print("Number of unique characters:", len(chars))
print("UNK ID:", unk_id)
print("vocab_size variable:", vocab_size)
print("Length of stoi:", len(stoi))
print("Length of itos:", len(itos))


# =============================================
# STEP 3: CREATE CHARACTER MAPPINGS
# =============================================

# Character to integer
stoi = {ch: i for i, ch in enumerate(chars)}

# Integer to character
itos = {i: ch for i, ch in enumerate(chars)}

# Unknown character ID
unk_id = num_chars


# =============================================
# STEP 4: ENCODING AND DECODING
# =============================================

def encode(text):
    # Convert characters into integer IDs
    return [stoi.get(ch, unk_id) for ch in text]


def decode(tokens):
    # Convert integer IDs back into characters
    return "".join(
        itos[i] if i in itos else "�"
        for i in tokens
    )


# =============================================
# STEP 5: CONVERT TEXT INTO PYTORCH TENSORS
# =============================================

train_data = torch.tensor(
    encode(train_text), dtype=torch.long
)

val_data = torch.tensor(
    encode(val_text), dtype=torch.long
)

test_data = torch.tensor(
    encode(test_text), dtype=torch.long
)

print("\n========== TENSOR INFORMATION ==========")

print("Training tensor:", train_data.shape)
print("Validation tensor:", val_data.shape)
print("Testing tensor:", test_data.shape)

print("Tensor data type:", train_data.dtype)


# =============================================
# STEP 6: UNDERSTAND INPUT AND TARGET
# =============================================

block_size = 8

# Take a small sequence from training data
x = train_data[:block_size]

# Target is shifted one position ahead
y = train_data[1:block_size + 1]

print("\n========== INPUT AND TARGET ==========")

print("Input (x):", x)
print("Target (y):", y)

print("\nInput decoded:", decode(x.tolist()))
print("Target decoded:", decode(y.tolist()))


# Show next-character predictions
print("\n========== NEXT TOKEN PREDICTIONS ==========")

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]

    print(
        f"Context: {decode(context.tolist())!r}"
        f" -> Next: {decode([target.item()])!r}"
    )


# =============================================
# STEP 7: CREATE TRAINING BATCHES
# =============================================

batch_size = 16
block_size = 64


def get_batch(split):

    # Select the appropriate dataset
    if split == "train":
        data = train_data

    elif split == "val":
        data = val_data

    else:
        raise ValueError("Split must be 'train' or 'val'")

    # Generate random starting positions
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    # Create input sequences
    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    # Create target sequences (shifted by one)
    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y


# =============================================
# STEP 8: TEST OUR BATCH GENERATOR
# =============================================

xb, yb = get_batch("train")

print("\n========== BATCH INFORMATION ==========")

print("Input batch shape:", xb.shape)
print("Target batch shape:", yb.shape)

print("\nInput batch:")
print(xb)

print("\nTarget batch:")
print(yb)

print("\n========== PREPARATION COMPLETE ==========")

# ==========================================
# STEP 9: DATASET CHARACTER ANALYSIS
# ==========================================

from collections import Counter
import unicodedata

char_counts = Counter(train_text)

print("\n========== CHARACTER FREQUENCY ==========")

for char, count in char_counts.most_common(30):
    print(repr(char), count)


# ==========================================
# STEP 10: SCRIPT DISTRIBUTION
# ==========================================

scripts = Counter()

for char in set(train_text):

    if char.isspace():
        continue

    name = unicodedata.name(char, "UNKNOWN")

    if "LATIN" in name:
        scripts["Latin"] += 1
    elif "CJK" in name or "IDEOGRAPH" in name:
        scripts["CJK"] += 1
    elif "HIRAGANA" in name:
        scripts["Hiragana"] += 1
    elif "KATAKANA" in name:
        scripts["Katakana"] += 1
    elif "HANGUL" in name:
        scripts["Hangul"] += 1
    elif "ARABIC" in name:
        scripts["Arabic"] += 1
    else:
        scripts["Other"] += 1

print("\n========== SCRIPT DISTRIBUTION ==========")

for script, count in scripts.items():
    print(script, count)