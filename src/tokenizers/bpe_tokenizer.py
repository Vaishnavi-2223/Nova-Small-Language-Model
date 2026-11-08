
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# ==========================================
# STEP 1: LOCATE ENGLISH DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

train_file = DATASET_DIR / "english_train.txt"

print("Loading English training dataset...")

if not train_file.exists():
    raise FileNotFoundError(train_file)

print("Dataset found:", train_file)

# ==========================================
# STEP 2: CREATE BPE TOKENIZER
# ==========================================

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Byte-level preprocessing
tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = ByteLevelDecoder()

# ==========================================
# STEP 3: CONFIGURE BPE TRAINER
# ==========================================

trainer = BpeTrainer(
    vocab_size=1000,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    show_progress=True
)

# ==========================================
# STEP 4: TRAIN TOKENIZER
# ==========================================

print("\nTraining BPE tokenizer...")

tokenizer.train(
    files=[str(train_file)],
    trainer=trainer
)

print("BPE training completed!")
print("Vocabulary size:", tokenizer.get_vocab_size())
print("Number of special tokens:", len(tokenizer.get_added_tokens_decoder()))

# ==========================================
# STEP 5: SAVE TOKENIZER
# ==========================================

tokenizer_path = BASE_DIR / "nova_bpe_tokenizer.json"

tokenizer.save(str(tokenizer_path))

print("\nTokenizer saved:", tokenizer_path)

# ==========================================
# STEP 6: TEST ENCODING
# ==========================================

sample_text = "Hello! Nova is learning English."

encoded = tokenizer.encode(sample_text)

print("\nOriginal text:")
print(sample_text)

print("\nEncoded token IDs:")
print(encoded.ids)

print("\nTokens:")
print(encoded.tokens)

# ==========================================
# STEP 7: TEST DECODING
# ==========================================

decoded = tokenizer.decode(encoded.ids)

print("\nDecoded text:")
print(decoded)

print("\nTokenizer test successful!")
