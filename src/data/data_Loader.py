
import os

# Dataset directory
DATASET_DIR = "dataset"

# File paths
train_path = os.path.join(DATASET_DIR, "wiki.train.raw")
valid_path = os.path.join(DATASET_DIR, "wiki.valid.raw")
test_path = os.path.join(DATASET_DIR, "wiki.test.raw")

# Read files
with open(train_path, "r", encoding="utf-8") as file:
    train_text = file.read()

with open(valid_path, "r", encoding="utf-8") as file:
    valid_text = file.read()

with open(test_path, "r", encoding="utf-8") as file:
    test_text = file.read()

# Display information
print("========== LOCAL DATASET ==========")

print("Training characters:", len(train_text))
print("Validation characters:", len(valid_text))
print("Testing characters:", len(test_text))

print("\n========== SAMPLE TEXT ==========")

print(train_text[:500])

print("\nLocal dataset loaded successfully!")
