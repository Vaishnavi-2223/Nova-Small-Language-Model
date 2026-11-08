
from datasets import load_dataset
import os

# Load the already downloaded dataset (cached locally)
dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

# Export all three splits
splits = {
    "train": "wiki.train.raw",
    "validation": "wiki.valid.raw",
    "test": "wiki.test.raw"
}

for split, filename in splits.items():

    text = "\n".join(dataset[split]["text"])

    path = os.path.join("dataset", filename)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"Saved: {path}")
    print(f"Characters: {len(text)}")

print("\nDataset exported successfully!")
