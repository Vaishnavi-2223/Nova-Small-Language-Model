from pathlib import Path
import re

# Locate the dataset folder
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

# Original files and filtered output files
splits = {
    "wiki.train.raw": "english_train.txt",
    "wiki.valid.raw": "english_valid.txt",
    "wiki.test.raw": "english_test.txt"
}

def filter_english(text):

    # Keep English letters, numbers, whitespace and common punctuation
    text = re.sub(
        r"[^A-Za-z0-9\s.,!?;:'\"()\-\[\]]",
        " ",
        text
    )

    # Clean repeated spaces without removing paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)

    return text


for original_file, output_file in splits.items():

    input_path = DATASET_DIR / original_file
    output_path = DATASET_DIR / output_file

    text = input_path.read_text(encoding="utf-8")

    english_text = filter_english(text)

    output_path.write_text(
        english_text,
        encoding="utf-8"
    )

    print(f"\nCreated: {output_file}")
    print(f"Original characters: {len(text):,}")
    print(f"Filtered characters: {len(english_text):,}")

print("\nEnglish dataset preparation complete!")