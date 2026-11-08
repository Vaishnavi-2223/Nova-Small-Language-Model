import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

from model_v2 import ContextLanguageModel


def generate(model, idx, max_new_tokens, tokenizer,
             block_size, temperature=1.0):

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    model.eval()

    for _ in range(max_new_tokens):

        # Keep only the latest context
        idx_cond = idx[:, -block_size:]

        # Get model predictions
        with torch.no_grad():
            logits, _ = model(idx_cond)

        # Select predictions for the last token
        logits = logits[:, -1, :]

        # Apply temperature
        logits = logits / temperature

        # Convert scores into probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample the next token
        next_token = torch.multinomial(probs, num_samples=1)

        # Append the new token
        idx = torch.cat((idx, next_token), dim=1)

    return idx

# ==============================
# CONFIGURATION
# ==============================

BASE_DIR = Path(__file__).resolve().parent

device = "cpu"
vocab_size = 1000
block_size = 128

# ==============================
# LOAD TOKENIZER
# ==============================

tokenizer_path = BASE_DIR / "nova_bpe_tokenizer.json"

tokenizer = Tokenizer.from_file(str(tokenizer_path))

# ==============================
# LOAD TRAINED MODEL
# ==============================

model = ContextLanguageModel(vocab_size).to(device)

checkpoint = torch.load(
    BASE_DIR / "checkpoints" / "nova_v2_10000.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Nova v2 loaded successfully!")
print("Training steps:", checkpoint["step"])
print("Validation loss:", checkpoint["val_loss"])

# ==============================
# GENERATION EXPERIMENT
# ==============================

prompts = [
    "The development of computers",
    "Artificial intelligence is",
    "The history of science"
]

torch.manual_seed(42)

for prompt in prompts:

    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device
    )

    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=100,
        tokenizer=tokenizer,
        block_size=block_size,
        temperature=0.5
    )

    generated_ids = output_ids[0].tolist()

    generated_text = tokenizer.decode(generated_ids)

    print("\n" + "=" * 55)
    print("PROMPT:", prompt)
    print("=" * 55)
    print(generated_text)
output_ids = generate(
    model,
    input_ids,
    max_new_tokens=100,
    tokenizer=tokenizer,
    block_size=block_size,
    temperature=0.5
)

generated_text = tokenizer.decode(output_ids[0].tolist())

print("\n========== NOVA V2 OUTPUT ==========")
print(generated_text)