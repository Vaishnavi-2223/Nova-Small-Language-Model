
import torch
from pathlib import Path

from model import ContextLanguageModel

# ==========================================
# 1. CONFIGURATION
# ==========================================

device = "cpu"

prompt = "The "
max_new_tokens = 300

temperatures = [0.3, 0.5, 0.7, 1.0]

# ==========================================
# 2. LOAD TRAINED NOVA CHECKPOINT
# ==========================================

checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "nova_v1.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

# Recover saved vocabulary
vocab_size = checkpoint["vocab_size"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

# Recover model configuration
config = checkpoint["config"]

# ==========================================
# 3. RECREATE NOVA MODEL
# ==========================================

model = ContextLanguageModel(
    vocab_size=vocab_size,
    n_embd=config["n_embd"],
    block_size=config["block_size"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"]
).to(device)

# Load trained weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)

# Switch to inference mode
model.eval()

print("=" * 50)
print("NOVA v1 LOADED SUCCESSFULLY!")
print("=" * 50)

print("Device:", device)
print("Vocabulary size:", vocab_size)
print("Prompt:", repr(prompt))

# ==========================================
# 4. ENCODE PROMPT
# ==========================================

unk_id = vocab_size - 1

encoded = [
    stoi.get(char, unk_id)
    for char in prompt
]

context = torch.tensor(
    [encoded],
    dtype=torch.long,
    device=device
)

# ==========================================
# 5. GENERATE TEXT AT DIFFERENT TEMPERATURES
# ==========================================

for temperature in temperatures:

    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    # Convert tokens into readable characters
    output = "".join(
        itos.get(token, "�")
        for token in generated[0].tolist()
    )

    print("\n" + "=" * 50)
    print(f"TEMPERATURE: {temperature}")
    print("=" * 50)

    print(output)

# ==========================================
# 6. FINISHED
# ==========================================

print("\n" + "=" * 50)
print("NOVA GENERATION COMPLETED!")
print("=" * 50)
