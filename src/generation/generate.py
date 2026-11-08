import torch
from pathlib import Path

from model import BigramLanguageModel
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "nova_bigram.pth"
)
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

vocab_size = checkpoint["vocab_size"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

print("Checkpoint loaded successfully!")
print("Vocabulary size:", vocab_size)

model = BigramLanguageModel(vocab_size)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Nova model restored successfully!")

start_token = stoi["T"]

context = torch.tensor([[start_token]], dtype=torch.long)

with torch.no_grad():
    generated_tokens = model.generate(
        context,
        max_new_tokens=200
    )

generated_text = "".join(
    itos.get(token, "�")
    for token in generated_tokens[0].tolist()
)

print("\n========== NOVA OUTPUT ==========")
print(generated_text)