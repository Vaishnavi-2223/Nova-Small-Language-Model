import torch

from model import ContextLanguageModel
from prep_data import vocab_size, get_batch



# Create Nova
model = ContextLanguageModel(vocab_size)

# Get a training batch
xb, yb = get_batch("train")

print("Input shape:", xb.shape)
print("Target shape:", yb.shape)

# Forward pass
logits, loss = model(xb, yb)

print("\n========== MODEL OUTPUT ==========")
print("Logits shape:", logits.shape)
print("Loss:", loss.item())

# Count parameters
total_params = sum(
    p.numel() for p in model.parameters()
)

print("Total parameters:", total_params)

print("\n========== PARAMETER BREAKDOWN ==========")

for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.numel())