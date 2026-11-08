import torch
from model_v2 import ContextLanguageModel

# Configuration
vocab_size = 1000
batch_size = 16
block_size = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the model
model = ContextLanguageModel(vocab_size).to(device)

# Generate dummy input
x = torch.randint(
    0,
    vocab_size,
    (batch_size, block_size),
    device=device
)

# Test forward pass
logits, loss = model(x)

# Display results
print("========== NOVA V2 MODEL TEST ==========")

print("Device:", device)

print("Input shape:", x.shape)

print("Output shape:", logits.shape)

print("Loss:", loss)

print("Parameters:", sum(
    p.numel() for p in model.parameters()
))

print("\nNOVA V2 MODEL TEST SUCCESSFUL!")