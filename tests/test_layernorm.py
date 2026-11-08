
import torch
import torch.nn as nn

# 16 batches, 64 tokens, 128 features
x = torch.randn(16, 64, 128)

layer_norm = nn.LayerNorm(128)

output = layer_norm(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

# Check the normalized values
print("Mean:", output.mean().item())
print("Standard deviation:", output.std().item())

print("LayerNorm test successful!")
