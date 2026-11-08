import torch
from model import FeedForward

"""FeedForward network processes each token's 
representation independently after attention has mixed information between tokens."""

x = torch.randn(16, 64, 128)

ff = FeedForward(128)

output = ff(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

params = sum(p.numel() for p in ff.parameters())

print("FeedForward parameters:", params)