
import torch
from model import TransformerBlock

batch_size = 16
block_size = 64
n_embd = 128
num_heads = 4

x = torch.randn(batch_size, block_size, n_embd)

block = TransformerBlock(
    n_embd=n_embd,
    num_heads=num_heads,
    block_size=block_size
)

output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

params = sum(p.numel() for p in block.parameters())

print("Transformer Block parameters:", params)

print("Transformer Block test successful!")
