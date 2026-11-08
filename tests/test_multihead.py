import torch

from model import MultiHeadAttention

batch_size = 16
block_size = 64
n_embd = 128
num_heads = 4
head_size = 32

x = torch.randn(batch_size, block_size, n_embd)

attention = MultiHeadAttention(
    num_heads=num_heads,
    n_embd=n_embd,
    head_size=head_size,
    block_size=block_size
)

output = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

parameters = sum(
    p.numel() for p in attention.parameters()
)

print("Multi-head parameters:", parameters)

print("\nMulti-head attention successful!")