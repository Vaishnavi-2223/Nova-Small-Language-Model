import torch

from model import Head
from prep_data import get_batch

batch_size = 16
block_size = 64
embedding_dim = 128
head_size = 32

xb, yb = get_batch("train")

# Create dummy embeddings for this test
x = torch.randn(batch_size, block_size, embedding_dim)

# Create one attention head
head = Head(
    n_embd=embedding_dim,
    head_size=head_size,
    block_size=block_size
)

output = head(x)

print("Input shape:", x.shape)
print("Attention output shape:", output.shape)

print("Total attention parameters:",
      sum(p.numel() for p in head.parameters()))

print("\nAttention test successful!")