
import torch

from model import ContextLanguageModel
from prep_data import vocab_size, stoi, itos

model = ContextLanguageModel(
    vocab_size=vocab_size,
    n_embd=128,
    block_size=64,
    num_heads=4,
    num_layers=1
)

# Start with the character "T"
start = torch.tensor(
    [[stoi["T"]]],
    dtype=torch.long
)

generated = model.generate(
    start,
    max_new_tokens=100
)

text = "".join(
    itos.get(i, "?")
    for i in generated[0].tolist()
)

print("Generated text:")
print(text)
