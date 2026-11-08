import torch
from model import ContextLanguageModel
from prep_data import vocab_size, get_batch

#Model Config
model=ContextLanguageModel(vocab_size=vocab_size,
                           n_embd=128,
                           block_size=64,
                           num_heads=4,
                           num_layers=1
)

#Get a real batch from Wiki-text2
x,y=get_batch("train")

#forward pass
logits, loss = model(x,y)

#count parameters
params=sum(
    p.numel() for p in model.parameters()
)

print("Input shape:", x.shape)
print("Logits shape:", logits.shape)
print("Loss:", loss.item())
print("Total parameters:", params)

print("NOVA MODEL TEST SUCCESSFUL!")