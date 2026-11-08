
import time
from pathlib import Path

import torch

from model import ContextLanguageModel
from prep_data import vocab_size, get_batch, stoi, itos

# Configuration
device = "cpu"

n_embd = 128
block_size = 64
num_heads = 4
num_layers = 1

max_iters = 1000
eval_interval = 100
eval_iters = 20

learning_rate = 3e-4

# Create model
model = ContextLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=num_layers
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            xb, yb = get_batch(split)

            xb = xb.to(device)
            yb = yb.to(device)

            logits, loss = model(xb, yb)

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


print("Starting Nova training...")
print("Device:", device)

start_time = time.time()

for step in range(max_iters):

    model.train()

    xb, yb = get_batch("train")

    xb = xb.to(device)
    yb = yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    if step % eval_interval == 0 or step == max_iters - 1:

        losses = estimate_loss()

        elapsed = time.time() - start_time

        print(
            f"Step {step:4d} | "
            f"Train: {losses['train']:.4f} | "
            f"Validation: {losses['val']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )


# Save trained model
checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

checkpoint_path = checkpoint_dir / "nova_v1.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "config": {
            "n_embd": n_embd,
            "block_size": block_size,
            "num_heads": num_heads,
            "num_layers": num_layers
        }
    },
    checkpoint_path
)

print("\nNova training completed!")
print("Checkpoint saved to:", checkpoint_path)
