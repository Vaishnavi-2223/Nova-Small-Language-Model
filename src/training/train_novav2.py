
import torch
from pathlib import Path

from src.models.model_v2 import ContextLanguageModel
from src.data.prep_datav2 import get_batch, vocab_size

# ==============================
# TRAINING CONFIGURATION
# ==============================

device = "cpu"

total_steps = 10000
eval_interval = 500
eval_iters = 50
learning_rate = 3e-4

BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Original 3000-step checkpoint
old_checkpoint_path = CHECKPOINT_DIR / "nova_v2.pth"

# New 10000-step checkpoint
new_checkpoint_path = CHECKPOINT_DIR / "nova_v2_10000.pth"

torch.manual_seed(42)

# ==============================
# INITIALIZE MODEL
# ==============================

model = ContextLanguageModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01
)

# ==============================
# RESUME PREVIOUS TRAINING
# ==============================

if not old_checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {old_checkpoint_path}"
    )

checkpoint = torch.load(
    old_checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_step = checkpoint["step"]

print("\n========== RESUMING NOVA V2 ==========")
print("Device:", device)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Previous training steps:", start_step)
print("Target total steps:", total_steps)
print("Additional steps:", total_steps - start_step)

if start_step >= total_steps:
    raise ValueError("Checkpoint has already reached the target steps.")

# ==============================
# EVALUATE MODEL
# ==============================

@torch.no_grad()
def estimate_loss():
    results = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split)

            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)
            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()
    return results


# ==============================
# INITIAL EVALUATION
# ==============================

initial_losses = estimate_loss()

print("\nStarting losses:")
print(f"Train: {initial_losses['train']:.4f}")
print(f"Validation: {initial_losses['val']:.4f}")

# ==============================
# TRAINING LOOP
# ==============================

model.train()

for step in range(start_step, total_steps):

    if step % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {step}: "
            f"Train loss = {losses['train']:.4f}, "
            f"Validation loss = {losses['val']:.4f}",
            flush=True
        )

    x, y = get_batch("train")

    x = x.to(device)
    y = y.to(device)

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# ==============================
# FINAL EVALUATION
# ==============================

final_losses = estimate_loss()

print("\n========== FINAL RESULTS ==========")
print(f"Total steps: {total_steps}")
print(f"Train loss: {final_losses['train']:.4f}")
print(f"Validation loss: {final_losses['val']:.4f}")

# ==============================
# SAVE NEW CHECKPOINT
# ==============================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_size": vocab_size,
        "step": total_steps,
        "train_loss": final_losses["train"],
        "val_loss": final_losses["val"],
    },
    new_checkpoint_path
)

print("\nNOVA V2 TRAINING COMPLETE!")
print("New checkpoint saved:", new_checkpoint_path)
