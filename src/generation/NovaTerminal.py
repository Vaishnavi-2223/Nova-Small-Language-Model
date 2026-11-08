
import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

from src.models.model_v2 import ContextLanguageModel
# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]
DEVICE = "cpu"

CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "nova_v2_10000.pth"
TOKENIZER_PATH = BASE_DIR / "nova_bpe_tokenizer.json"

TEMPERATURE = 0.7
MAX_NEW_TOKENS = 100
BLOCK_SIZE = 64

# ==========================================
# 2. LOAD TOKENIZER
# ==========================================

print("\nLoading Nova tokenizer...")

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
vocab_size = tokenizer.get_vocab_size()

# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

print("Loading Nova v2 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = ContextLanguageModel(vocab_size).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\n" + "=" * 50)
print("             NOVA AI TERMINAL")
print("=" * 50)
print("Model: Nova v2")
print("Dataset: WikiText-2")
print("Tokenizer: BPE")
print("Device: CPU")
print("Temperature:", TEMPERATURE)
print("\nType a prompt and Nova will continue your text.")
print("Commands: /help, /exit")
print("=" * 50)


# ==========================================
# 4. TEXT GENERATION
# ==========================================

def generate_text(prompt, temperature):

    encoded = tokenizer.encode(prompt).ids

    if not encoded:
        print("Please enter a non-empty prompt.")
        return

    context = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        for _ in range(MAX_NEW_TOKENS):

            idx_cond = context[:, -BLOCK_SIZE:]

            logits, loss = model(idx_cond)

            logits = logits[:, -1, :]
            logits = logits / temperature

            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            context = torch.cat(
                (context, next_token),
                dim=1
            )

    return tokenizer.decode(
        context[0].tolist(),
        skip_special_tokens=True
    )


# ==========================================
# 5. INTERACTIVE TERMINAL
# ==========================================

while True:

    try:
        prompt = input("\nYou: ").strip()

    except (KeyboardInterrupt, EOFError):
        print("\n\nNova: Goodbye!")
        break

    if not prompt:
        continue

    if prompt.lower() == "/exit":
        print("\nNova: Goodbye! See you again.")
        break

    if prompt.lower() == "/help":
        print("\nNOVA COMMANDS")
        print("/help       - Show commands")
        print("/exit       - Close Nova")
        print("/temp 0.5   - Change generation temperature")
        print("Any other text is treated as a generation prompt.")
        continue

    if prompt.lower().startswith("/temp "):

        try:
            new_temperature = float(prompt.split()[1])

            if new_temperature <= 0:
                raise ValueError

            TEMPERATURE = new_temperature
            print("Nova temperature changed to:", TEMPERATURE)

        except (ValueError, IndexError):
            print("Please enter a positive number, e.g. /temp 0.7")

        continue

    print("\nNova is thinking...\n")

    generated_text = generate_text(
        prompt,
        TEMPERATURE
    )

    print("Nova:", generated_text)

