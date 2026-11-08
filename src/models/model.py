import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.head_size = head_size

    def forward(self, x):

        B, T, C = x.shape

        # Create Query, Key and Value
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Calculate attention scores
        scores = q @ k.transpose(-2, -1)

        # Scale the scores
        scores = scores * (self.head_size ** -0.5)

        # Prevent looking at future positions
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores into probabilities
        weights = torch.softmax(scores, dim=-1)

        # Collect information from Value vectors
        output = weights @ v

        return output

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(
            num_heads * head_size,
            n_embd
        )

    def forward(self, x):

        outputs = [
            head(x) for head in self.heads
        ]

        combined = torch.cat(outputs, dim=-1)

        return self.proj(combined)

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        head_size = n_embd // num_heads

        self.sa = MultiHeadAttention(
            num_heads,
            n_embd,
            head_size,
            block_size
        )

        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # Attention + residual connection
        x = x + self.sa(self.ln1(x))

        # FeedForward + residual connection
        x = x + self.ffwd(self.ln2(x))

        return x




class ContextLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd=128,
        block_size=64,
        num_heads=4,
        num_layers=1
    ):
        super().__init__()

        self.block_size = block_size

        # 1. Token embeddings
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )

        # 2. Position embeddings
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )

        # 3. Transformer blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd,
                    num_heads,
                    block_size
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Final normalization
        self.ln_f = nn.LayerNorm(n_embd)

        # 5. Predict next character
        self.lm_head = nn.Linear(n_embd, vocab_size)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            # Temperature controls randomness
            logits = logits / temperature

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > self.block_size:
            raise ValueError("Sequence exceeds block size!")

        # Token embeddings
        token_embeddings = self.token_embedding_table(idx)

        # Position embeddings
        positions = torch.arange(
            T, device=idx.device
        )

        position_embeddings = (
            self.position_embedding_table(positions)
        )

        # Combine embeddings
        x = token_embeddings + position_embeddings

        # Transformer processing
        x = self.blocks(x)

        # Final normalization
        x = self.ln_f(x)

        # Vocabulary predictions
        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss