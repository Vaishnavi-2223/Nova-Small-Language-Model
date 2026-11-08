
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# NOVA V2 CONFIGURATION
# ==========================================

batch_size = 16
block_size = 128

max_iters = 3000
learning_rate = 3e-4

eval_interval = 300
eval_iters = 100

n_embd = 128
num_heads = 4
num_layers = 2

dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# CAUSAL SELF-ATTENTION HEAD
# ==========================================

class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        key = self.key(x)
        query = self.query(x)

        # Calculate attention scores
        weights = query @ key.transpose(-2, -1)

        weights = weights * (key.shape[-1] ** -0.5)

        # Prevent looking at future tokens
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)

        weights = self.dropout(weights)

        # Weighted aggregation of values
        value = self.value(x)

        output = weights @ value

        return output


# ==========================================
# MULTI-HEAD ATTENTION
# ==========================================

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):

        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            num_heads * head_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Run all attention heads
        output = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        # Combine their outputs
        output = self.projection(output)

        output = self.dropout(output)

        return output


# ==========================================
# FEEDFORWARD NETWORK
# ==========================================

class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(n_embd, 4 * n_embd),

            nn.ReLU(),

            nn.Linear(4 * n_embd, n_embd),

            nn.Dropout(dropout)

        )

    def forward(self, x):

        return self.network(x)


# ==========================================
# TRANSFORMER BLOCK
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        head_size = n_embd // num_heads

        self.sa = MultiHeadAttention(
            num_heads,
            head_size
        )

        self.ffwd = FeedForward()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # Attention + residual connection
        x = x + self.sa(self.ln1(x))

        # FeedForward + residual connection
        x = x + self.ffwd(self.ln2(x))

        return x


# ==========================================
# COMPLETE NOVA V2 LANGUAGE MODEL
# ==========================================

class ContextLanguageModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        # Token embeddings
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        # Position embeddings
        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.ln_f = nn.LayerNorm(n_embd)

        # Predict next token
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    # ======================================
    # FORWARD PASS
    # ======================================

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # Token embeddings
        token_emb = self.token_embedding_table(idx)

        # Position embeddings
        pos = torch.arange(T, device=idx.device)

        pos_emb = self.position_embedding_table(pos)

        # Combine embeddings
        x = token_emb + pos_emb

        x = self.dropout(x)

        # Transformer blocks
        x = self.blocks(x)

        # Final normalization
        x = self.ln_f(x)

        # Vocabulary prediction
        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)

            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    # ======================================
    # AUTOREGRESSIVE TEXT GENERATION
    # ======================================

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0
    ):

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than zero."
            )

        self.eval()

        for _ in range(max_new_tokens):

            # Keep the most recent 64 tokens.
            # This matches Nova v2's training context.
            idx_cond = idx[:, -64:]

            # Forward pass
            logits, loss = self(idx_cond)

            # Get predictions for the final position
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Convert logits into probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Append the new token
            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx
