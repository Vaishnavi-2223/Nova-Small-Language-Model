# Nova — Small Language Model

<p align="center">
  <strong>A GPT-style language model built from scratch using PyTorch.</strong>
  <br />
  Trained on WikiText-2 · Byte Pair Encoding · Decoder-only Transformer
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Dataset-WikiText--2-green" alt="WikiText-2" />
  <img src="https://img.shields.io/badge/Model-Transformer-purple" alt="Transformer" />
  <img src="https://img.shields.io/badge/Status-Experimental-orange" alt="Experimental" />
</p>

---

## Overview

**Nova** is a small autoregressive language model developed to explore the fundamental concepts behind Generative Pre-trained Transformers (GPT).

Built using PyTorch, Nova learns to predict the next token in a sequence through a decoder-only Transformer architecture.

The project follows the educational approach demonstrated in Andrej Karpathy's *Let's build GPT: from scratch, in code, spelled out*, using WikiText-2 instead of Tiny Shakespeare.

The goal is to understand how tokenization, embeddings, attention mechanisms, and neural network training work together to produce text.

> Nova is an experimental language model for learning and exploration, not an instruction-following chatbot.

## Architecture

Nova v2 uses a decoder-only Transformer with the following components:

1. **Token Embeddings** — Converts token IDs into vector representations.
2. **Positional Embeddings** — Provides information about token positions.
3. **Causal Self-Attention** — Allows each token to attend only to previous tokens and itself.
4. **Multi-Head Attention** — Learns different attention relationships in parallel.
5. **Feed-Forward Network** — Processes the attention representations.
6. **Residual Connections and Layer Normalization** — Support stable Transformer block computation.
7. **Language Model Head** — Predicts the next token in the sequence.

### Model Configuration

| Parameter | Value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feed-forward dimension | 512 |
| Dropout | 0.1 |
| Vocabulary size | 1,000 |
| Context used for generation | 64 tokens |
| Total parameters | 669,416 |

## Dataset & Tokenization

Nova uses **WikiText-2**, a dataset containing Wikipedia articles for language modeling.

The preprocessing pipeline produces filtered English text files, which are used to train a Byte Pair Encoding (BPE) tokenizer.

| Dataset split | Tokens |
|---|---:|
| Training | 4,078,866 |
| Validation | 426,549 |
| Testing | 484,800 |

**Tokenizer details:**
- Byte-level BPE
- Vocabulary size: 1,000
- Special tokens: `[UNK]`, `[PAD]`, `[BOS]`, `[EOS]`

The original and processed dataset files are kept locally and are not included in this repository.

## Training

Nova v2 was trained on a CPU for 10,000 optimization steps, continuing from an earlier 3,000-step checkpoint.

### Recorded Results

| Metric | Loss |
|---|---:|
| Training loss | 3.9015 |
| Validation loss | 3.9358 |
| Estimated test loss | 3.9822 |

The test loss was estimated using 100 randomly sampled batches rather than evaluating every token in the test split.

## Text Generation

Nova generates text autoregressively by repeatedly predicting and sampling the next token.

The interactive terminal supports:
- User-provided text prompts
- Temperature-controlled sampling
- Continuous text generation
- Basic terminal commands

### Running Nova

```bash
python -m src.generation.NovaTerminal
```

Example interaction:

```text
==================================================
             NOVA AI TERMINAL
==================================================
Model: Nova v2
Dataset: WikiText-2
Tokenizer: BPE
Device: CPU
Temperature: 0.7
==================================================

You: The history of science

Nova: [Generated continuation]
```

Available commands:

| Command | Function |
|---|---|
| `/help` | Display available commands |
| `/temp 0.5` | Change sampling temperature |
| `/exit` | Exit the terminal |

## Project Structure

```text
Nova/
│
├── src/
│   ├── data/
│   │   ├── data_Loader.py
│   │   ├── export_dataset.py
│   │   ├── filter_dataset.py
│   │   ├── prep_data.py
│   │   └── prep_datav2.py
│   │
│   ├── generation/
│   │   ├── NovaTerminal.py
│   │   ├── generate.py
│   │   ├── generate_Nova.py
│   │   └── generate_v2.py
│   │
│   ├── models/
│   │   ├── model.py
│   │   └── model_v2.py
│   │
│   ├── tokenizers/
│   │   ├── bpe_tokenizer.py
│   │   └── Tokenizer.py
│   │
│   └── training/
│       ├── train_nova.py
│       └── train_novav2.py
│
├── tests/
├── results/
├── docs/
├── nova_bpe_tokenizer.json
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### Requirements

- Python 3.11
- PyTorch
- Hugging Face Tokenizers

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Vaishnavi-2223/Nova-Small-Language-Model.git
cd Nova-Small-Language-Model
```

**2. Create a virtual environment**

```bash
conda create -n nova python=3.11
conda activate nova
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the trained checkpoint**

Download `nova_v2_10000.pth` from the repository's Releases section and place it inside:

```text
checkpoints/nova_v2_10000.pth
```

The tokenizer configuration is included in the repository.

**5. Launch Nova**

```bash
python -m src.generation.NovaTerminal
```

## Limitations

Nova is intentionally small and experimental. Its generated text can contain:

- Grammatically incorrect sentences
- Inconsistent or unrelated topics
- Repeated phrases
- Incorrect names and dates
- Incomplete or incoherent passages

These limitations reflect the model's scale, training setup, and available data. Nova is not designed for factual question answering or production use.

## Acknowledgements

- Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*
- WikiText-2 dataset
- PyTorch
- Hugging Face Tokenizers

## License

No license has been selected yet. Licensing will be addressed before reuse or redistribution.

---

<p align="center">
  <strong>Nova — Exploring language models, one token at a time.</strong>
</p>