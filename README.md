\# Nova – Small Language Model



\*\*A small GPT-style autoregressive language model built from scratch using PyTorch and trained on WikiText-2.\*\*



Nova is an educational project exploring how language models learn to predict the next token in a sequence using a decoder-only Transformer architecture.



The project follows the learning approach demonstrated in Andrej Karpathy's \*Let's build GPT: from scratch, in code, spelled out\*, using WikiText-2 instead of Tiny Shakespeare.



\---



\## Project Overview



Nova explores the fundamental building blocks of GPT-style language models:



\- Tokenization using Byte Pair Encoding (BPE)

\- Token and positional embeddings

\- Causal self-attention

\- Multi-head attention

\- Feed-forward neural networks

\- Transformer blocks

\- Layer normalization and residual connections

\- Autoregressive text generation



Nova is a small experimental language model, not an instruction-following chatbot.



\## Model Specifications



| Component | Nova v2 |

|---|---|

| Architecture | Decoder-only Transformer |

| Framework | PyTorch |

| Dataset | WikiText-2 |

| Tokenizer | BPE |

| Vocabulary size | 1,000 |

| Embedding dimension | 128 |

| Attention heads | 4 |

| Transformer layers | 2 |

| Feed-forward dimension | 512 |

| Dropout | 0.1 |

| Parameters | 669,416 |

| Training steps | 10,000 |

| Training device | CPU |

| Generation context | 64 tokens |



\## Dataset



Nova uses WikiText-2 with training, validation and test splits.



Processed dataset statistics:



| Split | Tokens |

|---|---:|

| Training | 4,078,866 |

| Validation | 426,549 |

| Testing | 484,800 |



The dataset files are excluded from this repository.



\## Training Results



Recorded Nova v2 results:



| Metric | Loss |

|---|---:|

| Training loss | 3.9015 |

| Validation loss | 3.9358 |

| Estimated test loss | 3.9822 |



The test loss was estimated using 100 randomly sampled batches, not a full evaluation over every test token.



\## Project Structure



```text

Nova/

├── src/

│   ├── data/

│   ├── generation/

│   ├── models/

│   ├── tokenizers/

│   └── training/

├── tests/

├── dataset/              # Local dataset files

├── checkpoints/          # Download model separately

├── results/

├── docs/

├── nova\_bpe\_tokenizer.json

├── requirements.txt

├── .gitignore

└── README.md

```



\## Installation



\### 1. Clone the repository



```bash

git clone https://github.com/Vaishnavi-2223/Nova-Small-Language-Model.git

cd Nova-Small-Language-Model

```



\### 2. Create a Python environment



```bash

conda create -n nova python=3.11

conda activate nova

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\### 4. Download the trained model



Download `nova\_v2\_10000.pth` from the project's GitHub Releases page.



Place it inside:



```text

checkpoints/nova\_v2\_10000.pth

```



The BPE tokenizer file is included in the repository.



\### 5. Run Nova



From the project root:



```bash

python -m src.generation.NovaTerminal

```



\## Interactive Terminal



Nova provides an interactive command-line interface.



Example:



```text

You: The history of science



Nova: \[Generated continuation]

```



Available commands:



| Command | Description |

|---|---|

| `/help` | Show available commands |

| `/temp 0.5` | Change sampling temperature |

| `/exit` | Exit Nova |



\## Limitations



Nova is a small educational model. Its generated text may contain:



\- Grammatically incorrect sentences

\- Unrelated topics and words

\- Repeated phrases

\- Incorrect names and dates

\- Incomplete or incoherent passages



These limitations are expected for a model of this size and training setup. Nova is not intended for factual question answering or production use.



\## Acknowledgements



\- Andrej Karpathy — \*Let's build GPT: from scratch, in code, spelled out\*

\- WikiText-2 dataset

\- PyTorch

\- Hugging Face Tokenizers



\## License



No license has been selected yet. A license will be added before public distribution.

