# Homemade GPT from scratch

A character/byte-level GPT language model built from scratch in PyTorch, implementing multi-head self-attention, transformer blocks, and a custom byte-pair encoding (BPE) tokenizer. The implementation involves a few deviations from the original transformer architecture introduced in ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762) paper, including decoder-only transformer, Pre-LN normalization, GELU activation in a feedforward block

## Features

- **Custom BPE tokenizer** (`tokenizer.py`) — a from-scratch byte-pair encoding tokenizer (`Tokenizer` class): iteratively merges the most frequent byte pairs into a learned vocabulary (`train`), encodes new text by greedily applying learned merges in order (`encode`), and decodes token sequences back to UTF-8 text (`decode`). No external tokenizer libraries used.
- **Self-attention from scratch** (`gpt.py`) — a single-head `SelfAttention` module implementing scaled dot-product attention with causal masking (`torch.tril`).
- **Multi-head attention & transformer blocks** — `TransformerBlock` combines multiple attention heads (concatenated and projected), a feed-forward MLP, residual connections, and pre-norm `LayerNorm`.
- **Full GPT model** — token + positional embeddings, a stack of transformer blocks, and a final linear head projecting to vocabulary logits, with built-in `cross_entropy` loss computation and autoregressive `generate()` for sampling new text.
- **Configurable architecture** (`config.py`) — a `ConfigGPT` dataclass controlling context length, embedding dimension, number/size of attention heads, and batch size.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.x and a Jupyter environment (or IDE with notebook support) to run `tests.ipynb`. GPU is used automatically if available (`torch.cuda.is_available()`), falling back to CPU otherwise.

## Usage

```python
from config import ConfigGPT
from tokenizer import Tokenizer
from gpt import GPT

tokenizer = Tokenizer()
tokens = tokenizer.train(text, num_merges=50)

config = ConfigGPT(vocab_size=tokenizer.vocab_size)
model = GPT(config)

logits, loss = model(x_batch, y_batch)  # training forward pass
generated = model.generate(context, num_generated_tokens=100)  # autoregressive sampling
```

See `tests.ipynb` for the full pipeline: data preparation, tokenizer training, batching, the training loop with train/val loss tracking, and text generation before/after training.

## Training

The model is trained on *Alice in Wonderland* and a corpus of *Twitter conversations* with a lightweight BPE vocabulary (mild compression — a small dataset and model don't benefit from aggressive token merging). Training tracks both train and validation loss over a fixed number of steps, plotted at the end for a quick over/underfitting check.

## Motivation

Built as a learning exercise in order to dive deeper into how self-attention mechanism and transformers work. Initially, the project had used character-level tokenization, which was then replaced with `tiktoken` tools. They, however, ended up working poorly for our small dataset and serving as a bottleneck during an optimization. This is why custom `Tokenizer` was built.  

## Notes

The tokenizer's `encode` method was built with AI assistance; the BPE training/merge logic, the attention mechanism, transformer architecture, and training loop were implemented independently.

## License

This project is licensed under the MIT License.