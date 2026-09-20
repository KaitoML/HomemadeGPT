# GPT from scratch

An educational implementation of a decoder-only GPT (transformer) in PyTorch, built from scratch — including self-attention, transformer blocks, and a custom byte-level BPE tokenizer (no `tiktoken` or other pre-built libraries).

The goal of this project is to understand how transformers and BPE tokenization actually work under the hood, not to produce production-quality text generation. The model is trained on a small text ("Alice's Adventures in Wonderland", ~150 KB) and is intentionally compact, so it does not generate coherent text — see the "Limitations" section below for why.

## Project structure

```
config.py          # dataclass with model hyperparameters (ConfigGPT)
gpt.py              # architecture: SelfAttention, TransformerBlock, GPT
tokenizer.py        # byte-level BPE tokenizer, trained from scratch
tests.ipynb         # data prep, training, generation
alice.txt           # training text (Lewis Carroll, public domain)
gpt_params.pth      # trained model weights
requirements.txt    # dependencies
```

## What's implemented

- **Self-attention from scratch**: key/query/value projections, causal masking, scaled dot-product attention.
- **Multi-head attention** via several independent `SelfAttention` heads, concatenated afterward.
- **Transformer block**: pre-norm (`LayerNorm`), residual connections, feed-forward network.
- **Byte-level BPE tokenizer** — trained on raw text by iteratively finding and merging the most frequent byte pairs (similar in spirit to Andrej Karpathy's `minbpe`). Supports `train`, `encode`, `decode`.
- **Autoregressive generation** with a sliding context window and sampling (`torch.multinomial`) from the softmax distribution.

## Running it

```bash
pip install -r requirements.txt
```

Then open and run `tests.ipynb` — it walks through: loading the text → training the tokenizer → preparing batches → training the model → generating text.

## Limitations

This is intentionally a learning project, not a production one:

- **Small dataset** (~150 KB of text) — for comparison, real language models are trained on terabytes of text.
- **Compact model** (few layers, small embedding dimension), trained in seconds to minutes on a CPU/consumer GPU rather than hours/days on a cluster.
- The project originally included a second, larger model configuration ("prod"). Experiments showed that at this data size and network depth, scaling the model up (more layers, larger embeddings, more aggressive BPE merging) didn't help — and sometimes hurt — because there wasn't enough data to meaningfully use the added capacity. For that reason, the project now focuses on a single compact baseline configuration.
- For the same reason, the number of BPE merges (`num_merges`) is intentionally kept low — with a small dataset and a simple model, tokenization close to character-level trains noticeably better than a large vocabulary full of rarely-seen tokens.

The generated text reproduces local language patterns (letter combinations, punctuation, occasional words) but not coherent meaning — this is an expected consequence of the project's scale, not an implementation bug.

## Possible next steps

- Train on a larger and more diverse text corpus.
- Tune `num_merges` for the current model size.
- Scale up `context_length`/depth alongside data size, rather than independently.