import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key = nn.Linear(config.num_embed, config.head_size, bias=False)
        self.query = nn.Linear(config.num_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.num_embed, config.head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B, T, C) @ (B, C, hs) -> (B, T, hs)
        q = self.query(x) # (B, T, C) @ (B, C, hs) -> (B, T, hs)
        v = self.value(x) # (B, T, C) @ (B, C, hs) -> (B, T, hs)

        qk = q @ k.transpose(-2, -1) # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        qk /= self.config.head_size**0.5 # scale

        mask = torch.tril(torch.ones(T, T, device=x.device))
        qk = qk.masked_fill(mask == 0, float('-inf'))
        qk = F.softmax(qk, dim=-1)

        out = qk @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs) |=> 'num_heads' blocks of (B, T, hs) concatenated on last dim will give (B, T, C)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert (config.num_heads * config.head_size == config.num_embed), 'Configuration mismatch: invalid number of heads or head size'
        self.attention_heads = nn.ModuleList([SelfAttention(config) for _ in range(config.num_heads)])
        self.layer_norm1 = nn.LayerNorm(config.num_embed)
        self.layer_norm2 = nn.LayerNorm(config.num_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.num_embed, 4 * config.num_embed),
            nn.GELU(), # modern versions use GELU instead of ReLU
            nn.Linear(4 * config.num_embed, config.num_embed)
        )
        self.proj = nn.Linear(config.num_embed, config.num_embed, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_resid = x
        x = self.layer_norm1(x) # Pre-LN instead of Post-LN from "Attention Is All You Need"
        x = torch.cat([head(x) for head in self.attention_heads], dim=-1) # concat 4*(B, T, hs) on hs -> (B, T, C)
        x = self.proj(x)
        x = self.dropout(x)
        x += x_resid

        x_resid = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x += x_resid
        return x # (B, T, C)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb_table = nn.Embedding(config.vocab_size, config.num_embed)
        self.pos_emb_table = nn.Embedding(config.context_length, config.num_embed)
        self.transformer = nn.Sequential(
            TransformerBlock(config),
            TransformerBlock(config),
            TransformerBlock(config),
        )
        self.linear = nn.Linear(config.num_embed, config.vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y=None):
        B, T = x.shape
        tok = self.tok_emb_table(x) # (B, T, C)
        pos = self.pos_emb_table(torch.arange(T, device=x.device)) # (T, C)
        out = tok + pos # (B, T, C)
        out = self.dropout(out)
        out = self.transformer(out) # (B, T, C)
        logits = self.linear(out) # (B, T, vocab_size)

        if y is None:
            loss = None
        else:
            vocab_sz = logits.shape[-1]
            logits = logits.view(B*T, vocab_sz)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, num_generated_tokens=100):
        # tokens.shape = B, T

        for i in range(num_generated_tokens):
            current_context = tokens[:, -self.config.context_length:] # crop to update the context window (take only 'context_length' last tokens to predict the next one)
            logits, _ = self(current_context) # logits shape = (B, T, vocab_size)
            logits = logits[:, -1, :]  # we take only the last position. Logits become (B, vocab_size), i.e. we're not interested in every position of T
            probs = F.softmax(logits, dim=-1) # softmax of vocab_size dimension, i.e. probability for every position in vocab
            new_token = torch.multinomial(probs, num_samples=1) # sample one token from the probability distribution
            tokens = torch.cat([tokens, new_token], dim=1) # append new token to the overall text

        return tokens