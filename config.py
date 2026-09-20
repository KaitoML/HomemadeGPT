from dataclasses import dataclass
import torch

@dataclass
class ConfigGPT:
    context_length: int = 64
    num_embed: int = 16
    head_size: int = 4
    num_heads : int = 4
    batch_size: int = 32
    vocab_size: int = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
