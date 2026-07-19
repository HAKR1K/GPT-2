import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """Maps token IDs to vectors."""
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wte(x)


class LearnedPositionEmbedding(nn.Module):
    """Learned absolute positional embeddings (GPT-2 style)."""
    def __init__(self, block_size: int, embed_dim: int):
        super().__init__()
        self.wpe = nn.Embedding(block_size, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        return self.wpe(positions)


class SinusoidalPositionEmbedding(nn.Module):
    """Static sinusoidal positional embeddings."""
    def __init__(self, block_size: int, embed_dim: int):
        super().__init__()
        pe = torch.zeros(block_size, embed_dim)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(-1)]
