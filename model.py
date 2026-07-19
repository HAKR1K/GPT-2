import math
import torch
import torch.nn as nn
from config import ModelConfig
from embeddings import TokenEmbedding, LearnedPositionEmbedding
from transformer import TransformerBlock
from layers import LayerNorm

class MiniGPT(nn.Module):
    """The complete Decoder-Only MiniGPT model."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = TokenEmbedding(config.vocab_size, config.embedding_dimension),
            wpe = LearnedPositionEmbedding(config.block_size, config.embedding_dimension),
            h = nn.ModuleList([
                TransformerBlock(config.embedding_dimension, config.number_of_heads, config.dropout)
                for _ in range(config.number_of_layers)
            ]),
            ln_f = LayerNorm(config.embedding_dimension)
        ))
        
        self.lm_head = nn.Linear(config.embedding_dimension, config.vocab_size, bias=False)
        
        # Weight tying: share word embedding weights with final classifier projection
        self.lm_head.weight = self.transformer.wte.wte.weight
        
        # Apply standard GPT-2 initialization
        self.apply(self._init_weights)
        
        # Scaled initialization for residual projection layers
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.number_of_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = idx.size()
        assert t <= self.config.block_size, f"Input sequence of length {t} exceeds max block size {self.config.block_size}"
        
        # Combine token embeddings and absolute learned position embeddings
        x = self.transformer.wte(idx) + self.transformer.wpe(idx)
        
        # Pass through N transformer block layers
        for block in self.transformer.h:
            x = block(x)
            
        # Final layer normalization
        x = self.transformer.ln_f(x)
        
        # Vocabulary projection logits
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                label_smoothing=0.05
            )
            
        return logits, loss

    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        method: str = "greedy", 
        temperature: float = 1.0, 
        top_k: int = None, 
        top_p: float = None
    ) -> torch.Tensor:
        """Autoregressive text generation."""
        from sampling import sample_next_token
        
        for _ in range(max_new_tokens):
            # Crop prompt sequence if it is longer than the context window size
            idx_cond = idx[:, -self.config.block_size:]
            
            # Forward pass to get logits for the last token position
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]
            
            # Sample the next token index
            next_token = sample_next_token(next_token_logits, method, temperature, top_k, top_p)
            
            # Append next token to context
            idx = torch.cat((idx, next_token.unsqueeze(-1)), dim=1)
            
        return idx
