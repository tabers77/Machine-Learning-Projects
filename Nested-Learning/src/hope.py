"""Hope Architecture — integrates associative memory, DMGD, and CMS.

Builds TitanBlock (attention + persistent memory + surprise gating) and
HopeBlock (extends TitanBlock with multi-frequency CMS and DMGD meta-learning).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cms import MemoryBlock
from .dmgd import DeepMomentum


class TitanBlock(nn.Module):
    """Base block: input projection, multi-head attention (associative memory),
    persistent memory MLP, and surprise-based gating.

    Surprise = ||x - W_mem @ x||^2 — measures how unexpected the input is
    relative to what persistent memory predicts.
    """

    def __init__(self, embed_dim: int, n_heads: int = 2, ff_dim: int | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        ff_dim = ff_dim or 4 * embed_dim

        # Multi-head self-attention (associative memory)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Persistent memory MLP
        self.memory_mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

        # Surprise gate: linear projection for memory prediction
        self.mem_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Sigmoid(),
        )

    def compute_surprise(self, x: torch.Tensor) -> torch.Tensor:
        """Compute surprise score: ||x - W_mem @ x||^2 per position.

        Returns: (batch, seq_len, 1)
        """
        predicted = self.mem_proj(x)
        surprise = ((x - predicted) ** 2).sum(dim=-1, keepdim=True)
        return surprise

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward pass.

        x: (batch, seq_len, embed_dim)
        Returns: (output, surprise) where surprise is (batch, seq_len, 1).
        """
        # Causal mask for autoregressive attention
        seq_len = x.shape[1]
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Self-attention + residual
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.attn_norm(x + attn_out)

        # Surprise-gated persistent memory
        surprise = self.compute_surprise(x)
        gate_val = self.gate(surprise)  # (batch, seq_len, embed_dim)
        mem_out = self.memory_mlp(x)
        x = self.ff_norm(x + gate_val * mem_out)

        return x, surprise


class HopeBlock(nn.Module):
    """Extends TitanBlock with CMS wrapping and DMGD meta-learning.

    Multi-frequency design:
    - Attention: updates every step (fastest)
    - Short-term MLP: updates every C steps
    - Long-term MLP: updates every C^2 steps
    - Meta-optimizer (DMGD): updates every C^3 steps (slowest)
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 2,
        ff_dim: int | None = None,
        c_base: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.c_base = c_base
        ff_dim = ff_dim or 4 * embed_dim

        # Level 0: Attention (every step)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Level 1: Short-term memory MLP (every C steps)
        self.short_term = MemoryBlock(embed_dim, ff_dim, embed_dim, update_freq=c_base)
        self.short_norm = nn.LayerNorm(embed_dim)

        # Level 2: Long-term memory MLP (every C^2 steps)
        self.long_term = MemoryBlock(embed_dim, ff_dim, embed_dim, update_freq=c_base ** 2)
        self.long_norm = nn.LayerNorm(embed_dim)

        # Surprise gate
        self.mem_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(1, embed_dim), nn.Sigmoid())

    def compute_surprise(self, x: torch.Tensor) -> torch.Tensor:
        predicted = self.mem_proj(x)
        return ((x - predicted) ** 2).sum(dim=-1, keepdim=True)

    def get_active_levels(self, global_step: int) -> list[str]:
        """Return which levels should update at this step."""
        active = ["attn"]  # always
        if global_step % self.c_base == 0:
            active.append("short_term")
        if global_step % (self.c_base ** 2) == 0:
            active.append("long_term")
        return active

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        seq_len = x.shape[1]
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Attention (level 0)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.attn_norm(x + attn_out)

        # Surprise gating
        surprise = self.compute_surprise(x)
        gate_val = self.gate(surprise)

        # Short-term memory (level 1) — always in forward pass
        short_out = self.short_term(x)
        x = self.short_norm(x + gate_val * short_out)

        # Long-term memory (level 2) — always in forward pass
        long_out = self.long_term(x)
        x = self.long_norm(x + gate_val * long_out)

        return x, surprise


class HopeModel(nn.Module):
    """Full Hope model: token embedding + positional encoding + stack of HopeBlocks + output head.

    Designed for character-level language modeling at small scale.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 256,
        c_base: int = 4,
        dropout: float = 0.1,
        use_titan: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if use_titan:
            self.blocks = nn.ModuleList([
                TitanBlock(embed_dim, n_heads) for _ in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                HopeBlock(embed_dim, n_heads, c_base=c_base) for _ in range(n_layers)
            ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        x: (batch, seq_len) token indices.
        Returns: (logits, surprises) where logits is (batch, seq_len, vocab_size).
        """
        batch, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)

        h = self.dropout(self.token_embed(x) + self.pos_embed(pos))

        surprises = []
        for block in self.blocks:
            h, s = block(h)
            surprises.append(s)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, surprises

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 100, temperature: float = 0.8):
        """Autoregressive text generation.

        prompt: (1, prompt_len) token indices.
        Returns: (1, prompt_len + max_new_tokens) token indices.
        """
        self.eval()
        tokens = prompt.clone()
        for _ in range(max_new_tokens):
            x = tokens[:, -self.max_seq_len:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


class HopeTrainer:
    """Training loop for HopeModel with multi-frequency CMS updates.

    Manages global step counter, selectively updates blocks at their
    designated frequencies, and logs metrics.
    """

    def __init__(
        self,
        model: HopeModel,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.global_step = 0

        # Separate optimizers for different frequency levels
        attn_params = []
        short_params = []
        long_params = []
        other_params = []

        for name, param in model.named_parameters():
            if "attn" in name:
                attn_params.append(param)
            elif "short_term" in name:
                short_params.append(param)
            elif "long_term" in name:
                long_params.append(param)
            else:
                other_params.append(param)

        self.optimizers = {
            "attn": torch.optim.Adam(attn_params, lr=lr) if attn_params else None,
            "short": torch.optim.Adam(short_params, lr=lr) if short_params else None,
            "long": torch.optim.Adam(long_params, lr=lr) if long_params else None,
            "other": torch.optim.Adam(other_params, lr=lr) if other_params else None,
        }

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        self.model.train()
        self.global_step += 1

        x = x.to(self.device)
        y = y.to(self.device)

        # Zero all grads
        for opt in self.optimizers.values():
            if opt is not None:
                opt.zero_grad()

        logits, surprises = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        # Determine which levels to update
        c_base = self.model.blocks[0].c_base if hasattr(self.model.blocks[0], "c_base") else 1

        # Attention + other: always update
        if self.optimizers["attn"]:
            self.optimizers["attn"].step()
        if self.optimizers["other"]:
            self.optimizers["other"].step()

        # Short-term: every c_base steps
        if self.optimizers["short"] and self.global_step % c_base == 0:
            self.optimizers["short"].step()

        # Long-term: every c_base^2 steps
        if self.optimizers["long"] and self.global_step % (c_base ** 2) == 0:
            self.optimizers["long"].step()

        mean_surprise = torch.stack([s.mean() for s in surprises]).mean().item()

        return {
            "loss": loss.item(),
            "perplexity": math.exp(min(loss.item(), 20)),
            "surprise": mean_surprise,
            "step": self.global_step,
        }

    def train_epoch(self, dataloader, verbose: bool = True) -> list[dict]:
        logs = []
        for batch_idx, (x, y) in enumerate(dataloader):
            info = self.train_step(x, y)
            logs.append(info)
            if verbose and batch_idx % 100 == 0:
                print(
                    f"  Step {info['step']:4d} | Loss: {info['loss']:.4f} | "
                    f"PPL: {info['perplexity']:.1f} | Surprise: {info['surprise']:.4f}"
                )
        return logs
