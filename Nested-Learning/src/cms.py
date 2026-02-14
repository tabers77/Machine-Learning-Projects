"""Continuum Memory System (CMS).

Core claim: Memory isn't binary (short/long-term) — it's a continuum of update
frequencies. Different memory blocks update at geometrically increasing intervals,
inspired by brain oscillations (gamma → theta → delta).
"""

import torch
import torch.nn as nn


class MemoryBlock(nn.Module):
    """Single MLP block with a configurable update frequency.

    Only updates (backward + optimizer step) when global_step % update_freq == 0.
    Always participates in the forward pass.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, update_freq: int = 1):
        super().__init__()
        self.update_freq = update_freq
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def should_update(self, global_step: int) -> bool:
        return global_step % self.update_freq == 0


class ContinuumMemorySystem(nn.Module):
    """Chain of MemoryBlocks with geometrically increasing update frequencies.

    Level 0 updates every step (fast / "gamma").
    Level l updates every c_base^l steps (slow / "delta").

    All blocks always participate in the forward pass. Only backward/update
    is selective.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_levels: int = 4,
        c_base: int = 4,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.c_base = c_base

        self.blocks = nn.ModuleList()
        for level in range(n_levels):
            freq = c_base ** level
            in_d = input_dim if level == 0 else hidden_dim
            out_d = output_dim if level == n_levels - 1 else hidden_dim
            self.blocks.append(MemoryBlock(in_d, hidden_dim, out_d, update_freq=freq))

        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all blocks (all participate regardless of update schedule)."""
        h = x
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)

    def get_active_blocks(self, global_step: int) -> list[int]:
        """Return indices of blocks that should update at this step."""
        return [i for i, b in enumerate(self.blocks) if b.should_update(global_step)]

    def get_update_frequencies(self) -> list[int]:
        """Return the update frequency of each block."""
        return [b.update_freq for b in self.blocks]


class CMSTrainer:
    """Training loop that selectively updates only active CMS blocks per step.

    Uses separate optimizer groups per frequency level so we can selectively
    call .step() only on active ones.
    """

    def __init__(self, model: ContinuumMemorySystem, lr: float = 1e-3):
        self.model = model
        self.global_step = 0

        # Separate optimizer per block + one for the output projection
        self.block_optimizers = []
        for block in model.blocks:
            self.block_optimizers.append(
                torch.optim.Adam(block.parameters(), lr=lr)
            )
        self.proj_optimizer = torch.optim.Adam(model.output_proj.parameters(), lr=lr)

    def train_step(self, x: torch.Tensor, y: torch.Tensor, loss_fn) -> dict:
        """One training step with selective block updates.

        Returns dict with 'loss', 'active_blocks', 'global_step'.
        """
        self.global_step += 1
        active = self.model.get_active_blocks(self.global_step)

        # Zero all grads
        for opt in self.block_optimizers:
            opt.zero_grad()
        self.proj_optimizer.zero_grad()

        # Forward
        pred = self.model(x)
        loss = loss_fn(pred, y)

        # Backward (computes gradients for all parameters)
        loss.backward()

        # Selective update: only step optimizers for active blocks
        for i in active:
            self.block_optimizers[i].step()

        # Always update output projection
        self.proj_optimizer.step()

        # Zero gradients for inactive blocks (prevent stale grad accumulation)
        for i in range(self.model.n_levels):
            if i not in active:
                self.block_optimizers[i].zero_grad()

        return {
            "loss": loss.item(),
            "active_blocks": active,
            "global_step": self.global_step,
        }

    def train_epoch(self, dataloader, loss_fn, verbose: bool = False) -> list[dict]:
        """Train for one full epoch."""
        self.model.train()
        logs = []
        for batch_idx, (x, y) in enumerate(dataloader):
            info = self.train_step(x, y, loss_fn)
            logs.append(info)
            if verbose and batch_idx % 50 == 0:
                print(
                    f"  Step {info['global_step']:4d} | Loss: {info['loss']:.4f} | "
                    f"Active: {info['active_blocks']}"
                )
        return logs
