"""Visualization and training helpers for Nested Learning experiments."""

import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss_curves(
    losses: dict[str, list[float]],
    title: str = "Training Loss",
    xlabel: str = "Step",
    ylabel: str = "Loss",
    log_scale: bool = False,
    ax=None,
):
    """Plot one or more loss curves on a single axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for label, vals in losses.items():
        ax.plot(vals, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_attention_weights(
    attn_weights: torch.Tensor,
    title: str = "Attention Weights",
    ax=None,
):
    """Plot attention weight matrix as a heatmap.

    attn_weights: 2-D tensor (query_len, key_len).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    w = attn_weights.detach().cpu().numpy()
    im = ax.imshow(w, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax)
    return ax


def plot_update_timeline(
    n_steps: int,
    n_blocks: int,
    c_base: int,
    title: str = "CMS Update Timeline",
    ax=None,
):
    """Heatmap showing which CMS blocks update at each step.

    x-axis: training step, y-axis: block index.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    grid = np.zeros((n_blocks, n_steps))
    for block_idx in range(n_blocks):
        freq = c_base ** block_idx
        for step in range(n_steps):
            if step % freq == 0:
                grid[block_idx, step] = 1.0
    ax.imshow(grid, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Block (slow → fast)")
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels([f"C^{i} = {c_base**i}" for i in range(n_blocks)])
    return ax


def plot_2d_trajectory(
    trajectories: dict[str, np.ndarray],
    loss_fn=None,
    title: str = "Optimization Trajectories",
    xlim: tuple[float, float] = (-3, 3),
    ylim: tuple[float, float] = (-3, 3),
    ax=None,
):
    """Plot 2D optimization trajectories on a contour loss surface.

    trajectories: dict mapping name -> array of shape (steps, 2).
    loss_fn: callable(x, y) -> z for contour background (optional).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    # Contour background
    if loss_fn is not None:
        xs = np.linspace(xlim[0], xlim[1], 200)
        ys = np.linspace(ylim[0], ylim[1], 200)
        X, Y = np.meshgrid(xs, ys)
        Z = loss_fn(X, Y)
        ax.contourf(X, Y, Z, levels=50, cmap="RdYlBu_r", alpha=0.6)
        ax.contour(X, Y, Z, levels=20, colors="gray", alpha=0.3, linewidths=0.5)
    # Trajectories
    for name, traj in trajectories.items():
        ax.plot(traj[:, 0], traj[:, 1], "-o", markersize=2, label=name, linewidth=1.5)
        ax.plot(traj[0, 0], traj[0, 1], "s", markersize=6)  # start
        ax.plot(traj[-1, 0], traj[-1, 1], "*", markersize=10)  # end
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    return ax


def plot_pattern(pattern: torch.Tensor, shape: tuple[int, int] = (8, 8), title: str = "", ax=None):
    """Display a binary (+1/-1) pattern as an image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    img = pattern.detach().cpu().numpy().reshape(shape)
    ax.imshow(img, cmap="gray_r", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    return ax
