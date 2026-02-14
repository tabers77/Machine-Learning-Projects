"""Continual learning harness for the Nested Learning research question.

RQ: How does the separation of update timescales affect the stability-plasticity
tradeoff in neural networks?

Provides:
- CMS-based continual learner (multi-timescale updates)
- EWC baseline (Elastic Weight Consolidation)
- Naive fine-tuning baseline
- Evaluation metrics: accuracy matrix, forgetting, plasticity
"""

import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Shared MLP backbone (same architecture for all methods)
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """Simple MLP for classification. Used as backbone by all methods."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, n_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.head(h)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden representation before the classification head."""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return h


# ---------------------------------------------------------------------------
# CMS-based continual learner
# ---------------------------------------------------------------------------

class CMSClassifier(nn.Module):
    """MLP with CMS-style multi-timescale parameter groups.

    Splits the network into frequency levels:
    - Level 0 (fast / plastic): first hidden layer — updates every step
    - Level 1 (medium): second hidden layer — updates every c_base steps
    - Level 2 (slow / stable): classification head — updates every c_base^2 steps

    The key insight: slow layers act as implicit memory regularizers,
    retaining knowledge from previous tasks while fast layers adapt to new ones.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        n_classes: int = 2,
        c_base: int = 4,
    ):
        super().__init__()
        self.c_base = c_base

        # Level 0: fast (updates every step)
        self.fast = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        # Level 1: medium (updates every c_base steps)
        self.medium = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        # Level 2: slow (updates every c_base^2 steps)
        self.slow = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fast(x)
        h = self.medium(h)
        return self.slow(h)

    def get_param_groups(self) -> list[dict]:
        """Return parameter groups labeled by frequency level."""
        return [
            {"params": list(self.fast.parameters()), "level": 0, "name": "fast"},
            {"params": list(self.medium.parameters()), "level": 1, "name": "medium"},
            {"params": list(self.slow.parameters()), "level": 2, "name": "slow"},
        ]


# ---------------------------------------------------------------------------
# Training methods
# ---------------------------------------------------------------------------

class NaiveTrainer:
    """Naive fine-tuning: train on each task sequentially with no protection."""

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_task(
        self, train_x: torch.Tensor, train_y: torch.Tensor,
        n_epochs: int = 10, batch_size: int = 64,
    ) -> list[float]:
        loader = DataLoader(
            TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
        )
        self.model.train()
        losses = []
        for _ in range(n_epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                loss = F.cross_entropy(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        return losses


class EWCTrainer:
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    After each task, computes Fisher information matrix (diagonal approximation)
    and penalizes changes to important parameters on subsequent tasks.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, ewc_lambda: float = 400.0):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.ewc_lambda = ewc_lambda

        # Fisher information and reference parameters from previous tasks
        self._fisher: list[dict[str, torch.Tensor]] = []
        self._ref_params: list[dict[str, torch.Tensor]] = []

    def _compute_fisher(self, train_x: torch.Tensor, train_y: torch.Tensor, n_samples: int = 200):
        """Compute diagonal Fisher information from data."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        idx = torch.randperm(len(train_x))[:n_samples]
        for i in idx:
            self.model.zero_grad()
            out = self.model(train_x[i : i + 1])
            loss = F.cross_entropy(out, train_y[i : i + 1])
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2

        for n in fisher:
            fisher[n] /= n_samples
        return fisher

    def _ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty across all previous tasks."""
        penalty = torch.tensor(0.0)
        for fisher, ref in zip(self._fisher, self._ref_params):
            for n, p in self.model.named_parameters():
                penalty = penalty + (fisher[n] * (p - ref[n]) ** 2).sum()
        return penalty

    def train_task(
        self, train_x: torch.Tensor, train_y: torch.Tensor,
        n_epochs: int = 10, batch_size: int = 64,
    ) -> list[float]:
        loader = DataLoader(
            TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
        )
        self.model.train()
        losses = []
        for _ in range(n_epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                ce_loss = F.cross_entropy(self.model(xb), yb)
                ewc_loss = self._ewc_penalty()
                loss = ce_loss + (self.ewc_lambda / 2.0) * ewc_loss
                loss.backward()
                self.optimizer.step()
                losses.append(ce_loss.item())

        # After training, consolidate knowledge
        fisher = self._compute_fisher(train_x, train_y)
        self._fisher.append(fisher)
        self._ref_params.append(
            {n: p.data.clone() for n, p in self.model.named_parameters()}
        )
        return losses


class CMSTrainerContinual:
    """CMS-based continual learning trainer with multi-timescale updates.

    Different parameter groups update at different frequencies:
    - Fast group: every step (high plasticity, adapts to new task quickly)
    - Medium group: every c_base steps
    - Slow group: every c_base^2 steps (high stability, retains old knowledge)

    This creates an implicit regularization: slow-updating parameters naturally
    preserve information from previous tasks because they change less frequently.
    """

    def __init__(self, model: CMSClassifier, lr: float = 1e-3):
        self.model = model
        self.c_base = model.c_base
        self.global_step = 0

        # Separate optimizer per frequency level
        groups = model.get_param_groups()
        self.optimizers = {}
        self.update_freqs = {}
        for g in groups:
            name = g["name"]
            level = g["level"]
            self.optimizers[name] = torch.optim.Adam(g["params"], lr=lr)
            self.update_freqs[name] = self.c_base ** level

    def train_task(
        self, train_x: torch.Tensor, train_y: torch.Tensor,
        n_epochs: int = 10, batch_size: int = 64,
    ) -> list[float]:
        loader = DataLoader(
            TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
        )
        self.model.train()
        losses = []
        for _ in range(n_epochs):
            for xb, yb in loader:
                self.global_step += 1

                # Zero all grads
                for opt in self.optimizers.values():
                    opt.zero_grad()

                loss = F.cross_entropy(self.model(xb), yb)
                loss.backward()

                # Selective update based on frequency
                for name, opt in self.optimizers.items():
                    freq = self.update_freqs[name]
                    if self.global_step % freq == 0:
                        opt.step()
                    else:
                        opt.zero_grad()  # discard gradients for inactive groups

                losses.append(loss.item())
        return losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute classification accuracy."""
    model.eval()
    preds = model(x).argmax(dim=1)
    return (preds == y).float().mean().item()


def run_continual_experiment(
    benchmark,
    model: nn.Module,
    trainer,
    n_epochs: int = 10,
    batch_size: int = 64,
) -> dict:
    """Run a full continual learning experiment on a benchmark.

    Returns dict with:
    - 'accuracy_matrix': A[i,j] = accuracy on task j after training on task i
    - 'task_losses': list of per-task training losses
    - 'forgetting': per-task forgetting after all tasks
    - 'avg_accuracy': average accuracy after all tasks
    - 'avg_forgetting': average forgetting across tasks
    - 'plasticity': per-task accuracy immediately after training on that task
    """
    n_tasks = benchmark.n_tasks
    accuracy_matrix = torch.zeros(n_tasks, n_tasks)
    task_losses = []
    plasticity = []

    for task_id in range(n_tasks):
        task = benchmark.get_task(task_id)
        losses = trainer.train_task(
            task["train_x"], task["train_y"],
            n_epochs=n_epochs, batch_size=batch_size,
        )
        task_losses.append(losses)

        # Evaluate on all tasks seen so far
        for eval_id in range(n_tasks):
            eval_task = benchmark.get_task(eval_id)
            acc = evaluate(model, eval_task["test_x"], eval_task["test_y"])
            accuracy_matrix[task_id, eval_id] = acc

        plasticity.append(accuracy_matrix[task_id, task_id].item())

    # Compute forgetting: max accuracy on task j minus final accuracy
    forgetting = []
    for j in range(n_tasks):
        max_acc = accuracy_matrix[:, j].max().item()
        final_acc = accuracy_matrix[-1, j].item()
        forgetting.append(max_acc - final_acc)

    avg_accuracy = accuracy_matrix[-1, :].mean().item()
    avg_forgetting = sum(forgetting) / len(forgetting)

    return {
        "accuracy_matrix": accuracy_matrix,
        "task_losses": task_losses,
        "forgetting": forgetting,
        "avg_accuracy": avg_accuracy,
        "avg_forgetting": avg_forgetting,
        "plasticity": plasticity,
    }


# ---------------------------------------------------------------------------
# Visualization helpers for the research question
# ---------------------------------------------------------------------------

def plot_accuracy_matrix(acc_matrix: torch.Tensor, task_names: list[str] | None = None, title: str = "", ax=None):
    """Plot the accuracy matrix A[i,j] as a heatmap.

    A[i,j] = accuracy on task j after training on task i.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    n = acc_matrix.shape[0]
    if task_names is None:
        task_names = [f"Task {i}" for i in range(n)]

    im = ax.imshow(acc_matrix.numpy(), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticklabels([f"After {t}" for t in task_names])
    ax.set_xlabel("Evaluated on")
    ax.set_ylabel("Trained through")
    ax.set_title(title or "Accuracy Matrix")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = acc_matrix[i, j].item()
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Accuracy")
    return ax


def plot_forgetting_vs_c_base(
    results: dict[str, dict],
    metric: str = "avg_forgetting",
    ax=None,
):
    """Plot a metric (forgetting, accuracy, etc.) vs C_base or method name.

    results: dict mapping label -> experiment result dict.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    labels = list(results.keys())
    values = [results[k][metric] for k in labels]

    bars = ax.bar(labels, values, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"][:len(labels)])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Method")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    return ax


def plot_stability_plasticity(results: dict[str, dict], ax=None):
    """Scatter plot: plasticity (x) vs stability (y = 1 - forgetting).

    Each point is a method. The ideal is top-right corner (high plasticity + high stability).
    Uses adjustText-style manual offsets when points cluster tightly.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    # Collect data points
    points = []
    for name, res in results.items():
        plasticity = sum(res["plasticity"]) / len(res["plasticity"])
        stability = 1.0 - res["avg_forgetting"]
        points.append((name, plasticity, stability))

    # Sort by stability for consistent layout
    points.sort(key=lambda p: p[2])

    # Group methods by type for coloring
    colors_map = {}
    for name, pl, st in points:
        if "CMS" in name:
            colors_map[name] = "#2ecc71"
        elif "EWC" in name:
            colors_map[name] = "#3498db"
        else:
            colors_map[name] = "#e74c3c"

    markers_map = {}
    for name, _, _ in points:
        if "CMS" in name:
            markers_map[name] = "o"
        elif "EWC" in name:
            markers_map[name] = "D"
        else:
            markers_map[name] = "s"

    # Plot points
    for name, pl, st in points:
        ax.scatter(pl, st, s=180, c=colors_map[name], marker=markers_map[name],
                   zorder=5, edgecolors="black", linewidth=1, label=name)

    # Smart label placement: spread labels vertically when clustered
    label_y_used = []
    for name, pl, st in points:
        # Find a y-offset that doesn't overlap
        offset_y = 12
        for used_y in label_y_used:
            if abs(st - used_y) < 0.008:
                offset_y += 14
        label_y_used.append(st)
        ax.annotate(
            name, (pl, st),
            textcoords="offset points",
            xytext=(-60, offset_y),
            fontsize=8,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    ax.set_xlabel("Plasticity (avg task accuracy right after learning)", fontsize=11)
    ax.set_ylabel("Stability (1 - avg forgetting)", fontsize=11)
    ax.set_title("Stability-Plasticity Tradeoff", fontsize=13)

    # Zoom into the actual data range with padding
    all_pl = [p[1] for p in points]
    all_st = [p[2] for p in points]
    pad = 0.02
    ax.set_xlim(min(all_pl) - pad, max(all_pl) + pad)
    ax.set_ylim(min(all_st) - pad, max(all_st) + pad)

    # Ideal direction arrow
    ax.annotate(
        "Better", xy=(max(all_pl) + pad * 0.3, max(all_st) + pad * 0.3),
        fontsize=10, fontstyle="italic", color="gray",
        ha="right", va="top",
    )

    ax.grid(True, alpha=0.3)
    return ax


def plot_task_accuracy_over_time(acc_matrix: torch.Tensor, task_names: list[str] | None = None, title: str = "", ax=None):
    """Line plot: per-task accuracy as training progresses through tasks.

    Shows how each task's accuracy changes as the model trains on subsequent tasks.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    n = acc_matrix.shape[0]
    if task_names is None:
        task_names = [f"Task {i}" for i in range(n)]

    for j in range(n):
        accs = acc_matrix[:, j].numpy()
        # Only show from when task j was first trained
        x = list(range(j, n))
        y = accs[j:]
        ax.plot(x, y, "o-", label=task_names[j], linewidth=2, markersize=6)

    ax.set_xticks(range(n))
    ax.set_xticklabels([f"After {t}" for t in task_names], rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title or "Per-Task Accuracy Over Training Sequence")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    return ax
