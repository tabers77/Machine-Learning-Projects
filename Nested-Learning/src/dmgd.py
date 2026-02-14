"""Deep Momentum Gradient Descent (DMGD).

Core claim: Replace linear momentum with a learned MLP to model non-linear
loss landscape dynamics. The momentum function becomes a trainable neural
network that learns to predict better update directions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMomentum(nn.Module):
    """Explicit implementation of standard linear momentum (baseline).

    m_t = beta * m_{t-1} + g_t
    theta_{t+1} = theta_t - lr * m_t
    """

    def __init__(self, beta: float = 0.9):
        super().__init__()
        self.beta = beta

    def forward(self, gradient: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        """Compute new momentum: m_t = beta * m_{t-1} + g_t."""
        return self.beta * momentum + gradient


class DeepMomentum(nn.Module):
    """MLP replacing linear momentum. Per-parameter shared-weight network.

    Input features per coordinate: [g_t, m_t, g_t^2, m_t^2, sign(g_t)]
    Output: new momentum value (scalar per coordinate).
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_as_linear_momentum()

    def _init_as_linear_momentum(self):
        """Initialize weights so the network approximates linear momentum (warm start)."""
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.bias)
                    nn.init.normal_(layer.weight, std=0.01)

    def forward(self, gradient: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        """Compute new momentum via learned MLP.

        gradient, momentum: tensors of any shape (processed per-element).
        Returns: new momentum, same shape as input.
        """
        orig_shape = gradient.shape
        g = gradient.reshape(-1, 1)
        m = momentum.reshape(-1, 1)
        features = torch.cat([
            g,
            m,
            g ** 2,
            m ** 2,
            torch.sign(g),
        ], dim=1)  # (N, 5)
        out = self.net(features)  # (N, 1)
        return out.reshape(orig_shape)


class DMGDOptimizer:
    """PyTorch-compatible optimizer wrapping DeepMomentum.

    Meta-updates the DeepMomentum MLP every `meta_every` steps via
    truncated BPTT over `unroll_steps` optimization steps.
    """

    def __init__(
        self,
        params,
        deep_momentum: DeepMomentum,
        lr: float = 0.01,
        meta_lr: float = 1e-4,
        unroll_steps: int = 5,
        meta_every: int = 5,
        grad_clip: float = 1.0,
    ):
        self.params = list(params)
        self.deep_momentum = deep_momentum
        self.lr = lr
        self.meta_lr = meta_lr
        self.unroll_steps = unroll_steps
        self.meta_every = meta_every
        self.grad_clip = grad_clip

        self.meta_optimizer = torch.optim.Adam(
            deep_momentum.parameters(), lr=meta_lr
        )

        # Initialize momentum buffers
        self.momentum_buffers = [torch.zeros_like(p.data) for p in self.params]
        self.step_count = 0

        # For truncated BPTT: accumulate base losses
        self._unroll_losses = []

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self, loss: torch.Tensor | None = None):
        """Apply one optimization step.

        If loss is provided AND it's time for meta-update, performs BPTT.
        """
        self.step_count += 1

        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad.data
                # Compute new momentum via the learned MLP (no meta-grad here)
                new_m = self.deep_momentum(g, self.momentum_buffers[i])
                self.momentum_buffers[i] = new_m.clone()
                p.data -= self.lr * new_m

    def meta_step(self, loss_fn, forward_fn):
        """Perform a meta-optimization step via truncated BPTT.

        loss_fn: callable() -> loss (recomputes the current loss)
        forward_fn: callable() that runs the forward pass

        This creates a computational graph through unroll_steps of optimization
        and backprops through the deep momentum network.
        """
        # Save parameter state
        saved_params = [p.data.clone() for p in self.params]
        saved_momentum = [m.clone() for m in self.momentum_buffers]

        # Create parameter copies that track gradients through momentum
        param_copies = [p.data.clone().requires_grad_(True) for p in self.params]
        momentum_copies = [m.clone() for m in self.momentum_buffers]

        total_meta_loss = torch.tensor(0.0)

        for _ in range(self.unroll_steps):
            # Temporarily replace params
            for p, pc in zip(self.params, param_copies):
                p.data = pc.data

            # Forward pass and compute gradients
            loss = loss_fn()
            grads = torch.autograd.grad(
                loss, param_copies, create_graph=True, allow_unused=True
            )

            # Apply deep momentum
            new_param_copies = []
            new_momentum_copies = []
            for pc, g, m in zip(param_copies, grads, momentum_copies):
                if g is None:
                    new_param_copies.append(pc)
                    new_momentum_copies.append(m)
                    continue
                new_m = self.deep_momentum(g, m)
                new_p = pc - self.lr * new_m
                new_param_copies.append(new_p)
                new_momentum_copies.append(new_m)

            param_copies = new_param_copies
            momentum_copies = new_momentum_copies

            # Accumulate loss
            for p, pc in zip(self.params, param_copies):
                p.data = pc.data
            meta_loss = loss_fn()
            total_meta_loss = total_meta_loss + meta_loss

        # Backprop through the meta-loss to update deep momentum
        self.meta_optimizer.zero_grad()
        total_meta_loss.backward()
        nn.utils.clip_grad_norm_(self.deep_momentum.parameters(), self.grad_clip)
        self.meta_optimizer.step()

        # Restore original parameters (meta-step is separate from base optimization)
        for p, sp in zip(self.params, saved_params):
            p.data = sp
        self.momentum_buffers = saved_momentum

        return total_meta_loss.item()
