"""Associative memory modules: Hopfield networks, attention, backprop & momentum views.

Core claim from the paper: backprop, attention, and momentum are all associative
memory modules operating at different timescales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HopfieldNetwork:
    """Classical binary Hopfield network with Hebbian learning.

    Energy: E = -0.5 * x^T W x
    Retrieval: x_new = sign(W @ x)
    Capacity: ~0.14 * dim patterns (classical bound).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.W = torch.zeros(dim, dim)
        self.n_stored = 0

    def store(self, patterns: torch.Tensor):
        """Store patterns via outer-product (Hebbian) rule.

        patterns: (n_patterns, dim) tensor with +1/-1 entries.
        """
        n = patterns.shape[0]
        # W = (1/N) * sum_i x_i x_i^T, zero diagonal
        self.W = (patterns.T @ patterns) / self.dim
        self.W.fill_diagonal_(0)
        self.n_stored = n

    def retrieve(self, query: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """Retrieve pattern from a (possibly corrupted) query via synchronous update.

        query: (dim,) tensor with +1/-1 entries.
        Returns: retrieved pattern (dim,).
        """
        x = query.clone().float()
        for _ in range(steps):
            x_new = torch.sign(self.W @ x)
            # Replace zeros with previous values (no change if net input is 0)
            x_new[x_new == 0] = x[x_new == 0]
            if torch.equal(x_new, x):
                break
            x = x_new
        return x

    def energy(self, x: torch.Tensor) -> float:
        """Compute Hopfield energy E = -0.5 * x^T W x."""
        return -0.5 * (x @ self.W @ x).item()


class ModernHopfield(nn.Module):
    """Modern (continuous) Hopfield network with exponential energy.

    Retrieval is mathematically equivalent to attention:
        output = softmax(X^T q / sqrt(d)) @ X

    where X is the stored memory matrix and q is the query.
    """

    def __init__(self):
        super().__init__()
        self.memory: torch.Tensor | None = None

    def store(self, patterns: torch.Tensor):
        """Store patterns as memory matrix.

        patterns: (n_patterns, dim) — each row is a stored pattern.
        """
        self.memory = patterns

    def retrieve(self, query: torch.Tensor, beta: float | None = None) -> torch.Tensor:
        """Retrieve via modern Hopfield update (= attention).

        query: (dim,) or (n_queries, dim).
        beta: inverse temperature. Default: 1/sqrt(dim).
        """
        assert self.memory is not None, "No patterns stored"
        X = self.memory  # (N, d)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        d = query.shape[-1]
        if beta is None:
            beta = 1.0 / math.sqrt(d)
        # scores: (n_queries, N)
        scores = query @ X.T * beta
        weights = F.softmax(scores, dim=-1)
        # output: (n_queries, d)
        return weights @ X


class AttentionAsAssociativeMemory(nn.Module):
    """Standard scaled dot-product attention with explicit store/retrieve interface.

    Demonstrates the equivalence: attention IS modern Hopfield retrieval.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def store(self, keys: torch.Tensor, values: torch.Tensor):
        """Store key-value pairs.

        keys: (n, dim), values: (n, dim).
        """
        self.keys = keys
        self.values = values

    def retrieve(self, queries: torch.Tensor) -> torch.Tensor:
        """Standard attention retrieval: softmax(Q K^T / sqrt(d)) V.

        queries: (m, dim) -> returns (m, dim).
        """
        assert self.keys is not None and self.values is not None
        scale = math.sqrt(self.dim)
        scores = queries @ self.keys.T / scale
        weights = F.softmax(scores, dim=-1)
        return weights @ self.values


class BackpropAsAssociativeMemory:
    """Diagnostic wrapper showing that weight updates in backprop are
    outer products of errors and activations — the same Hebbian structure
    as associative memory storage.

    For a linear layer y = Wx, the gradient is:
        dW = (dL/dy) @ x^T   (outer product of error signal and activation)
    """

    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self._activations: dict[str, torch.Tensor] = {}
        self._gradients: dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                h1 = module.register_forward_hook(self._save_activation(name))
                h2 = module.register_full_backward_hook(self._save_gradient(name))
                self._hooks.extend([h1, h2])

    def _save_activation(self, name):
        def hook(module, input, output):
            self._activations[name] = input[0].detach()
        return hook

    def _save_gradient(self, name):
        def hook(module, grad_input, grad_output):
            self._gradients[name] = grad_output[0].detach()
        return hook

    def get_outer_product_update(self) -> torch.Tensor | None:
        """After a forward + backward pass, return the outer product dL/dy @ x^T.

        This is the 'associative memory write' of backprop.
        """
        act = self._activations.get(self.layer_name)
        grad = self._gradients.get(self.layer_name)
        if act is None or grad is None:
            return None
        # grad: (batch, out_dim), act: (batch, in_dim)
        # Average over batch
        return (grad.T @ act) / act.shape[0]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class MomentumAsAssociativeMemory:
    """Wrapper that exposes SGD momentum buffer as an associative memory.

    The momentum buffer m_t = beta * m_{t-1} + g_t is an exponentially-weighted
    sum of past gradients — a form of key-value memory where keys are timesteps
    and values are gradient directions, weighted by recency.
    """

    def __init__(self, optimizer: torch.optim.SGD):
        assert isinstance(optimizer, torch.optim.SGD), "Requires SGD optimizer"
        self.optimizer = optimizer

    def get_momentum_buffers(self) -> list[torch.Tensor]:
        """Return current momentum buffers for all parameter groups."""
        buffers = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                if "momentum_buffer" in state:
                    buffers.append(state["momentum_buffer"].clone())
        return buffers

    def get_momentum_history(
        self, model: nn.Module, loss_fn, data_iter, n_steps: int
    ) -> list[list[torch.Tensor]]:
        """Run n_steps of SGD and record momentum buffer snapshots.

        Returns list of length n_steps, each element is a list of buffers.
        """
        history = []
        for step, (x, y) in zip(range(n_steps), data_iter):
            self.optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            history.append(self.get_momentum_buffers())
        return history
