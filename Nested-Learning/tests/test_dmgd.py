"""Tests for DMGD components."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from src.dmgd import LinearMomentum, DeepMomentum, DMGDOptimizer


class TestLinearMomentum:
    def test_output_shape(self):
        lm = LinearMomentum(beta=0.9)
        g = torch.randn(10)
        m = torch.zeros(10)
        out = lm(g, m)
        assert out.shape == (10,)

    def test_first_step_equals_gradient(self):
        lm = LinearMomentum(beta=0.9)
        g = torch.randn(10)
        m = torch.zeros(10)
        out = lm(g, m)
        assert torch.allclose(out, g), "With zero momentum, output should equal gradient"

    def test_matches_pytorch_sgd(self):
        """Verify LinearMomentum matches torch.optim.SGD momentum behavior."""
        torch.manual_seed(0)
        # Our implementation
        lm = LinearMomentum(beta=0.9)
        m = torch.zeros(5)
        g1 = torch.randn(5)
        g2 = torch.randn(5)
        m = lm(g1, m)  # m1 = g1
        m = lm(g2, m)  # m2 = 0.9*g1 + g2

        expected = 0.9 * g1 + g2
        assert torch.allclose(m, expected, atol=1e-6)


class TestDeepMomentum:
    def test_output_shape(self):
        dm = DeepMomentum(hidden_dim=16)
        g = torch.randn(3, 4)
        m = torch.zeros(3, 4)
        out = dm(g, m)
        assert out.shape == (3, 4)

    def test_differentiable(self):
        dm = DeepMomentum(hidden_dim=16)
        g = torch.randn(10, requires_grad=True)
        m = torch.zeros(10)
        out = dm(g, m)
        loss = out.sum()
        loss.backward()
        assert g.grad is not None


class TestDMGDOptimizer:
    def test_basic_step(self):
        model = nn.Linear(4, 2)
        dm = DeepMomentum(hidden_dim=16)
        opt = DMGDOptimizer(model.parameters(), dm, lr=0.01)

        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        w_before = model.weight.data.clone()
        opt.step()
        # Weights should have changed
        assert not torch.equal(model.weight.data, w_before)

    def test_meta_step_returns_loss(self):
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        dm = DeepMomentum(hidden_dim=16)
        opt = DMGDOptimizer(model.parameters(), dm, lr=0.01, meta_lr=1e-3)

        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        def loss_fn():
            return nn.CrossEntropyLoss()(model(x), y)

        meta_loss = opt.meta_step(loss_fn, lambda: model(x))
        assert isinstance(meta_loss, float)
        assert meta_loss > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
