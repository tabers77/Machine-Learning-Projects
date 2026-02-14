"""Tests for Hope architecture components."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from src.hope import TitanBlock, HopeBlock, HopeModel, HopeTrainer


class TestTitanBlock:
    def test_output_shape(self):
        block = TitanBlock(embed_dim=32, n_heads=2)
        x = torch.randn(2, 10, 32)
        out, surprise = block(x)
        assert out.shape == (2, 10, 32)
        assert surprise.shape == (2, 10, 1)

    def test_surprise_non_negative(self):
        block = TitanBlock(embed_dim=32, n_heads=2)
        x = torch.randn(2, 10, 32)
        _, surprise = block(x)
        assert (surprise >= 0).all()

    def test_gradient_flow(self):
        block = TitanBlock(embed_dim=32, n_heads=2)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out, _ = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestHopeBlock:
    def test_output_shape(self):
        block = HopeBlock(embed_dim=32, n_heads=2, c_base=4)
        x = torch.randn(2, 10, 32)
        out, surprise = block(x)
        assert out.shape == (2, 10, 32)
        assert surprise.shape == (2, 10, 1)

    def test_active_levels(self):
        block = HopeBlock(embed_dim=32, n_heads=2, c_base=4)
        assert block.get_active_levels(1) == ["attn"]
        assert "short_term" in block.get_active_levels(4)
        assert "long_term" in block.get_active_levels(16)


class TestHopeModel:
    def test_output_shape(self):
        model = HopeModel(vocab_size=50, embed_dim=32, n_heads=2, n_layers=2, max_seq_len=64)
        x = torch.randint(0, 50, (2, 20))
        logits, surprises = model(x)
        assert logits.shape == (2, 20, 50)
        assert len(surprises) == 2

    def test_titan_mode(self):
        model = HopeModel(vocab_size=50, embed_dim=32, n_heads=2, n_layers=2, use_titan=True)
        x = torch.randint(0, 50, (2, 20))
        logits, _ = model(x)
        assert logits.shape == (2, 20, 50)

    def test_generate(self):
        model = HopeModel(vocab_size=50, embed_dim=32, n_heads=2, n_layers=2, max_seq_len=64)
        prompt = torch.randint(0, 50, (1, 5))
        out = model.generate(prompt, max_new_tokens=10)
        assert out.shape == (1, 15)

    def test_gradient_flow(self):
        model = HopeModel(vocab_size=50, embed_dim=32, n_heads=2, n_layers=2)
        x = torch.randint(0, 50, (2, 20))
        logits, _ = model(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 50), x.view(-1))
        loss.backward()
        # Check all parameters got gradients
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestHopeTrainer:
    def test_train_step(self):
        model = HopeModel(vocab_size=50, embed_dim=32, n_heads=2, n_layers=2)
        trainer = HopeTrainer(model, lr=1e-3)
        x = torch.randint(0, 50, (4, 20))
        y = torch.randint(0, 50, (4, 20))
        info = trainer.train_step(x, y)
        assert "loss" in info
        assert "perplexity" in info
        assert "surprise" in info
        assert info["step"] == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
