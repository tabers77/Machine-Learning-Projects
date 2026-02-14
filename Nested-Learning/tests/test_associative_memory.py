"""Tests for associative memory components."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from src.associative_memory import (
    HopfieldNetwork,
    ModernHopfield,
    AttentionAsAssociativeMemory,
    BackpropAsAssociativeMemory,
)
from src.data import generate_binary_patterns


class TestHopfieldNetwork:
    def test_store_and_perfect_retrieval(self):
        dim = 64
        net = HopfieldNetwork(dim)
        patterns = generate_binary_patterns(3, dim, rng=__import__("numpy").random.default_rng(0))
        net.store(patterns)
        for i in range(3):
            retrieved = net.retrieve(patterns[i])
            assert torch.equal(retrieved, patterns[i]), f"Pattern {i} not retrieved exactly"

    def test_corrupted_retrieval(self):
        dim = 64
        net = HopfieldNetwork(dim)
        rng = __import__("numpy").random.default_rng(42)
        patterns = generate_binary_patterns(3, dim, rng=rng)
        net.store(patterns)
        # Flip 10% of bits
        query = patterns[0].clone()
        flip_idx = torch.randperm(dim)[:dim // 10]
        query[flip_idx] *= -1
        retrieved = net.retrieve(query)
        # Should recover original (overlap > 90%)
        overlap = (retrieved == patterns[0]).float().mean()
        assert overlap > 0.9, f"Overlap too low: {overlap}"

    def test_energy_decreases(self):
        dim = 64
        net = HopfieldNetwork(dim)
        patterns = generate_binary_patterns(3, dim, rng=__import__("numpy").random.default_rng(0))
        net.store(patterns)
        query = patterns[0].clone()
        query[:dim // 4] *= -1
        e_before = net.energy(query)
        retrieved = net.retrieve(query)
        e_after = net.energy(retrieved)
        assert e_after <= e_before, "Energy should decrease during retrieval"

    def test_weight_matrix_shape(self):
        dim = 32
        net = HopfieldNetwork(dim)
        patterns = generate_binary_patterns(2, dim)
        net.store(patterns)
        assert net.W.shape == (dim, dim)
        assert torch.diag(net.W).abs().sum() == 0, "Diagonal should be zero"


class TestModernHopfield:
    def test_output_shape(self):
        dim = 16
        mh = ModernHopfield()
        patterns = torch.randn(5, dim)
        mh.store(patterns)
        query = torch.randn(3, dim)
        out = mh.retrieve(query)
        assert out.shape == (3, dim)

    def test_single_query(self):
        dim = 16
        mh = ModernHopfield()
        patterns = torch.randn(5, dim)
        mh.store(patterns)
        out = mh.retrieve(torch.randn(dim))
        assert out.shape == (1, dim)


class TestAttentionEquivalence:
    def test_attention_equals_modern_hopfield(self):
        """Key test: attention and modern Hopfield should produce identical results
        when K=V=X (the stored patterns)."""
        torch.manual_seed(0)
        dim = 32
        n_patterns = 10
        n_queries = 4
        X = torch.randn(n_patterns, dim)
        Q = torch.randn(n_queries, dim)

        # Modern Hopfield
        mh = ModernHopfield()
        mh.store(X)
        hopfield_out = mh.retrieve(Q)

        # Attention with K=V=X
        attn = AttentionAsAssociativeMemory(dim)
        attn.store(keys=X, values=X)
        attn_out = attn.retrieve(Q)

        assert torch.allclose(hopfield_out, attn_out, atol=1e-5), (
            f"Max diff: {(hopfield_out - attn_out).abs().max()}"
        )


class TestBackpropAsAssociativeMemory:
    def test_outer_product_shape(self):
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(4, 8))
        model.add_module("relu", nn.ReLU())
        model.add_module("fc2", nn.Linear(8, 2))

        wrapper = BackpropAsAssociativeMemory(model, "fc1")
        x = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        update = wrapper.get_outer_product_update()
        assert update is not None
        assert update.shape == (8, 4), f"Expected (8,4), got {update.shape}"
        wrapper.remove_hooks()

    def test_gradient_flow(self):
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(4, 8))
        model.add_module("relu", nn.ReLU())
        model.add_module("fc2", nn.Linear(8, 2))

        wrapper = BackpropAsAssociativeMemory(model, "fc1")
        x = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        update = wrapper.get_outer_product_update()
        assert update is not None
        assert update.abs().sum() > 0, "Update should be non-zero"
        wrapper.remove_hooks()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
