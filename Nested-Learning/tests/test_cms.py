"""Tests for Continuum Memory System components."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from src.cms import MemoryBlock, ContinuumMemorySystem, CMSTrainer


class TestMemoryBlock:
    def test_output_shape(self):
        block = MemoryBlock(input_dim=16, hidden_dim=32, output_dim=16, update_freq=4)
        x = torch.randn(8, 16)
        out = block(x)
        assert out.shape == (8, 16)

    def test_should_update(self):
        block = MemoryBlock(input_dim=4, hidden_dim=8, output_dim=4, update_freq=4)
        assert block.should_update(0)
        assert not block.should_update(1)
        assert not block.should_update(2)
        assert not block.should_update(3)
        assert block.should_update(4)
        assert block.should_update(8)


class TestContinuumMemorySystem:
    def test_output_shape(self):
        cms = ContinuumMemorySystem(
            input_dim=16, hidden_dim=32, output_dim=8, n_levels=3, c_base=4
        )
        x = torch.randn(4, 16)
        out = cms(x)
        assert out.shape == (4, 8)

    def test_update_frequencies(self):
        cms = ContinuumMemorySystem(
            input_dim=16, hidden_dim=32, output_dim=8, n_levels=4, c_base=3
        )
        freqs = cms.get_update_frequencies()
        assert freqs == [1, 3, 9, 27]

    def test_active_blocks(self):
        cms = ContinuumMemorySystem(
            input_dim=16, hidden_dim=32, output_dim=8, n_levels=3, c_base=4
        )
        # Step 1: only block 0 (freq=1) should update? No: 1%1==0, 1%4!=0, 1%16!=0
        active = cms.get_active_blocks(1)
        assert 0 in active
        assert 1 not in active

        # Step 4: blocks 0 and 1
        active = cms.get_active_blocks(4)
        assert 0 in active
        assert 1 in active
        assert 2 not in active

        # Step 16: all blocks
        active = cms.get_active_blocks(16)
        assert active == [0, 1, 2]

    def test_gradient_flow(self):
        cms = ContinuumMemorySystem(
            input_dim=8, hidden_dim=16, output_dim=4, n_levels=3, c_base=2
        )
        x = torch.randn(4, 8, requires_grad=True)
        out = cms(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestCMSTrainer:
    def test_training_step(self):
        cms = ContinuumMemorySystem(
            input_dim=8, hidden_dim=16, output_dim=4, n_levels=3, c_base=2
        )
        trainer = CMSTrainer(cms, lr=1e-3)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        info = trainer.train_step(x, y, nn.MSELoss())

        assert "loss" in info
        assert "active_blocks" in info
        assert info["global_step"] == 1

    def test_selective_updates(self):
        """Verify that inactive blocks don't get updated."""
        cms = ContinuumMemorySystem(
            input_dim=8, hidden_dim=16, output_dim=4, n_levels=3, c_base=4
        )
        trainer = CMSTrainer(cms, lr=1e-3)

        # Save block 2 weights (freq=16, won't update for first 15 steps)
        w_block2_before = cms.blocks[2].net[0].weight.data.clone()

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)

        # Take 3 steps (block 2 has freq=16, should not update)
        for _ in range(3):
            trainer.train_step(x, y, nn.MSELoss())

        w_block2_after = cms.blocks[2].net[0].weight.data.clone()
        assert torch.equal(w_block2_before, w_block2_after), \
            "Block 2 (freq=16) should not have updated in 3 steps"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
