"""Tests for continual learning module."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
from src.continual import (
    MLPClassifier,
    CMSClassifier,
    NaiveTrainer,
    EWCTrainer,
    CMSTrainerContinual,
    evaluate,
    run_continual_experiment,
)


def _make_synthetic_benchmark(n_tasks=3, n_samples=100, input_dim=32, n_classes=2):
    """Create a tiny synthetic benchmark for testing (no MNIST download needed)."""

    class SyntheticBenchmark:
        def __init__(self):
            self.n_tasks = n_tasks
            self.tasks = []
            rng = np.random.default_rng(42)
            for t in range(n_tasks):
                # Each task has a different linear separator
                w = rng.standard_normal(input_dim).astype(np.float32)
                x = torch.from_numpy(rng.standard_normal((n_samples, input_dim)).astype(np.float32))
                y = (x @ torch.from_numpy(w) > 0).long()
                # Split 80/20 train/test
                split = int(0.8 * n_samples)
                self.tasks.append({
                    "train_x": x[:split], "train_y": y[:split],
                    "test_x": x[split:], "test_y": y[split:],
                    "name": f"task_{t}",
                })

        def get_task(self, task_id):
            return self.tasks[task_id]

    return SyntheticBenchmark()


class TestMLPClassifier:
    def test_output_shape(self):
        model = MLPClassifier(input_dim=32, hidden_dim=64, n_classes=2)
        x = torch.randn(8, 32)
        out = model(x)
        assert out.shape == (8, 2)

    def test_gradient_flow(self):
        model = MLPClassifier(input_dim=32, hidden_dim=64, n_classes=2)
        x = torch.randn(8, 32)
        y = torch.randint(0, 2, (8,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestCMSClassifier:
    def test_output_shape(self):
        model = CMSClassifier(input_dim=32, hidden_dim=64, n_classes=2, c_base=4)
        x = torch.randn(8, 32)
        out = model(x)
        assert out.shape == (8, 2)

    def test_param_groups(self):
        model = CMSClassifier(input_dim=32, hidden_dim=64, n_classes=2, c_base=4)
        groups = model.get_param_groups()
        assert len(groups) == 3
        assert groups[0]["name"] == "fast"
        assert groups[1]["name"] == "medium"
        assert groups[2]["name"] == "slow"

    def test_c_base_stored(self):
        model = CMSClassifier(input_dim=32, hidden_dim=64, n_classes=2, c_base=8)
        assert model.c_base == 8


class TestNaiveTrainer:
    def test_training_reduces_loss(self):
        model = MLPClassifier(input_dim=32, hidden_dim=64, n_classes=2)
        trainer = NaiveTrainer(model, lr=1e-2)
        x = torch.randn(100, 32)
        y = (x[:, 0] > 0).long()
        losses = trainer.train_task(x, y, n_epochs=5, batch_size=32)
        assert losses[-1] < losses[0], "Training should reduce loss"


class TestEWCTrainer:
    def test_ewc_penalty_starts_zero(self):
        model = MLPClassifier(input_dim=32, hidden_dim=64, n_classes=2)
        trainer = EWCTrainer(model, lr=1e-2, ewc_lambda=100.0)
        # Before any task, penalty should be 0
        penalty = trainer._ewc_penalty()
        assert penalty.item() == 0.0

    def test_ewc_accumulates_fisher(self):
        model = MLPClassifier(input_dim=32, hidden_dim=64, n_classes=2)
        trainer = EWCTrainer(model, lr=1e-2, ewc_lambda=100.0)
        x = torch.randn(100, 32)
        y = (x[:, 0] > 0).long()
        trainer.train_task(x, y, n_epochs=2, batch_size=32)
        assert len(trainer._fisher) == 1
        assert len(trainer._ref_params) == 1

        # Train second task
        trainer.train_task(x, y, n_epochs=2, batch_size=32)
        assert len(trainer._fisher) == 2


class TestCMSTrainerContinual:
    def test_selective_updates(self):
        model = CMSClassifier(input_dim=32, hidden_dim=64, n_classes=2, c_base=100)
        trainer = CMSTrainerContinual(model, lr=1e-2)

        # Save slow layer weights
        slow_w_before = model.slow.weight.data.clone()

        x = torch.randn(50, 32)
        y = (x[:, 0] > 0).long()
        trainer.train_task(x, y, n_epochs=1, batch_size=50)

        # With c_base=100, slow layer (freq=10000) should NOT have updated
        # in just ~1 step
        slow_w_after = model.slow.weight.data.clone()
        assert torch.equal(slow_w_before, slow_w_after), \
            "Slow layer should not update in 1 step with c_base=100"


class TestEvaluate:
    def test_perfect_accuracy(self):
        model = MLPClassifier(input_dim=4, hidden_dim=8, n_classes=2)
        # Rig the model to always predict class 0
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias[0] = 10.0
        x = torch.randn(20, 4)
        y = torch.zeros(20, dtype=torch.long)
        acc = evaluate(model, x, y)
        assert acc == 1.0


class TestRunExperiment:
    def test_full_experiment(self):
        benchmark = _make_synthetic_benchmark(n_tasks=3, n_samples=80, input_dim=32)
        model = MLPClassifier(input_dim=32, hidden_dim=32, n_classes=2)
        trainer = NaiveTrainer(model, lr=1e-2)

        result = run_continual_experiment(
            benchmark, model, trainer, n_epochs=2, batch_size=32
        )

        assert result["accuracy_matrix"].shape == (3, 3)
        assert 0 <= result["avg_accuracy"] <= 1
        assert len(result["forgetting"]) == 3
        assert len(result["plasticity"]) == 3
        assert len(result["task_losses"]) == 3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
