"""Shared data utilities for Nested Learning experiments."""

import math
import os
import string

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Binary pattern generators (for Hopfield / associative memory)
# ---------------------------------------------------------------------------

def generate_binary_patterns(n_patterns: int, dim: int, rng=None) -> torch.Tensor:
    """Generate random binary (+1/-1) patterns for Hopfield experiments.

    Returns: Tensor of shape (n_patterns, dim).
    """
    if rng is None:
        rng = np.random.default_rng()
    bits = rng.choice([-1, 1], size=(n_patterns, dim)).astype(np.float32)
    return torch.from_numpy(bits)


def generate_letter_patterns() -> dict[str, torch.Tensor]:
    """Generate 8x8 binary pixel patterns for letters A-E.

    Returns dict mapping letter -> flat tensor of shape (64,) with values +1/-1.
    """
    letters = {
        "A": [
            "..####..",
            ".#....#.",
            "#......#",
            "#......#",
            "########",
            "#......#",
            "#......#",
            "#......#",
        ],
        "B": [
            "#######.",
            "#......#",
            "#......#",
            "#######.",
            "#......#",
            "#......#",
            "#......#",
            "#######.",
        ],
        "C": [
            ".#######",
            "#.......",
            "#.......",
            "#.......",
            "#.......",
            "#.......",
            "#.......",
            ".#######",
        ],
        "D": [
            "######..",
            "#.....#.",
            "#......#",
            "#......#",
            "#......#",
            "#......#",
            "#.....#.",
            "######..",
        ],
        "E": [
            "########",
            "#.......",
            "#.......",
            "#####...",
            "#.......",
            "#.......",
            "#.......",
            "########",
        ],
    }
    result = {}
    for name, rows in letters.items():
        pixels = []
        for row in rows:
            for ch in row:
                pixels.append(1.0 if ch == "#" else -1.0)
        result[name] = torch.tensor(pixels, dtype=torch.float32)
    return result


# ---------------------------------------------------------------------------
# Signal generators (for CMS experiments)
# ---------------------------------------------------------------------------

def generate_multifreq_signal(
    freqs: list[float],
    duration: float = 1.0,
    sample_rate: int = 1000,
    amplitudes: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a sum-of-sines signal.

    Returns (times, signal) both as 1-D tensors.
    """
    t = torch.linspace(0, duration, int(duration * sample_rate))
    if amplitudes is None:
        amplitudes = [1.0] * len(freqs)
    signal = torch.zeros_like(t)
    for freq, amp in zip(freqs, amplitudes):
        signal = signal + amp * torch.sin(2 * math.pi * freq * t)
    return t, signal


# ---------------------------------------------------------------------------
# Copy / recall task (for CMS long-range memory)
# ---------------------------------------------------------------------------

def generate_copy_task(
    seq_len: int,
    delay: int,
    vocab_size: int = 8,
    n_samples: int = 1000,
    rng=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate copy/recall task data.

    Input sequence: [random tokens | blank tokens for `delay` steps | random tokens repeated]
    Target: the original random tokens after the delay.

    Returns (inputs, targets) both LongTensors of shape (n_samples, total_len).
    The blank token is `vocab_size` and the total_len = 2*seq_len + delay.
    """
    if rng is None:
        rng = np.random.default_rng()

    blank = vocab_size  # extra token for blank
    total_len = 2 * seq_len + delay
    inputs = np.full((n_samples, total_len), blank, dtype=np.int64)
    targets = np.full((n_samples, total_len), blank, dtype=np.int64)

    tokens = rng.integers(0, vocab_size, size=(n_samples, seq_len))
    inputs[:, :seq_len] = tokens
    # After the delay, we expect the model to reproduce the tokens
    targets[:, seq_len + delay :] = tokens

    return torch.from_numpy(inputs), torch.from_numpy(targets)


# ---------------------------------------------------------------------------
# Tiny Shakespeare (character-level)
# ---------------------------------------------------------------------------

_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_tiny_shakespeare(max_chars: int | None = None, data_dir: str = "data") -> str:
    """Load Tiny Shakespeare text, downloading if necessary."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tiny_shakespeare.txt")
    if not os.path.exists(path):
        import urllib.request
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(_SHAKESPEARE_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if max_chars is not None:
        text = text[:max_chars]
    return text


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices) -> str:
        return "".join(self.idx_to_char[int(i)] for i in indices)


class SequenceDataset(Dataset):
    """PyTorch dataset that chunks a token sequence into fixed-length windows."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y
