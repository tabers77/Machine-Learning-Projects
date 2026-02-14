# Nested Learning — PyTorch Implementation

Implementation of key components from **"Nested Learning: The Illusion of Deep Learning Architectures"** (Ali Behrouz et al., NeurIPS 2025).

## Core Ideas

The paper argues that backprop, attention, and momentum are all **associative memory** modules operating at different timescales. It proposes:

1. **Associative Memory Unification** — Attention is mathematically equivalent to modern Hopfield retrieval; backprop weight updates are Hebbian outer products; momentum is exponentially-weighted gradient memory.

2. **Deep Momentum Gradient Descent (DMGD)** — Replace linear momentum with a learned MLP that models non-linear loss landscape dynamics.

3. **Continuum Memory System (CMS)** — Memory blocks with geometrically increasing update frequencies (inspired by brain oscillations: gamma → theta → delta).

4. **Hope Architecture** — Integrates attention, CMS, and DMGD with surprise-based gating into a unified model.

## Project Structure

```
src/
  associative_memory.py  — Hopfield networks, attention-as-memory, backprop/momentum views
  dmgd.py                — Deep Momentum GD optimizer with meta-learning
  cms.py                 — Continuum Memory System with multi-frequency blocks
  hope.py                — Hope architecture (TitanBlock + HopeBlock + full model)
  data.py                — Toy dataset generators
  utils.py               — Visualization & training helpers

notebooks/
  01_associative_memory.ipynb  — Pattern storage, capacity, attention=Hopfield proof
  02_dmgd.ipynb                — 2D loss surfaces, meta-learning, MNIST comparison
  03_cms.ipynb                 — Signal decomposition, update timelines, ablations
  04_hope.ipynb                — Full model on Tiny Shakespeare, surprise visualization
  05_comparisons.ipynb         — Side-by-side summary of all experiments

tests/                         — pytest unit tests for all modules
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Notebooks

```bash
cd notebooks
jupyter notebook
```

## References

- [Paper (arXiv)](https://arxiv.org/abs/2512.24695)
- [Paper (PDF)](https://abehrouz.github.io/files/NL.pdf)
