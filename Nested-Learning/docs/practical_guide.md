# Practical Guide: Applying Nested Learning Findings to Model Training

> Based on the research findings from the [Nested Learning experiment](../README.md#research-question), this guide translates the results into actionable steps for data scientists training supervised models.

---

## Why This Matters

The CMS (Continuum Memory System) experiment showed that **simply making different parts of a network update at different speeds** reduces catastrophic forgetting by 14% — with almost zero cost to learning ability. This principle applies broadly to everyday model training, fine-tuning, and production retraining workflows.

---

## 1. Always Use Layerwise Learning Rates

**When**: Any time you're training a model with more than one layer group.

**Why it works**: The CMS finding shows that early layers capture general, reusable features and benefit from slower updates, while later layers need to adapt quickly to the current task. Applying different learning rates across layers gives you stability for free (Finding 3: plasticity drops less than 1%).

### Steps

1. Split your model into layer groups (early, middle, head)
2. Assign decreasing LRs to earlier layers (10x slower per group is a good starting point)
3. Pass parameter groups to your optimizer

### PyTorch Example

```python
# Step 1: Define your model normally
model = torchvision.models.resnet18(pretrained=True)

# Step 2-3: Group parameters with different LRs
optimizer = torch.optim.Adam([
    {"params": model.layer1.parameters(), "lr": 1e-5},   # earliest — slowest
    {"params": model.layer2.parameters(), "lr": 1e-4},   # middle
    {"params": model.layer3.parameters(), "lr": 5e-4},   # middle-high
    {"params": model.layer4.parameters(), "lr": 1e-3},   # late — fast
    {"params": model.fc.parameters(),     "lr": 1e-3},   # head — fastest
])

# Then train normally — no other changes needed
for batch in dataloader:
    loss = criterion(model(batch["x"]), batch["y"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### sklearn Equivalent (Two-Stage Warm Start)

When using sklearn models without direct layer control, you can approximate the same idea with a two-stage approach:

```python
# Stage 1: Train full model on original data with low LR
mlp = MLPClassifier(learning_rate_init=0.0001, max_iter=200)
mlp.fit(X_train_v1, y_train_v1)

# Stage 2: Warm-start on new data (only later layers adapt significantly
# because earlier layers have already converged to stable features)
mlp.set_params(learning_rate_init=0.001, max_iter=50, warm_start=True)
mlp.fit(X_train_v2, y_train_v2)
```

---

## 2. When Fine-Tuning, Freeze Aggressively

**When**: You're taking a pretrained model (ResNet, BERT, etc.) and adapting it to your dataset.

**Why it works**: Freezing early layers is an extreme version of CMS (C_base = infinity for frozen layers). The research showed plasticity barely drops (98.1% vs 98.6%), meaning you lose almost nothing by keeping early layers fixed.

### Steps

1. Load pretrained model
2. Freeze everything
3. Unfreeze only the last 1-2 layers + head
4. Train. If accuracy is insufficient, unfreeze one more layer and repeat

### PyTorch Example

```python
# Step 1: Load pretrained
model = torchvision.models.resnet18(pretrained=True)

# Step 2: Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Step 3: Unfreeze only the head + last block
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Step 4: Train only unfrozen params
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

for batch in dataloader:
    loss = criterion(model(batch["x"]), batch["y"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Step 5: If accuracy is too low, unfreeze one more layer and retrain
# for param in model.layer3.parameters():
#     param.requires_grad = True
# ...retrain with a smaller LR for layer3
```

### How Much to Unfreeze (Rule of Thumb)

| Your dataset size | What to unfreeze |
|---|---|
| < 1,000 samples | Head only |
| 1,000 – 10,000 | Head + last block |
| 10,000 – 100,000 | Head + last 2 blocks (with LR decay) |
| > 100,000 | All layers (but still use layerwise LRs from point 1) |

---

## 3. When Models Degrade After Retraining, Try Simple Fixes First

**When**: You retrain a production model on new data and notice performance drops on old categories/classes.

**Why it works**: The research showed EWC (a complex regularization method) gave less than 1% improvement, while the simple architectural change of multi-timescale updates gave 14%. Simple fixes should be your first resort, not your last.

### The Fix Ladder (Stop as Soon as the Problem is Fixed)

```
Fix 1 (simplest): Partial freezing
         ↓ didn't work?
Fix 2: Layerwise LR decay
         ↓ didn't work?
Fix 3: Mix old + new data (replay)
         ↓ didn't work?
Fix 4: EWC or other complex methods
```

### Fix 1 — Partial Freezing (5 min to implement)

```python
# Before retraining on new data, freeze early layers
for name, param in model.named_parameters():
    if "layer1" in name or "layer2" in name:
        param.requires_grad = False

# Retrain on new data
train(model, new_data)
```

### Fix 2 — Layerwise LR Decay (5 min to implement)

```python
# Give early layers 10-100x smaller LR
optimizer = torch.optim.Adam([
    {"params": model.layer1.parameters(), "lr": 1e-6},   # barely moves
    {"params": model.layer2.parameters(), "lr": 1e-5},
    {"params": model.layer3.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(),     "lr": 1e-3},   # adapts freely
])
train(model, new_data)
```

### Fix 3 — Simple Replay (30 min to implement)

```python
# Keep a small buffer of old data (5-10% of original dataset)
old_buffer = random_sample(old_training_data, frac=0.1)

# Mix it with new data during retraining
combined = ConcatDataset([new_data, old_buffer])
train(model, combined)
```

### Fix 4 — EWC (hours to implement, rarely needed)

Only reach for this if the 3 simple fixes above all failed. The research showed EWC gave < 1% improvement on the benchmark while partial freezing gave 14%.

---

## 4. Monitor Your Most Vulnerable Classes

**When**: Any production model that gets retrained over time.

**Why it works**: Finding 2 showed that tasks most prone to forgetting benefit the most from timescale separation. In the experiment, the 6/7 digit pair (most forgetting) saw a 6x improvement with CMS. Identifying which classes degrade the most tells you exactly where to apply protection.

### Steps

1. After each retraining, compute per-class metrics (not just overall accuracy)
2. Track the delta per class between model versions
3. Flag classes that dropped the most — those are your vulnerable ones
4. Apply heavier protection (freezing, replay) specifically for those classes

### Per-Class Monitoring Script

```python
from sklearn.metrics import classification_report
import pandas as pd

def compare_model_versions(model_old, model_new, X_test, y_test):
    """Run after every retraining to detect vulnerable classes."""

    # Step 1: Per-class metrics for both versions
    pred_old = model_old.predict(X_test)
    pred_new = model_new.predict(X_test)

    report_old = classification_report(y_test, pred_old, output_dict=True)
    report_new = classification_report(y_test, pred_new, output_dict=True)

    # Step 2: Compute per-class delta
    results = []
    for cls in sorted(set(y_test)):
        cls_str = str(cls)
        old_f1 = report_old[cls_str]["f1-score"]
        new_f1 = report_new[cls_str]["f1-score"]
        delta = new_f1 - old_f1
        results.append({
            "class": cls,
            "old_f1": round(old_f1, 3),
            "new_f1": round(new_f1, 3),
            "delta": round(delta, 3),
        })

    df = pd.DataFrame(results).sort_values("delta")
    print(df.to_string(index=False))

    # Step 3: Flag vulnerable classes (dropped more than 5%)
    vulnerable = df[df["delta"] < -0.05]["class"].tolist()
    if vulnerable:
        print(f"\nVULNERABLE CLASSES: {vulnerable}")
        print("-> Add more replay samples for these classes")
        print("-> Or freeze early layers before next retraining")

    return df
```

**Example output**:

```
 class  old_f1  new_f1  delta
     6   0.920   0.780 -0.140   <-- vulnerable
     7   0.880   0.760 -0.120   <-- vulnerable
     3   0.950   0.930 -0.020
     1   0.970   0.980  0.010
```

### Targeted Protection for Vulnerable Classes

```python
# Over-sample vulnerable classes in your replay buffer
vulnerable_classes = [6, 7]
buffer_vulnerable = old_data[old_data["label"].isin(vulnerable_classes)].sample(500)
buffer_other = old_data[~old_data["label"].isin(vulnerable_classes)].sample(200)
replay_buffer = pd.concat([buffer_vulnerable, buffer_other])

# Retrain with this targeted replay buffer
combined = ConcatDataset([new_data, replay_buffer])
train(model, combined)
```

---

## Summary Cheat Sheet

| Point | What to do | Time to implement | When |
|---|---|---|---|
| **1. Layerwise LRs** | Pass param groups to optimizer with 10x decay per group | 5 min | Every training run |
| **2. Freeze aggressively** | `requires_grad = False` on early layers | 5 min | Every fine-tuning job |
| **3. Simple fixes first** | Freeze → LR decay → replay → EWC (in that order) | 5–30 min | When model degrades after retraining |
| **4. Monitor per-class** | `classification_report` + track deltas between versions | 30 min | Every production retraining cycle |

---

## Connection to the Research

Each practical point maps directly to a finding from the CMS experiment:

| Practical Point | Research Finding |
|---|---|
| Layerwise LRs | Finding 1: More timescale separation = less forgetting (monotonically) |
| Freeze aggressively | Finding 3: Plasticity barely drops — you get stability for free |
| Simple fixes first | Finding 4: EWC provides essentially no benefit; architecture-level changes beat algorithmic complexity |
| Monitor vulnerable classes | Finding 2: Tasks most prone to forgetting benefit dramatically more from CMS |

For the full experiment details, methodology, and results tables, see the [main README](../README.md).
