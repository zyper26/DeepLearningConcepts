# Deep Learning From Scratch

Python implementations of core deep learning concepts built with NumPy and PyTorch — no high-level wrappers, just the math.

## Files

### `adam_optimizer.py`
Implements **Adam** and **AdamW** optimizers from scratch using NumPy.

- Adam: adaptive learning rates via 1st moment (momentum) and 2nd moment (RMS) of gradients
- AdamW: decoupled weight decay — the production-standard variant
- Bias-correction for both moments (`m̂`, `v̂`)

Key formulas:
```
m = β1·m + (1-β1)·g          ← velocity
v = β2·v + (1-β2)·g²         ← magnitude
θ = θ - lr · m̂ / (√v̂ + ε)
```

---

### `batch_normalization.py`
Implements **Batch Normalization** (Ioffe & Szegedy, 2015) using NumPy.

- Normalizes activations across the batch dimension during training
- Maintains running mean/variance for inference (no batch stats needed)
- Learnable scale (`γ`) and shift (`β`) parameters
- Momentum-based running stat updates

---

### `infoNCE.py`
Implements the **InfoNCE loss** used in contrastive self-supervised learning (e.g., SimCLR, CLIP).

- Cosine similarity between query and positive/negative embeddings
- Temperature-scaled softmax over negatives
- Low loss when positive is similar to query, high loss when indistinguishable from negatives

```
L = -log( exp(sim(q, p⁺)/τ) / Σ exp(sim(q, pᵢ)/τ) )
```

---

### `knowledge_distillation.py`
Implements **Knowledge Distillation** (Hinton et al., 2015) with PyTorch — compressing XLM-RoBERTa (560M params) into a BiLSTM (2.5M params).

Three components:
1. **Output distillation** — soft-label KL divergence with temperature scaling (`T²` gradient compensation)
2. **Feature distillation** — MSE between teacher and student intermediate representations
3. **Combined training loop** — joint hard-label CE + soft-label KL loss

Interview notes embedded in docstrings (temperature scaling, KLDiv input format, BiLSTM hidden state indexing).

---

### `loss_function.py`
Common loss functions implemented from scratch with NumPy.

| Function | Use case |
|---|---|
| `cross_entropy_loss` | Multi-class classification |
| `binary_cross_entropy` | Binary classification |
| `mse_loss` | Regression |

Each function includes numerical stability fixes (log-sum-exp trick for cross-entropy).

---

### `regularization.py`
Regularization techniques to prevent overfitting, implemented with NumPy.

- **L1 regularization** — sparsity-inducing penalty (`λ·|w|`), gradient = `λ·sign(w)`
- **L2 regularization** — weight decay penalty (`λ·w²`), gradient = `2λ·w`
- **Dropout** — randomly zeros neurons during training with inverted scaling (`1/(1-p)`) to keep expected output unchanged

---

## Running the scripts

```bash
python adam_optimizer.py
python batch_normalization.py
python infoNCE.py
python loss_function.py
python regularization.py
python knowledge_distillation.py   # requires torch
```

Each script prints test cases with expected vs actual output.

## Dependencies

```bash
pip install numpy torch
```
