
# Arithmetic-Transformer

**A transformer-based approach to solving arithmetic expressions with strong generalization on edge cases (carrying, large numbers).**

---

## Overview

This project implements a sequence-to-sequence Transformer that learns to evaluate arithmetic expressions (e.g., `12+345 = 357`).  
The goal is not only to fit training data but to **generalize to harder edge-cases** such as long numbers, carrying across many digits, and unseen operand lengths. You will find experiment code, training loops, evaluation metrics and ablation studies inside `code.ipynb`.

---

## Highlights

- Synthetic dataset generator for arithmetic expressions with configurable operand lengths and operators.  
- Transformer-based encoder-decoder model tailored for exact-match sequence prediction.  
- Ablation studies exploring depth, model size, attention heads, and dropout to find a good trade-off between accuracy and compute.  
- Reproducible training recipe (seeds, deterministic settings, and GPU usage suggestions).  
- Optional use of parameter-efficient fine-tuning (LoRA) for faster experimentation on larger pre-trained models (if used).

---


## Example usage (in Python)

Below is a minimal example to run inference using a trained seq2seq transformer checkpoint (pseudocode). See notebook for actual API details.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("path_or_model_name")
model = AutoModelForSeq2SeqLM.from_pretrained("checkpoints/transformer-best")

expr = "12345+67890"
inp = tokenizer(expr, return_tensors="pt")
with torch.no_grad():
    output_ids = model.generate(**inp, max_new_tokens=20)
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"{expr} = {prediction}")
```

---

## Dataset

The notebook contains a configurable synthetic dataset generator. Typical parameters you can (and should) vary:

- Operators: `+`, `-`, `*` (or a subset)
- Operand length range: e.g. `1..N` digits per operand
- Number of operands per sample
- Train / validation / test splits (including out-of-distribution test sets for generalization)

**Tip:** Create an `o.o.d_test` set of expressions with longer lengths than the train set to measure generalization.

---

## Model & Training

- **Architecture:** Standard Transformer encoder-decoder (custom small Transformer in notebook) or a pre-trained seq2seq adapted for arithmetic.
- **Loss:** Cross-entropy token-level loss; evaluation uses exact-match (sequence-level) accuracy.
- **Optimization:** AdamW or Adam with learning rate scheduling and gradient clipping.
- **Checkpointing:** Save best weights via validation exact-match.

**Suggested hyperparameter table** (fill after experiments):

| Param            | Value (example) |
|------------------|-----------------|
| d_model          | 256             |
| n_layers         | 4               |
| n_heads          | 8               |
| ff_dim           | 1024            |
| dropout          | 0.1             |
| batch_size       | 64              |
| epochs           | 30              |
| learning_rate    | 3e-4            |

---

## Ablation studies

The notebook includes ablations on:
- Model depth vs. generalization
- Number of attention heads
- Hidden dimension scaling (d_model)
- Dropout and regularization

Record outcomes in a table and plot trends (validation exact-match vs. compute / params) to justify the final model selection.

---

## Evaluation & Metrics

Primary metrics to report:

- **Exact match (EM)**: percentage of expressions where predicted sequence exactly equals ground truth.
- **Token-level accuracy**: percentage of correctly predicted tokens.
- **Mean Absolute Error (MAE)** or **MSE** on numeric outputs (if you decode predictions to integers).
- **Generalization tests**: EM on OOD sets (larger operands, more carries).

Add a clear results section in the notebook and in this README after running experiments.






## Contact

Ayush Sheta — ayush.sheta@students.iiit.ac.in  
GitHub: https://github.com/Ayushsheta2005

---

## License

This repository is released under the **MIT License**. See `LICENSE` for details.
