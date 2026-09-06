# Causal Transformer for Autoregressive Arithmetic Reasoning

A compact causal Transformer trained from scratch in PyTorch to generate step-by-step reductions of parenthesized addition expressions.

## Overview

This project explores autoregressive sequence modeling through a controlled arithmetic task. Given a parenthesized addition expression ending in `=`, the model generates the intermediate reduction steps and final answer one token at a time.

Example:

```text
Input:
(((1+2)+1)+8)=

Output:
(((1+2)+1)+8)=((3+1)+8)=(4+8)=12
```

Rather than using a pretrained language model, the system trains a small Transformer from scratch on synthetically generated arithmetic sequences. The project covers the full workflow from data generation and tokenization to causal language-model training, exact-match evaluation, model serialization, and standalone command-line inference.

## Key Results

The final model achieved **100% exact-sequence accuracy** on the required benchmark configurations, using **1,000 independently generated examples per setting**.

| Number of Integers | Operand Digits | Exact-Match Accuracy |
| ---: | ---: | ---: |
| 2 | 1 | **100.00%** |
| 2 | 2 | **100.00%** |
| 2 | 3 | **100.00%** |
| 3 | 1 | **100.00%** |
| 3 | 2 | **100.00%** |
| 3 | 3 | **100.00%** |
| 4 | 1 | **100.00%** |
| 4 | 2 | **100.00%** |
| 5 | 1 | **100.00%** |
| 5 | 2 | **100.00%** |

Evaluation uses **exact string matching**. A prediction is counted as correct only when the entire generated reduction sequence matches the target sequence.

> These results measure performance on synthetic test cases generated under the same task rules as the training distribution. They should not be interpreted as evidence of general mathematical reasoning.

## Problem Formulation

The task is modeled as autoregressive next-token prediction.

A complete training sequence may look like:

```text
((1+4)+((2+2)+9))=(5+(4+9))=(5+13)=18
```

The model must learn to generate both:

- the correct intermediate expression reductions, and
- the final numerical result.

This provides a compact environment for studying causal attention, sequence generation, positional representations, padding, and autoregressive inference.

## Project Pipeline

```text
Synthetic Expression Generation
            |
            v
 Character-Level Tokenization
            |
            v
 Dynamic Padding / Batching
            |
            v
 Token + Positional Embeddings
            |
            v
 Causal Transformer
            |
            v
 Next-Token Cross-Entropy Training
            |
            v
 Curriculum-Based Optimization
            |
            v
 Exact-Match Benchmarking
            |
            v
 Saved Model Weights
            |
            v
 Standalone CPU Inference
```

## Synthetic Data Generation

Training examples are generated programmatically instead of being loaded from a fixed dataset.

Each generated example:

1. samples positive integers,
2. recursively combines adjacent operands with addition and parentheses,
3. evaluates reducible inner expressions,
4. records each intermediate reduction,
5. concatenates the stages with `=` symbols.

Generating examples online allows the model to train on a large variety of expressions without maintaining a large static dataset.

## Tokenization

The model uses a compact character-level vocabulary:

```text
<bos>
<eos>
<pad>
(
)
+
0 1 2 3 4 5 6 7 8 9
=
```

Total vocabulary size: **17 tokens**.

Each expression is encoded as token IDs and prepended with `<bos>`. During training, `<eos>` marks the end of the target sequence. Variable-length sequences are right-padded with `<pad>`.

## Model Architecture

The model uses PyTorch Transformer encoder blocks together with an explicit causal attention mask, making the network behave as a decoder-style autoregressive Transformer.

### Configuration

| Component | Value |
| --- | ---: |
| Model dimension (`d_model`) | **192** |
| Attention heads | **6** |
| Transformer layers | **6** |
| Feed-forward dimension | **384** |
| Maximum sequence length | **128** |
| Dropout | **0.1** |
| Vocabulary size | **17** |

Architecture:

```text
Token IDs
   |
   +--> Token Embeddings
   |
   +--> Learned Positional Embeddings
   |
   v
6 x Transformer Self-Attention Blocks
   |
   v
Linear Language-Model Head
   |
   v
Next-Token Logits
```

## Causal Attention

An upper-triangular causal mask prevents each position from attending to future tokens.

For token position `t`:

```text
visible:  positions <= t
masked:   positions > t
```

A separate key-padding mask prevents `<pad>` positions from contributing to attention.

Although the implementation uses `torch.nn.TransformerEncoder`, the causal mask makes the computation autoregressive and decoder-style.

## Training Objective

The model is trained using standard next-token prediction.

For a tokenized sequence:

```text
x1, x2, x3, ..., xT
```

the input is:

```text
x1, x2, ..., x(T-1)
```

and the target is:

```text
x2, x3, ..., xT
```

Padding tokens are excluded from the loss:

```python
torch.nn.CrossEntropyLoss(ignore_index=PAD)
```

## Curriculum-Style Training

Training difficulty increases progressively across three phases.

| Phase | Number of Integers | Operand Digits | Steps |
| --- | --- | --- | ---: |
| 1 | 2–3 | 1 digit | **3,000** |
| 2 | 2–4 | 1–2 digits | **6,000** |
| 3 | 2–5 | 1–3 digits | **9,000** |

Total optimization steps: **18,000**

Training batch size: **1,024 synthetic examples**

This curriculum gradually increases both expression length and operand magnitude.

## Optimization

```text
Optimizer: AdamW
Learning rate: 5e-4
Loss: Cross-Entropy Loss
Batch size: 1,024
Training steps: 18,000
Maximum sequence length: 128
```

CUDA is used for training when available.

The recorded loss at the end of the final training phase was approximately **0.4309** for the sampled batch printed at step 18,000.

Because the primary task is deterministic sequence generation, exact-match benchmark accuracy is more meaningful than training loss alone.

## Autoregressive Inference

At inference time, the model receives an expression through the first `=` sign and repeatedly predicts the next token.

```text
Prompt
  |
  v
Transformer
  |
  v
Next-token logits
  |
  v
Argmax token
  |
  v
Append to sequence
  |
  +---- repeat until <eos> or max length
```

Generation uses greedy decoding:

```python
next_token = torch.argmax(logits, dim=-1)
```

Greedy decoding is suitable here because the arithmetic target is deterministic rather than open-ended.

## Repository Structure

```text
transformer-math-reasoning/
├── README.md
├── requirements.txt
├── predict.py
├── math_transformer.pt
│
├── examples/
│   └── input.txt
│
└── notebooks/
    └── training_and_evaluation.ipynb
```

### File roles

- `notebooks/training_and_evaluation.ipynb` — synthetic data generation, model definition, training, evaluation, and experimentation.
- `predict.py` — standalone command-line inference script.
- `math_transformer.pt` — trained model state dictionary.
- `examples/input.txt` — sample prompts for testing inference.
- `requirements.txt` — minimal Python dependency list.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/JasonHonghzj/transformer-math-reasoning.git
cd transformer-math-reasoning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run inference

```bash
python predict.py examples/input.txt
```

Example output:

```text
(((1+2)+1)+8)=((3+1)+8)=(4+8)=12
((12+35)+(41+22))=(47+63)=110
((7+8)+(9+6))=(15+15)=30
```

The included inference script runs on CPU by default, so a GPU is not required to test the pretrained model.

## Important Checkpoint Rename

The portfolio version of this repository uses:

```text
math_transformer.pt
```

instead of the original coursework filename:

```text
math.pt
```

Therefore, `predict.py` should load the renamed checkpoint with:

```python
state_dict = torch.load("math_transformer.pt", map_location=device)
```

## Benchmarking

Generalization is evaluated across combinations of:

- number of integers,
- operand digit length,
- expression complexity.

Each benchmark configuration contains **1,000 randomly generated examples**.

The evaluation criterion is:

```python
correct = generated_expression == target_expression
```

This is stricter than checking only the final arithmetic answer because the entire generated reduction trajectory must be correct.

## Tech Stack

- **Python**
- **PyTorch**
- **Transformer architecture**
- **Multi-head self-attention**
- **Causal masking**
- **Token and positional embeddings**
- **Autoregressive sequence modeling**
- **Synthetic data generation**
- **Dynamic sequence padding**
- **AdamW**
- **Cross-entropy loss**
- **CUDA training**
- **Greedy decoding**
- **Model serialization**
- **Command-line inference**

## What This Project Demonstrates

This project provides a compact implementation of several ideas used in modern autoregressive language models:

- discrete tokenization,
- learned embeddings,
- positional representations,
- multi-head self-attention,
- causal masking,
- padding-aware batched training,
- next-token prediction,
- autoregressive decoding,
- curriculum-style training,
- exact-match sequence evaluation,
- separation of training and inference workflows.

The task is intentionally constrained so that these mechanisms can be studied without relying on pretrained model weights or large external datasets.

## Limitations

This is a controlled synthetic arithmetic task rather than a general-purpose language model.

Key limitations include:

- a vocabulary of only 17 tokens,
- positive-integer addition only,
- synthetic training and evaluation data,
- a maximum sequence length of 128 tokens,
- benchmark cases generated under the same task rules as the training data,
- no natural-language mathematical reasoning,
- no systematic out-of-distribution evaluation.

The reported benchmark accuracy should therefore be interpreted as strong performance within the defined arithmetic-expression task.

## Potential Improvements

Future extensions could include:

- evaluating substantially longer expressions and larger operands,
- testing true out-of-distribution sequence lengths,
- adding subtraction, multiplication, and mixed operators,
- comparing curriculum learning with uniform sampling,
- reporting token-level accuracy alongside exact-match accuracy,
- visualizing attention patterns,
- comparing learned positional embeddings with sinusoidal or rotary embeddings,
- testing alternative decoding strategies,
- refactoring training logic into reusable Python modules,
- adding unit tests for tokenization, masking, generation, and CLI inference.

## Acknowledgements

This project was developed as part of **DS 542 coursework**.

The project framework and arithmetic-instance generator were provided as part of the course. The Transformer model configuration, causal sequence-modeling pipeline, curriculum-based training strategy, benchmarking workflow, model serialization, and standalone inference implementation were developed and extended for the final project.

---

This repository is presented as a machine-learning portfolio project demonstrating causal Transformer training, autoregressive sequence modeling, synthetic-data generation, and reproducible inference with PyTorch.
