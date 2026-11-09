# LoRA (Low-Rank Adaptation) on FashionMNIST

This experiment demonstrates the implementation and fine-tuning of **Low-Rank Adaptation (LoRA)** on a fully connected neural network trained on the **FashionMNIST** dataset using PyTorch. This work was inspired from the original paper "LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS".

## Table of Contents
- [Overview](#overview)
- [Network Architecture](#network-architecture)
- [LoRA Implementation](#lora-implementation)
- [Experimental Procedure](#experimental-procedure)
- [Results & Performance](#results--performance)
- [How to Run](#how-to-run)

## Overview
LoRA is a parameter-efficient fine-tuning (PEFT) technique that freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer or dense architecture. This drastically reduces the number of trainable parameters for downstream tasks while maintaining or improving performance.

In this project, we implement LoRA parameterization from scratch utilizing PyTorch's `torch.nn.utils.parametrize` module to dynamically inject adapters into linear layers.

## Network Architecture
The baseline model (`RandomNet`) is a multi-layer perceptron (MLP) consisting of:
- **Input Layer:** Flat $28 \times 28$ image vector ($784$ units)
- **Hidden Layer 1:** $1000$ units with ReLU activation
- **Hidden Layer 2:** $2000$ units with ReLU activation
- **Output Layer:** $10$ units (representing FashionMNIST classes)

## LoRA Implementation
Our LoRA layer is parameterized as follows:
$$\Delta W = B \times A \times \frac{\alpha}{r}$$
Where:
- $A \in \mathbb{R}^{r \times d_{out}}$ is initialized from a normal distribution.
- $B \in \mathbb{R}^{d_{in} \times r}$ is initialized to zero, ensuring $\Delta W = 0$ at the start of training.
- $r$ (rank) = 1
- $\alpha$ (scaling factor) = 1

By leveraging PyTorch parametrizations, the forward pass dynamically computes weight matrices as:
$$W_{updated} = W_{original} + \Delta W$$

## Experimental Procedure
1. **Base Training:** The MLP model is trained for 1 epoch on the FashionMNIST training dataset (Base Accuracy: **86%**).
2. **Freeze & Parameterize:** All original weights are frozen (`requires_grad = False`). LoRA adapters ($r=1$) are registered to `linear1`, `linear2`, and `linear3` weight parameters.
3. **LoRA Fine-Tuning:** The network is fine-tuned for 1 epoch with **only** the LoRA weights ($A$ and $B$) configured to update.
4. **Evaluation:** Performance is evaluated with LoRA enabled and compared against the baseline with LoRA disabled.

## Results & Performance

| Configuration | Accuracy | Trainable Parameters |
|---|---|---|
| **Baseline (Base Training)** | 86.0% | ~2,806,000 (all parameters) |
| **With LoRA Enabled** | **87.9%** | **5,796** (0.2% of total) |
| **With LoRA Disabled** | 86.4% | 0 (fully frozen) |

### Observations
- Applying LoRA with a rank of $r=1$ introduced only **5,796** trainable parameters during fine-tuning.
- Fine-tuning the adapter for just 1 epoch increased model accuracy from **86.0%** to **87.9%**.
- Disabling LoRA reverts the network close to its base performance, demonstrating that the adaptation is successfully contained within the low-rank matrices.
