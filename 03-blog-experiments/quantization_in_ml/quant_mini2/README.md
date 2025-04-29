```
quant_mini2/
├── train_mlp.py
├── ptq_quantize.py
├── notebooks/
│   └── training_and_ptq.ipynb
├── tests/
│   └── test_ptq.py
└── README.md
```

# Mini-Project 2: Post-Training Quantization (PTQ) on a Small MLP (MNIST)

## Overview
Quantize a fully-connected neural network’s **weights** after training on MNIST, then measure accuracy drop.

### Goals
1. Train a simple MLP (two hidden layers) on MNIST in FP32.
2. Apply PTQ to its weight tensors at 4 bits.
3. Replace weights in the saved model and evaluate test accuracy.
4. Compare FP32 vs. quantized accuracy and log results.

## Folder Structure

- `train_mlp.py` — Script to define and train the MLP (PyTorch), save the FP32 model.
- `ptq_quantize.py` — Script to:
  1. Load the saved FP32 model.
  2. Apply affine quantization to each weight tensor.
  3. Replace model weights and run inference on the test set.
- `notebooks/training_and_ptq.ipynb` — Notebook walkthrough: training curves, PTQ steps, accuracy comparison.
- `tests/test_ptq.py` — Unit tests for:
  - Model loading and weight shapes.
  - Ensuring quantized weights lie in the proper integer range.
  - Checking accuracy drops by at most a given margin (e.g. <5%).
- `README.md` — This file, with instructions and examples.

## Installation
```bash
pip install torch torchvision numpy pytest
```

## Usage

### 1. Train the MLP (FP32)
```bash
python train_mlp.py --epochs 5 --batch-size 128 --save-path model_fp32.pth
```

### 2. Apply PTQ and Evaluate
```bash
python ptq_quantize.py --model model_fp32.pth --bits 4
```
This script will print test accuracy before and after quantization.

### 3. Notebook
Open `notebooks/training_and_ptq.ipynb` for a guided analysis of:
- Training loss & accuracy curves
- PTQ quantization errors
- Accuracy comparison table

## Running Tests
```bash
pytest tests/test_ptq.py
```

