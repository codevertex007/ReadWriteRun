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
2. Apply PTQ using PyTorch.
3. Compare FP32 vs. quantized accuracy.

## Folder Structure

- `train_mlp.py` — Script to define and train the MLP (PyTorch), save the FP32 model.
- `ptq_quantize.py` — Script to:
  1. Load the saved FP32 model.
  2. Apply quantization to each weight tensor.
  3. Then run inference on the test set.
- `README.md` — This file, with instructions and examples.

## Installation
```bash
pip install torch torchvision numpy pytest
```

## Usage

### 1. Train the MLP (FP32)
```bash
python train_mlp.py --epochs 15 --batch-size 128 --save-path model_fp32.pth
```

### 2. Apply PTQ and Evaluate
```bash
python ptq_quantize.py --model-path ../../saved_models/mnist_model_fp32.pth --batch-size 128
```
This script will print test accuracy before and after quantization.




