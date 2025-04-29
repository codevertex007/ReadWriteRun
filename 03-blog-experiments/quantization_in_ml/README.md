# Quantization in Machine Learning

This repository explores practical and theoretical aspects of **Quantization** in machine learning models, especially for neural networks. Quantization helps in reducing model size, improving inference speed, and lowering memory usage — critical for deploying models on edge devices or at scale.

The project is organized into 4 mini-projects
---

## Mini-Project 1: NumPy Affine Quantizer

**Goal:** Implement uniform affine quantization using pure NumPy.

- Developed `quantize` and `dequantize` functions.
- Measured and plotted maximum quantization error versus bit-width.
- Focused on learning scale, zero-point calculation, and error behavior.

**Folder:** `quant-mini1/`

---

## Mini-Project 2: Post-Training Quantization (PTQ) on MLP (MNIST)

**Goal:** Train a small MLP in FP32, apply dynamic post-training quantization, and analyze accuracy.

- Trained a 2-hidden-layer MLP model on MNIST.
- Saved the trained FP32 model and evaluated its test accuracy.
- Applied dynamic PTQ to compress the model.
- Compared FP32 vs. quantized model accuracy.


**Folder:** `quant-mini2/`

---

## Mini-Project 3: Quantization-Aware Training (QAT) *(Upcoming)*

**Goal:** Introduce quantization simulation during training to minimize post-quantization accuracy loss.

- Replace standard Linear layers with fake-quantized versions.
- Train model while simulating quantization effects.
- Fine-tune a model to achieve better quantized accuracy than PTQ.

**Folder:** `quant-mini3/`

---

## Mini-Project 4: 4-bit and 2-bit Quantization Experiments *(Upcoming)*

**Goal:** Push quantization to more extreme low-bit settings (4 bits, 2 bits) and study:

- Accuracy degradation trends.
- Whether QAT helps more at lower bit-widths.
- Practical deployment limits.

**Folder:** `quant-mini4/`

---

# Goals of the Full Project

- Build practical intuition on quantization step-by-step.
- Apply both theoretical and PyTorch-native quantization techniques.


# License

This project is open-sourced under the MIT License.
