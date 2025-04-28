```
quant-mini1/
├── quantize.py
├── notebooks/
│   └── error_vs_bits.ipynb
├── tests/
│   └── test_quantize.py
└── README.md
```

# Mini-Project 1: NumPy Affine Quantizer

## Overview
This project implements an **affine quantization** scheme for NumPy arrays.

- **Quantize:** Convert FP32 arrays to integer representations using `scale` and `zero_point`.
- **Dequantize:** Recover approximate FP32 values from quantized integers.
- **Error Analysis:** Measure and plot max absolute error versus bit-width.

## Folder Structure

- `quantize.py` — Core implementation of `quantize` and `dequantize` functions.
- `notebooks/error_vs_bits.ipynb` — Jupyter notebook demonstrating error analysis for bit-widths [2, 4, 8].
- `tests/test_quantize.py`
- `README.md` — Project description, usage, and examples.

## Getting Started

### Prerequisites
- Python 3.11+
- NumPy
- Matplotlib (for plotting)
- pytest (for tests)


### Usage
```python
from quantize import quantize, dequantize
import numpy as np

# Example
x = np.array([0.1, -0.5, 0.7], dtype=np.float32)
q, scale, zp = quantize(x, bits=4)
x_recovered = dequantize(q, scale, zp)
print("Original:", x)
print("Quantized:", q)
print("Dequantized:", x_recovered)
```

## Running Tests
```bash
pytest tests/test_quantize.py
```

## Notebook
Open `notebooks/error_vs_bits.ipynb` to see plots of max error vs. bit-width.

