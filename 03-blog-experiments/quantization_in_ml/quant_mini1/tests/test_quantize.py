import numpy as np
import pytest
from quant_mini1.quantize import quantize, dequantize


# Test that constant arrays recover perfectly after dequantization
def test_constant_array():
    x = np.full((10,), 3.14, dtype=np.float32)
    q, m, z = quantize(x, bits=4)
    x_deq = dequantize(q, m, z)
    assert np.allclose(x_deq, x, atol=1e-6)


# Test that quantized integers lie within the correct signed range
def test_quantized_range():
    x = np.linspace(-5, 5, 5).astype(np.float32)
    bits = 3
    q, m, z = quantize(x, bits=bits)
    assert q.min() >= -2**(bits-1)
    assert q.max() <= 2**(bits-1) - 1

# Test that increasing bit-width reduces max error
def test_random_array_error_decreases():
    x = np.random.RandomState(0).randn(1000).astype(np.float32)
    q4, m4, z4 = quantize(x, bits=4)
    x4 = dequantize(q4, m4, z4)
    q8, m8, z8 = quantize(x, bits=8)
    x8 = dequantize(q8, m8, z8)
    err4 = np.max(np.abs(x - x4))
    err8 = np.max(np.abs(x - x8))
    assert err8 < err4