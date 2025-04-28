import numpy as np


def quantize(x, bits=4):
    """
    Affine quantization to signed ints in [-2^(b-1), 2^(b-1)-1].
    Returns (q, scale, zero_point).
    """

    # 1) signed integer range
    qmin = -2 ** (bits-1)
    qmax = 2**(bits-1) - 1

    # 2) find float range
    xmax = x.max()
    xmin = x.min()

    if xmax == xmin:
        q = np.zeros_like(x, dtype=int)
        return q, 0.0, float(xmin)

    # 3) compute multiplier and zero_point
    m = (qmax - qmin)/ (xmax - xmin)
    z = qmin - round(xmin * m)

    q = np.round(m*x).astype(int) + z
    q = np.clip(q, qmin, qmax)
    return q, m, z


def dequantize(q, m, zp):
    """De-quantize value"""

    if m == 0:
        return np.full_like(q, float(zp), dtype=float)

    x = (q - zp) / m
    return x

