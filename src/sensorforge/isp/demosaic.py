"""Bilinear demosaic for the RGGB pattern.

Reconstructs full RGB from the single-channel raw by linearly interpolating each
color's missing samples from its same-color neighbors. Bilinear is the simplest
demosaic and visibly produces zipper/false-color artifacts on high-frequency
edges, which is the point: it gives the calibration a real spatial-artifact
signature to match. Malvar et al. (2004) is the obvious upgrade if v1 can't hit
the color target; deferred.

We convolve sparse per-channel planes with the classic bilinear kernels rather
than calling cv2's Bayer cvtColor, so the interpolation is explicit and matches
our mosaic_rggb layout exactly.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

# Green is known on a checkerboard of pixels: average the 4 orthogonal neighbors.
_G_KERNEL = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float64) / 4.0
# Red/Blue are known on a quarter grid: orthogonal + diagonal bilinear weights.
_RB_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 4.0


def demosaic_bilinear(raw: NDArray) -> NDArray:
    """RGGB raw (H, W) -> RGB (H, W, 3) by bilinear interpolation."""
    r = np.zeros_like(raw)
    g = np.zeros_like(raw)
    b = np.zeros_like(raw)
    r[0::2, 0::2] = raw[0::2, 0::2]
    g[0::2, 1::2] = raw[0::2, 1::2]
    g[1::2, 0::2] = raw[1::2, 0::2]
    b[1::2, 1::2] = raw[1::2, 1::2]

    rf = cv2.filter2D(r, -1, _RB_KERNEL, borderType=cv2.BORDER_REFLECT)
    gf = cv2.filter2D(g, -1, _G_KERNEL, borderType=cv2.BORDER_REFLECT)
    bf = cv2.filter2D(b, -1, _RB_KERNEL, borderType=cv2.BORDER_REFLECT)
    return np.stack([rf, gf, bf], axis=-1).astype(raw.dtype)
