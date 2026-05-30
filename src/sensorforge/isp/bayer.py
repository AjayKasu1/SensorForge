"""Bayer color-filter-array mosaic (RGGB).

Collapses a full RGB image into the single-channel raw a real sensor produces:
each photosite sits behind one color filter and sees only that channel. Runs
right after the optics and before the noise stages (ADR 004), so shot/read/dark
noise act on this raw, as on real hardware.

    RGGB layout within every 2x2 block:
        (0,0)=R  (0,1)=G
        (1,0)=G  (1,1)=B

Bayer (1976), US Patent 3,971,065.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# [row % 2][col % 2] -> RGB channel index (0=R, 1=G, 2=B) for the RGGB pattern.
BAYER_RGGB_CHANNELS = ((0, 1), (1, 2))


def mosaic_rggb(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """RGB (H, W, 3) -> single-channel raw (H, W) under the RGGB pattern.

    Assumes even H and W so the 2x2 tile divides evenly (true for our 640x480
    webcam scenes); odd sizes just leave a partial final row/column.
    """
    h, w = rgb.shape[:2]
    raw = np.empty((h, w), dtype=rgb.dtype)
    raw[0::2, 0::2] = rgb[0::2, 0::2, 0]  # R
    raw[0::2, 1::2] = rgb[0::2, 1::2, 1]  # G (top-right)
    raw[1::2, 0::2] = rgb[1::2, 0::2, 1]  # G (bottom-left)
    raw[1::2, 1::2] = rgb[1::2, 1::2, 2]  # B
    return raw
