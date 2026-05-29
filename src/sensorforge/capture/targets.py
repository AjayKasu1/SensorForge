"""Calibration target generators.

Targets are intensity patterns returned as float32 arrays in [0, 1], the same
convention the renderer and ISP use (linear intensity, not 8-bit). The 8-bit
encoding is deferred to an explicit ``save_png`` so we keep one numeric
convention across the codebase.

Patterns are single-channel (H, W). A checkerboard is achromatic and the
uniform field is gray, so there is nothing to gain from carrying three
identical channels around; callers that need RGB can broadcast.
"""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def checkerboard(
    inner_corners: tuple[int, int] = (9, 6),
    square_px: int = 80,
    light: float = 1.0,
    dark: float = 0.0,
) -> NDArray[np.float32]:
    """Generate a checkerboard target.

    Parameters
    ----------
    inner_corners
        (cols, rows) of *internal* corners, the quantity OpenCV's
        ``findChessboardCorners`` expects. The board has one more square than
        corners along each axis, so (9, 6) yields a 10x7 square grid.
    square_px
        Side length of each square in pixels.
    light, dark
        Intensities of the two square colors, in [0, 1].
    """
    cols_sq = inner_corners[0] + 1
    rows_sq = inner_corners[1] + 1

    # (row + col) parity gives the classic alternating pattern; 0 -> light.
    parity = (np.indices((rows_sq, cols_sq)).sum(axis=0) % 2).astype(np.float32)
    pattern = np.where(parity == 0, light, dark).astype(np.float32)

    # Upscale each cell to square_px without interpolation (Kronecker product
    # with a block of ones is exact, no edge blurring).
    block = np.ones((square_px, square_px), dtype=np.float32)
    return np.kron(pattern, block).astype(np.float32)


def uniform_field(
    height: int = 480,
    width: int = 640,
    level: float = 0.5,
) -> NDArray[np.float32]:
    """Flat gray field at a constant intensity ``level`` in [0, 1].

    Used for flat-field response, AWB, and the EMVA-1288 uniform-illumination
    measurements in Phase 3.
    """
    return np.full((height, width), float(level), dtype=np.float32)


def save_png(target: NDArray[np.float32], path: str | Path) -> None:
    """Write a [0, 1] target to an 8-bit PNG for printing.

    Rounds rather than truncates; cv2 handles single-channel arrays as
    grayscale directly.
    """
    arr8 = np.clip(target * 255.0 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), arr8)
