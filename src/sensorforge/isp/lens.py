"""Optical stages: vignetting and geometric distortion.

Both run on the linear RGB image before the Bayer mosaic (ADR 004).

Coordinate convention: image coordinates are centered on the optical center
(image center) and normalized by the half-diagonal, so the corner sits at
radius r = 1. Distortion coefficients are therefore in *corner-normalized*
units, NOT OpenCV's focal-normalized convention. They are not interchangeable
with cv2.calibrateCamera outputs; this keeps the stage self-contained (no
intrinsics needed) at the cost of that compatibility.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def _centered_normalized_grid(h: int, w: int) -> tuple[NDArray, NDArray, float]:
    """Return centered pixel-offset grids (x, y) and the half-diagonal norm."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    norm = float(np.hypot(cx, cy))  # corner -> r = 1
    return (xx - cx) / norm, (yy - cy) / norm, norm


def apply_vignetting(rgb: NDArray[np.float32], strength: float) -> NDArray[np.float32]:
    """Multiplicative cos^4 illumination falloff.

    Using the identity cos^4(theta) = (1 + r^2)^-2 when r = tan(theta), with r
    the corner-normalized image radius (so the corner is treated as a 45 deg
    field angle). ``strength`` blends between no vignette (0) and full cos^4
    darkening (1); it absorbs the lens-specific deviation from the 45 deg
    assumption.

    Cosine-fourth law: Kingslake, *Optics in Photography* (1992).
    """
    x, y, _ = _centered_normalized_grid(*rgb.shape[:2])
    r2 = x * x + y * y
    cos4 = 1.0 / (1.0 + r2) ** 2
    gain = 1.0 - strength * (1.0 - cos4)
    return (rgb * gain[..., None]).astype(np.float32)


def apply_distortion(
    rgb: NDArray[np.float32],
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
) -> NDArray[np.float32]:
    """Brown-Conrady radial + tangential distortion.

    For each output pixel (treated as an ideal/undistorted location) we compute
    the distorted source coordinate and sample the input there, which bends
    straight lines as a real lens does. Brown (1966), *Decentering Distortion
    of Lenses*.
    """
    h, w = rgb.shape[:2]
    x, y, norm = _centered_normalized_grid(h, w)
    r2 = x * x + y * y

    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    map_x = (x_d * norm + cx).astype(np.float32)
    map_y = (y_d * norm + cy).astype(np.float32)
    return cv2.remap(
        rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
