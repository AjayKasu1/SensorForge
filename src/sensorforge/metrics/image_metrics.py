"""Full-reference image comparison metrics: SSIM, PSNR, CIEDE2000.

All three accept either uint8 [0, 255] or float [0, 1] RGB (or grayscale, for
SSIM/PSNR) and normalize internally, so callers can pass a webcam frame and a
sim frame without matching dtypes first.

SSIM/PSNR: Wang et al. (2004) via scikit-image. CIEDE2000: Sharma et al. (2005),
via colour-science; the sim/real frames are display-referred sRGB, so we decode
to CIE L*a*b* (D65) before the color-difference.
"""

from __future__ import annotations

import colour
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_float01(img: NDArray) -> NDArray[np.float64]:
    out = img.astype(np.float64)
    if img.dtype == np.uint8:
        out /= 255.0
    return np.clip(out, 0.0, 1.0)


def ssim(a: NDArray, b: NDArray) -> float:
    """Structural similarity in [-1, 1]; 1.0 is identical."""
    fa, fb = _to_float01(a), _to_float01(b)
    channel_axis = -1 if fa.ndim == 3 else None
    return float(structural_similarity(fa, fb, data_range=1.0, channel_axis=channel_axis))


def psnr(a: NDArray, b: NDArray) -> float:
    """Peak signal-to-noise ratio in dB; inf for identical inputs."""
    return float(peak_signal_noise_ratio(_to_float01(a), _to_float01(b), data_range=1.0))


def delta_e2000(a: NDArray, b: NDArray) -> float:
    """Mean CIEDE2000 color difference over all pixels. ~1.0 is a just-
    noticeable difference; the calibration target is < 3.
    """
    lab_a = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(_to_float01(a)))
    lab_b = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(_to_float01(b)))
    return float(np.mean(colour.delta_E(lab_a, lab_b, method="CIE 2000")))
