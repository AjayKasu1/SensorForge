"""Digital color stages: normalize + black level, AWB, CCM, gamma, quantize.

These run after the sensor noise stages. AWB acts on the still-mosaiced raw
(white balance on raw, the standard ISP location); CCM and gamma act on the
demosaiced RGB; quantize is the final 8-bit encode.

Values are allowed to exceed 1.0 between AWB and gamma so highlights are not
clipped prematurely; the only hard clamps are the ADC range in ``to_digital``
and the final ``quantize_8bit``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sensorforge.isp.params import Mat3


def to_digital(electrons: NDArray, full_well_e: float, black_level: float) -> NDArray[np.float64]:
    """Electrons -> normalized digital raw in [0, 1], with the black-level
    pedestal added.

    We add the pedestal and do NOT subtract it. A real ISP subtracts black
    before white balance, but its *residual* output black floor is exactly what
    we must reproduce to match a real camera. Modeling black level as an
    uncorrected pedestal makes it a meaningful calibration knob (it moves the
    output dark floor); a correct-then-recorrect round trip would cancel out.
    """
    normalized = electrons / full_well_e
    return np.clip(normalized + black_level, 0.0, 1.0)


def apply_awb_raw(raw: NDArray, gain_r: float, gain_g: float, gain_b: float) -> NDArray:
    """Per-channel white-balance gains on the RGGB raw, by CFA position.

    No clip: gains > 1 may push values past 1.0, recovered or clamped downstream.
    """
    out = raw.copy()
    out[0::2, 0::2] *= gain_r  # R sites
    out[0::2, 1::2] *= gain_g  # G (top-right)
    out[1::2, 0::2] *= gain_g  # G (bottom-left)
    out[1::2, 1::2] *= gain_b  # B sites
    return out


def apply_ccm(rgb: NDArray, ccm: Mat3) -> NDArray:
    """Apply a 3x3 color-correction matrix: out[i] = sum_j M[i,j] * in[j]."""
    m = np.asarray(ccm, dtype=np.float64)
    return rgb @ m.T


def apply_gamma(rgb: NDArray, gamma: float) -> NDArray:
    """Encode linear -> display with a pure power law, output = signal**(1/gamma).

    A pure power approximates the IEC 61966-2-1 sRGB curve, which is actually
    piecewise (a linear toe near black). The difference is small and confined to
    deep shadows; we take the single-exponent form for one tunable knob.
    """
    return np.clip(rgb, 0.0, 1.0) ** (1.0 / gamma)


def quantize_8bit(rgb: NDArray) -> NDArray[np.uint8]:
    """Round display-referred [0, 1] RGB to 8-bit, clamping out-of-range."""
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
