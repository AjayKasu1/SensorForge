"""Forward ISP pipeline: linear scene radiance -> 8-bit sRGB.

Composes the stages in the order fixed by ADR 004. Forward pass only; there is
no inverse ISP in v1. Pass a seeded ``rng`` for reproducible noise (the
calibration agent does this so a run is repeatable).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sensorforge.isp.bayer import mosaic_rggb
from sensorforge.isp.color import (
    apply_awb_raw,
    apply_ccm,
    apply_gamma,
    quantize_8bit,
    to_digital,
)
from sensorforge.isp.demosaic import demosaic_bilinear
from sensorforge.isp.lens import apply_distortion, apply_vignetting
from sensorforge.isp.noise import (
    add_dark_current,
    apply_read_noise,
    apply_shot_noise,
    integrate,
    saturate,
)
from sensorforge.isp.params import ISPParams


def forward(
    linear_rgb: NDArray, params: ISPParams, rng: np.random.Generator | None = None
) -> NDArray[np.uint8]:
    """Run linear RGB in [0, 1] through the full ISP to a uint8 RGB frame."""
    if rng is None:
        rng = np.random.default_rng()
    o, s, c = params.optics, params.sensor, params.color

    # Optics (linear RGB)
    x = apply_vignetting(linear_rgb, o.vignette_strength)
    x = apply_distortion(x, o.radial_k1, o.radial_k2, o.radial_k3, o.tangential_p1, o.tangential_p2)

    # Sensor (single-channel raw, electrons)
    raw = mosaic_rggb(x)
    e = integrate(raw, s.full_well_e, s.exposure_ms)
    e = add_dark_current(e, s.dark_current_e_per_s, s.exposure_ms)
    e = apply_shot_noise(e, rng)
    e = saturate(e, s.full_well_e)
    e = apply_read_noise(e, s.read_noise_e, rng)

    # Digital (raw -> RGB)
    dn = to_digital(e, s.full_well_e, s.black_level)
    dn = apply_awb_raw(dn, c.awb_gain_r, c.awb_gain_g, c.awb_gain_b)
    rgb = demosaic_bilinear(dn)
    rgb = apply_ccm(rgb, c.ccm)
    rgb = apply_gamma(rgb, c.gamma)
    return quantize_8bit(rgb)
