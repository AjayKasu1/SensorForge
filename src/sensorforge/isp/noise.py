"""Sensor noise stages, in electrons.

The signal chain on the single-channel Bayer raw (ADR 004):

    integrate -> + dark current -> shot noise (Poisson) -> well saturation
    -> read noise (Gaussian)

Total collected charge is Poisson with mean (photo + dark) electrons; that one
draw captures both photon shot noise and dark shot noise. Read noise is a
signal-independent Gaussian added at readout. Healey & Kondepudy (1994),
"Radiometric CCD Camera Calibration and Noise Estimation", and the EMVA-1288
linear sensor model.

All randomness comes from a caller-supplied ``np.random.Generator`` so runs are
reproducible; no global ``np.random`` state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Exposure at which a raw value of 1.0 just fills the well. Ties the otherwise
# free radiance->electrons scale to a physical exposure, so full_well_e and
# exposure_ms are both meaningful, tradeable knobs.
EXPOSURE_REF_MS = 10.0


def integrate(raw: NDArray, full_well_e: float, exposure_ms: float) -> NDArray:
    """Linear raw in [0, 1] -> mean photoelectrons."""
    return raw * full_well_e * (exposure_ms / EXPOSURE_REF_MS)


def add_dark_current(mean_e: NDArray, dark_current_e_per_s: float, exposure_ms: float) -> NDArray:
    """Add thermally generated electrons accumulated over the exposure."""
    mean_dark = dark_current_e_per_s * (exposure_ms / 1000.0)
    return mean_e + mean_dark


def apply_shot_noise(mean_e: NDArray, rng: np.random.Generator) -> NDArray:
    """Sample actual collected charge ~ Poisson(mean). Captures photon and dark
    shot noise in one draw.
    """
    return rng.poisson(mean_e).astype(np.float64)


def saturate(electrons: NDArray, full_well_e: float) -> NDArray:
    """Clip collected charge at the well capacity (highlight saturation)."""
    return np.minimum(electrons, full_well_e)


def apply_read_noise(electrons: NDArray, read_noise_e: float, rng: np.random.Generator) -> NDArray:
    """Add zero-mean Gaussian readout noise, in electrons."""
    if read_noise_e == 0.0:
        return electrons
    return electrons + rng.normal(0.0, read_noise_e, size=electrons.shape)
