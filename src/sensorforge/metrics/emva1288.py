"""EMVA-1288 subset: temporal dark noise, photon-transfer gain, PRNU, DSNU.

Scope and exclusions are fixed in ADR 003. All values are in digital numbers
(DN) with electron equivalents available via the system gain K. Method
references are to EMVA-1288 Release 3.1.

The estimators take frames or frame stacks straight from a camera (or the sim
ISP). The two-image difference method is used wherever temporal noise must be
separated from fixed-pattern noise: the spatial variance of the difference of
two nominally-identical frames is twice the temporal variance, because the
fixed pattern cancels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def temporal_dark_noise(dark_a: NDArray, dark_b: NDArray) -> float:
    """Temporal dark noise sigma (DN) from two dark frames.

    EMVA-1288 two-image method: var(a - b) = 2 * temporal_variance.
    """
    diff = dark_a.astype(np.float64) - dark_b.astype(np.float64)
    return float(np.sqrt(diff.var() / 2.0))


@dataclass
class PhotonTransfer:
    """Result of the photon-transfer fit. ``gain_dn_per_e`` is the EMVA system
    gain K; ``mean_signal_dn`` / ``temporal_variance_dn2`` are the swept points
    for plotting the curve.
    """

    gain_dn_per_e: float
    dark_variance_dn2: float
    mean_signal_dn: NDArray[np.float64]
    temporal_variance_dn2: NDArray[np.float64]


def photon_transfer(levels: Sequence[tuple[NDArray, NDArray]]) -> PhotonTransfer:
    """Fit system gain K from an illumination sweep.

    Each level is a pair of frames at the same illumination. Per level we take
    the mean signal and the two-image temporal variance, then fit
    ``variance = K * mean + c``. The slope is K (DN/e-); offsetting the mean by
    the dark level only shifts the intercept, so we fit against raw mean.
    """
    means = np.empty(len(levels))
    variances = np.empty(len(levels))
    for i, (a, b) in enumerate(levels):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        means[i] = 0.5 * (a.mean() + b.mean())
        variances[i] = (a - b).var() / 2.0
    slope, intercept = np.polyfit(means, variances, 1)
    return PhotonTransfer(
        gain_dn_per_e=float(slope),
        dark_variance_dn2=float(intercept),
        mean_signal_dn=means,
        temporal_variance_dn2=variances,
    )


def _mean_temporal_variance(stack: NDArray) -> float:
    """Per-pixel temporal variance, averaged spatially."""
    return float(np.var(stack.astype(np.float64), axis=0, ddof=1).mean())


def dsnu(dark_stack: NDArray) -> float:
    """Dark signal non-uniformity (spatial std of dark FPN, DN).

    Averaging L dark frames suppresses temporal noise by L; we still subtract
    the residual temporal variance (var/L) from the spatial variance of the
    averaged image so DSNU reflects fixed pattern alone. EMVA-1288 sec. 8.
    """
    averaged = dark_stack.astype(np.float64).mean(axis=0)
    n = dark_stack.shape[0]
    s2 = averaged.var() - _mean_temporal_variance(dark_stack) / n
    return float(np.sqrt(max(s2, 0.0)))


def prnu(bright_stack: NDArray, dark_stack: NDArray) -> float:
    """Photo-response non-uniformity (percent).

    Spatial std of the dark-subtracted mean response over its spatial mean,
    with the residual temporal variance removed. EMVA-1288 sec. 8.
    """
    response = bright_stack.astype(np.float64).mean(axis=0) - dark_stack.astype(np.float64).mean(
        axis=0
    )
    nb, nd = bright_stack.shape[0], dark_stack.shape[0]
    residual = _mean_temporal_variance(bright_stack) / nb + _mean_temporal_variance(dark_stack) / nd
    s2 = response.var() - residual
    return float(100.0 * np.sqrt(max(s2, 0.0)) / response.mean())


def snr_curve(
    levels: Sequence[tuple[NDArray, NDArray]], dark: tuple[NDArray, NDArray]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """SNR vs signal across the sweep.

    Returns (signal_dn, snr) where signal is dark-subtracted mean and SNR is
    signal / temporal_sigma, per level.
    """
    dark_mean = 0.5 * (dark[0].astype(np.float64).mean() + dark[1].astype(np.float64).mean())
    signal = np.empty(len(levels))
    snr = np.empty(len(levels))
    for i, (a, b) in enumerate(levels):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        signal[i] = 0.5 * (a.mean() + b.mean()) - dark_mean
        sigma = np.sqrt((a - b).var() / 2.0)
        snr[i] = signal[i] / sigma if sigma > 0 else np.inf
    return signal, snr
