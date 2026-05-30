"""Shared test fixtures.

The EMVA inject-and-recover tests run against a synthetic, minimal
EMVA-linear sensor with known ground-truth parameters, deliberately decoupled
from the project's ISP so the tests validate the *estimators*, not the
pipeline.
"""

import numpy as np
import pytest


@pytest.fixture
def synth_sensor():
    """Factory producing frame stacks from a known linear sensor model.

        signal_e = photons_e * prnu_map + dark_e + dsnu_map_e
        collected = Poisson(signal_e) + Normal(0, read_noise_e)   # shot + read
        y_dn = K * collected

    Returns an (n, H, W) float64 stack in DN.
    """

    def make(
        shape,
        photons_e,
        gain_dn_per_e,
        dark_e=0.0,
        read_noise_e=0.0,
        prnu_map=None,
        dsnu_map_e=None,
        rng=None,
        n=2,
    ):
        rng = np.random.default_rng(0) if rng is None else rng
        prnu_map = np.ones(shape) if prnu_map is None else prnu_map
        dsnu_map_e = np.zeros(shape) if dsnu_map_e is None else dsnu_map_e

        mean_e = np.clip(photons_e * prnu_map + dark_e + dsnu_map_e, 0.0, None)
        frames = np.empty((n, *shape), dtype=np.float64)
        for i in range(n):
            collected = rng.poisson(mean_e).astype(np.float64)
            if read_noise_e > 0:
                collected += rng.normal(0.0, read_noise_e, size=shape)
            frames[i] = gain_dn_per_e * collected
        return frames

    return make
