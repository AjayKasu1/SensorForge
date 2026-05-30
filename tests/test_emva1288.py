import numpy as np

from sensorforge.metrics.emva1288 import (
    photon_transfer,
    snr_curve,
    temporal_dark_noise,
)

SHAPE = (256, 256)


def test_temporal_dark_noise_recovers_read_noise(synth_sensor):
    # Dark, uniform: temporal dark noise should be K * read_noise_e (DN).
    K, read = 0.5, 4.0
    rng = np.random.default_rng(1)
    a, b = synth_sensor(SHAPE, photons_e=0.0, gain_dn_per_e=K, read_noise_e=read, rng=rng, n=2)
    measured = temporal_dark_noise(a, b)
    assert np.isclose(measured, K * read, rtol=0.05)


def test_photon_transfer_recovers_gain(synth_sensor):
    # Sweep illumination; slope of variance vs mean is the system gain K.
    K = 0.8
    rng = np.random.default_rng(2)
    levels = [
        tuple(synth_sensor(SHAPE, photons_e=p, gain_dn_per_e=K, read_noise_e=3.0, rng=rng, n=2))
        for p in (200, 600, 1200, 2400, 4800)
    ]
    result = photon_transfer(levels)
    assert np.isclose(result.gain_dn_per_e, K, rtol=0.05)
    assert result.mean_signal_dn.shape == (5,)


def test_snr_increases_with_signal(synth_sensor):
    K = 0.7
    rng = np.random.default_rng(3)
    dark = tuple(synth_sensor(SHAPE, photons_e=0.0, gain_dn_per_e=K, read_noise_e=3.0, rng=rng))
    levels = [
        tuple(synth_sensor(SHAPE, photons_e=p, gain_dn_per_e=K, read_noise_e=3.0, rng=rng))
        for p in (100, 1000, 5000)
    ]
    signal, snr = snr_curve(levels, dark)
    assert np.all(np.diff(snr) > 0)  # SNR grows with signal
    # Shot-limited SNR ~ sqrt(photons); roughly tracks within the sweep.
    assert snr[-1] > snr[0]
