import numpy as np

from sensorforge.isp.noise import (
    EXPOSURE_REF_MS,
    add_dark_current,
    apply_read_noise,
    apply_shot_noise,
    integrate,
    saturate,
)


def test_integrate_is_linear_and_reference_fills_well():
    raw = np.array([0.0, 0.5, 1.0])
    out = integrate(raw, full_well_e=10000.0, exposure_ms=EXPOSURE_REF_MS)
    assert out[0] == 0.0
    assert out[2] == 10000.0  # raw=1 at reference exposure fills the well
    # Doubling exposure doubles signal.
    out2 = integrate(raw, full_well_e=10000.0, exposure_ms=2 * EXPOSURE_REF_MS)
    assert np.allclose(out2, 2 * out)


def test_dark_current_accumulates_with_exposure():
    mean = np.zeros(3)
    out = add_dark_current(mean, dark_current_e_per_s=100.0, exposure_ms=20.0)
    assert np.allclose(out, 2.0)  # 100 e-/s * 0.02 s = 2 e-


def test_shot_noise_variance_equals_mean():
    # The defining Poisson property: Var == mean. Healey & Kondepudy eq. for
    # photon shot noise. Large sample so the empirical match is tight.
    rng = np.random.default_rng(42)
    mean = np.full(200_000, 500.0)
    samples = apply_shot_noise(mean, rng)
    assert np.isclose(samples.mean(), 500.0, rtol=0.01)
    assert np.isclose(samples.var(), 500.0, rtol=0.03)


def test_read_noise_is_zero_mean_and_signal_independent():
    rng = np.random.default_rng(7)
    low = apply_read_noise(np.full(200_000, 10.0), read_noise_e=4.0, rng=rng)
    high = apply_read_noise(np.full(200_000, 5000.0), read_noise_e=4.0, rng=rng)
    # Variance from read noise alone (no shot noise here) ~ read_noise_e^2,
    # and it does not depend on the signal level.
    assert np.isclose(low.var(), 16.0, rtol=0.05)
    assert np.isclose(high.var(), 16.0, rtol=0.05)
    assert np.isclose((low - 10.0).mean(), 0.0, atol=0.05)


def test_saturation_clips_at_full_well():
    e = np.array([100.0, 9999.0, 12000.0])
    out = saturate(e, full_well_e=10000.0)
    assert np.array_equal(out, [100.0, 9999.0, 10000.0])


def test_shot_noise_is_deterministic_under_seed():
    mean = np.full(1000, 42.0)
    a = apply_shot_noise(mean, np.random.default_rng(123))
    b = apply_shot_noise(mean, np.random.default_rng(123))
    assert np.array_equal(a, b)


def test_read_noise_zero_is_passthrough():
    e = np.linspace(0, 100, 50)
    assert np.array_equal(apply_read_noise(e, 0.0, np.random.default_rng(0)), e)
