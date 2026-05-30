import numpy as np

from sensorforge.isp.color import (
    apply_awb_raw,
    apply_ccm,
    apply_gamma,
    quantize_8bit,
    to_digital,
)
from sensorforge.isp.params import IDENTITY_CCM


def test_to_digital_normalizes_and_lifts_floor():
    e = np.array([0.0, 5000.0, 10000.0, 20000.0])
    dn = to_digital(e, full_well_e=10000.0, black_level=0.02)
    assert dn[0] == 0.02  # zero signal sits at the pedestal
    assert np.isclose(dn[1], 0.52)
    assert dn[2] == 1.0  # full well + pedestal clamps at ADC max
    assert dn[3] == 1.0  # over-full also clamps


def test_awb_scales_only_target_channel():
    raw = np.ones((2, 2), dtype=np.float64)
    out = apply_awb_raw(raw, gain_r=2.0, gain_g=1.0, gain_b=3.0)
    assert out[0, 0] == 2.0  # R site
    assert out[0, 1] == 1.0  # G site unchanged
    assert out[1, 0] == 1.0  # G site
    assert out[1, 1] == 3.0  # B site


def test_ccm_identity_is_passthrough():
    rgb = np.random.default_rng(0).random((4, 4, 3))
    assert np.allclose(apply_ccm(rgb, IDENTITY_CCM), rgb)


def test_ccm_gray_preserved_when_rows_sum_to_one():
    # A row-stochastic matrix leaves neutral gray neutral.
    ccm = ((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8))
    gray = np.full((3, 3, 3), 0.5)
    assert np.allclose(apply_ccm(gray, ccm), 0.5)


def test_gamma_monotonic_with_known_midpoint():
    x = np.linspace(0, 1, 256)
    y = apply_gamma(x, gamma=2.2)
    assert y[0] == 0.0 and np.isclose(y[-1], 1.0)
    assert np.all(np.diff(y) >= 0)
    assert np.isclose(apply_gamma(np.array([0.5]), 2.2)[0], 0.5 ** (1 / 2.2))


def test_gamma_clips_negative_from_ccm():
    # CCM can produce small negatives; gamma must not return NaN on them.
    out = apply_gamma(np.array([-0.1, 0.0, 0.5]), gamma=2.2)
    assert out[0] == 0.0 and not np.isnan(out).any()


def test_quantize_endpoints_and_round_trip():
    assert quantize_8bit(np.array([0.0]))[0] == 0
    assert quantize_8bit(np.array([1.0]))[0] == 255
    assert quantize_8bit(np.array([2.0]))[0] == 255  # clamps
    vals = np.arange(256) / 255.0
    assert np.array_equal(quantize_8bit(vals), np.arange(256, dtype=np.uint8))
